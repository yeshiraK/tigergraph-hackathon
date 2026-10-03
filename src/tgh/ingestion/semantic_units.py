"""Semantic unit detection and classification for the L1 Ingestion pipeline.

Implements validated structural predicates for infoboxes, tables, headings,
and paragraph boundaries adhering to the production chunking specification.
"""

import re
from dataclasses import dataclass
from enum import StrEnum


class SemanticUnitType(StrEnum):
    """Classification of semantic content blocks."""

    INFOBOX = "infobox"
    TABLE = "table"
    HEADING = "heading"
    PROSE = "prose"


@dataclass(frozen=True)
class SemanticUnit:
    """A structural semantic block with text and document offset boundaries.

    Attributes:
        text: Exact source text content including trailing block delimiter.
        unit_type: Structural classification (infobox, table, heading, prose).
        source_start: Character offset start in the original document.
        source_end: Character offset end in the original document.
    """

    text: str
    unit_type: SemanticUnitType
    source_start: int
    source_end: int


def is_infobox(block: str) -> bool:
    """Determine whether a block is an Infobox.

    Specification: A block is an infobox when:
        block.strip().lower().startswith("[infobox")
    """
    stripped = block.strip()
    return stripped.lower().startswith("[infobox")


def _is_pipe_table_line(line: str) -> bool:
    """Check if a line contains table column delimiters without caption markers."""
    if "|" not in line:
        return False
    # Strip common MediaWiki image caption artifacts before checking for table pipes
    cleaned = re.sub(
        r"\|(?:alt=[^|]*|left\b|right\b|thumb\b|upright\b|\d*x?\d*px\b)",
        "",
        line,
        flags=re.IGNORECASE,
    )
    return "|" in cleaned


def is_table(block: str) -> bool:
    """Determine whether a block is a pipe-delimited table.

    Specification:
    - Explicit 'Results table' followed by data rows => table.
    - Otherwise, for multi-line blocks:
      - at least 2 non-empty lines contain '|'
      - pipe-containing lines are at least 50% of all non-empty lines
      - exclude obvious caption markers (|alt=, |left, |right, |thumb, |upright, |px)
    - For single-line blocks:
      - require at least 2 pipe delimiters
      - reject if the line ends with sentence-ending punctuation ('.', '!', '?')
    """
    lines = [line.strip() for line in block.split("\n") if line.strip()]
    if not lines:
        return False

    # Explicit header convention
    if lines[0].lower().startswith("results table") and len(lines) > 1:
        return True

    # Single-line tabular row check
    if len(lines) == 1:
        l0 = lines[0]
        cleaned = re.sub(
            r"\|(?:alt=[^|]*|left\b|right\b|thumb\b|upright\b|\d*x?\d*px\b)",
            "",
            l0,
            flags=re.IGNORECASE,
        )
        return cleaned.count("|") >= 2 and not cleaned.strip().endswith(
            (".", "!", "?")
        )

    # Multi-line table check
    pipe_lines = [line for line in lines if _is_pipe_table_line(line)]
    return len(pipe_lines) >= 2 and (len(pipe_lines) / len(lines) >= 0.5)


def is_heading(block: str, has_following_block: bool = True) -> bool:
    """Determine whether a block is a structural heading.

    Specification:
    - exactly 1 non-empty line
    - stripped length is between 2 and 80 characters
    - does not contain '|'
    - does not start with '-', '*', or '•'
    - does not match ^[-=_*]{3,}$
    - is not purely numeric
    - does not end with '.', '!', '?', or ':'
    - must have a following semantic block in the same document
    """
    if not has_following_block:
        return False

    non_empty_lines = [line for line in block.split("\n") if line.strip()]
    if len(non_empty_lines) != 1:
        return False

    stripped = non_empty_lines[0].strip()
    if not (2 <= len(stripped) <= 80):
        return False

    if "|" in stripped:
        return False

    if stripped.startswith(("-", "*", "•")):
        return False

    if re.match(r"^[-=_*]{3,}$", stripped):
        return False

    if stripped.isdigit():
        return False

    if stripped.endswith((".", "!", "?", ":")):
        return False

    return True


def extract_semantic_units(
    text: str, attach_headings: bool = True
) -> list[SemanticUnit]:
    """Extract primary semantic units by splitting text at blank line runs (\\n\\n+).

    Preserves exact source content, whitespace, and character offset ranges.
    When attach_headings is True, detected headings are combined with their
    immediately following block.
    """
    if not text:
        return []

    parts = re.split(r"(\n\n+)", text)
    raw_units: list[tuple[str, int, int]] = []
    current_offset = 0

    start_idx = 0
    leading_delim = ""
    if parts and parts[0] == "" and len(parts) > 1:
        leading_delim = parts[1]
        start_idx = 2

    for i in range(start_idx, len(parts), 2):
        block_text = parts[i]
        delimiter = parts[i + 1] if i + 1 < len(parts) else ""
        content = (
            (leading_delim + block_text + delimiter)
            if i == start_idx
            else (block_text + delimiter)
        )
        if content:
            start = current_offset
            end = start + len(content)
            raw_units.append((content, start, end))
            current_offset = end

    # Initial classification
    classified_units: list[SemanticUnit] = []
    total = len(raw_units)
    for idx, (content, start, end) in enumerate(raw_units):
        has_following = idx + 1 < total
        if is_infobox(content):
            unit_type = SemanticUnitType.INFOBOX
        elif is_table(content):
            unit_type = SemanticUnitType.TABLE
        elif is_heading(content, has_following_block=has_following):
            unit_type = SemanticUnitType.HEADING
        else:
            unit_type = SemanticUnitType.PROSE

        classified_units.append(
            SemanticUnit(
                text=content,
                unit_type=unit_type,
                source_start=start,
                source_end=end,
            )
        )

    if not attach_headings:
        return classified_units

    # Attach heading(s) to following block
    attached_units: list[SemanticUnit] = []
    i = 0
    while i < len(classified_units):
        unit = classified_units[i]
        if unit.unit_type == SemanticUnitType.HEADING and (
            i + 1 < len(classified_units)
        ):
            # Accumulate one or more consecutive headings
            headings_text = [unit.text]
            start_offset = unit.source_start
            j = i + 1
            while (
                j < len(classified_units)
                and classified_units[j].unit_type == SemanticUnitType.HEADING
                and j + 1 < len(classified_units)
            ):
                headings_text.append(classified_units[j].text)
                j += 1

            body_unit = classified_units[j]
            combined_text = "".join(headings_text) + body_unit.text
            attached_units.append(
                SemanticUnit(
                    text=combined_text,
                    unit_type=body_unit.unit_type,
                    source_start=start_offset,
                    source_end=body_unit.source_end,
                )
            )
            i = j + 1
        else:
            attached_units.append(unit)
            i += 1

    return attached_units
