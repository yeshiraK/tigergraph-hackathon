"""Production structure-aware semantic chunker for L1 Ingestion.

Implements lossless, deterministic chunking respecting Infobox, Table,
and Heading semantic boundaries with exact source reconstruction.
"""

import re
from dataclasses import dataclass

from tgh.domain.models import Chunk
from tgh.ingestion.semantic_units import (
    SemanticUnit,
    SemanticUnitType,
    extract_semantic_units,
)
from tgh.ingestion.tokenizer import ApproximateCharTokenizer, Tokenizer
from tgh.ingestion.validator import validate_chunks


@dataclass
class ChunkingStats:
    """Statistics recorded during chunking."""

    total_chunks: int = 0
    oversized_units_split: int = 0
    infobox_units_split: int = 0
    table_units_split: int = 0
    chunks_exceeding_target: int = 0


class SemanticChunker:
    """Structure-aware deterministic document chunker.

    Attributes:
        target_tokens: Target maximum tokens per chunk (default 768).
        tokenizer: Tokenizer instance for counting tokens.
        attach_headings: Whether to attach heading blocks to following content.
    """

    def __init__(
        self,
        target_tokens: int = 768,
        tokenizer: Tokenizer | None = None,
        attach_headings: bool = True,
    ) -> None:
        if target_tokens <= 0:
            raise ValueError("target_tokens must be positive")
        self.target_tokens = target_tokens
        self.tokenizer = tokenizer or ApproximateCharTokenizer()
        self.attach_headings = attach_headings

    def _split_oversized_unit(
        self, unit: SemanticUnit
    ) -> list[SemanticUnit]:
        """Deterministically split an oversized semantic unit preserving exact text.

        Hierarchy:
        1. Line boundaries (\\n+)
        2. Whitespace boundaries (\\s+)
        3. Character-level slicing (final fallback)
        """
        # Determine character budget based on tokenizer or target ratio
        if isinstance(self.tokenizer, ApproximateCharTokenizer):
            target_chars = int(self.target_tokens * self.tokenizer.chars_per_token)
        else:
            # Fallback estimation for target chars
            target_chars = self.target_tokens * 4

        line_parts = re.split(r"(\n+)", unit.text)
        lines: list[str] = []
        for i in range(0, len(line_parts), 2):
            l_str = line_parts[i]
            d_str = line_parts[i + 1] if i + 1 < len(line_parts) else ""
            if l_str or d_str:
                lines.append(l_str + d_str)

        sub_texts: list[str] = []
        curr_lines: list[str] = []
        curr_chars = 0

        for line_item in lines:
            line_len = len(line_item)
            if line_len > target_chars:
                if curr_lines:
                    sub_texts.append("".join(curr_lines))
                    curr_lines = []
                    curr_chars = 0
                # Split line by whitespace boundaries
                word_parts = re.split(r"(\s+)", line_item)
                words: list[str] = []
                for j in range(0, len(word_parts), 2):
                    w_str = word_parts[j]
                    s_str = word_parts[j + 1] if j + 1 < len(word_parts) else ""
                    if w_str or s_str:
                        words.append(w_str + s_str)

                curr_words: list[str] = []
                curr_word_chars = 0
                for word_item in words:
                    w_len = len(word_item)
                    if w_len > target_chars:
                        if curr_words:
                            sub_texts.append("".join(curr_words))
                            curr_words = []
                            curr_word_chars = 0
                        # Final fallback: character-level hard slicing
                        for k in range(0, w_len, target_chars):
                            sub_texts.append(word_item[k : k + target_chars])
                    elif curr_word_chars + w_len > target_chars:
                        sub_texts.append("".join(curr_words))
                        curr_words = [word_item]
                        curr_word_chars = w_len
                    else:
                        curr_words.append(word_item)
                        curr_word_chars += w_len

                if curr_words:
                    sub_texts.append("".join(curr_words))
            elif curr_chars + line_len > target_chars:
                sub_texts.append("".join(curr_lines))
                curr_lines = [line_item]
                curr_chars = line_len
            else:
                curr_lines.append(line_item)
                curr_chars += line_len

        if curr_lines:
            sub_texts.append("".join(curr_lines))

        # Convert sub_texts to SemanticUnit objects with exact contiguous offsets
        sub_units: list[SemanticUnit] = []
        offset = unit.source_start
        for st in sub_texts:
            end_offset = offset + len(st)
            sub_units.append(
                SemanticUnit(
                    text=st,
                    unit_type=unit.unit_type,
                    source_start=offset,
                    source_end=end_offset,
                )
            )
            offset = end_offset

        return sub_units

    def chunk_document(
        self,
        document_id: str,
        text: str,
        stats: ChunkingStats | None = None,
    ) -> list[Chunk]:
        """Chunk a document text into validated deterministic Chunk objects.

        Args:
            document_id: Identifier of the source document.
            text: Raw document text string.
            stats: Optional stats collector.

        Returns:
            List of validated Chunk instances preserving exact source text.
        """
        if not text:
            return []

        # 1. Extract semantic units respecting \n\n+ and heading attachment
        raw_units = extract_semantic_units(
            text, attach_headings=self.attach_headings
        )

        # 2. Check and handle oversized semantic units
        prepared_units: list[SemanticUnit] = []
        for unit in raw_units:
            unit_tokens = self.tokenizer.count_tokens(unit.text)
            if unit_tokens > self.target_tokens:
                if stats is not None:
                    stats.oversized_units_split += 1
                    if unit.unit_type == SemanticUnitType.INFOBOX:
                        stats.infobox_units_split += 1
                    elif unit.unit_type == SemanticUnitType.TABLE:
                        stats.table_units_split += 1
                sub_units = self._split_oversized_unit(unit)
                prepared_units.extend(sub_units)
            else:
                prepared_units.append(unit)

        # 3. Group consecutive units up to target_tokens
        chunks: list[Chunk] = []
        curr_units: list[SemanticUnit] = []
        curr_tokens = 0
        chunk_idx = 0

        for unit in prepared_units:
            unit_tokens = self.tokenizer.count_tokens(unit.text)
            if curr_units and (curr_tokens + unit_tokens > self.target_tokens):
                # Flush current chunk
                chunk_text = "".join(u.text for u in curr_units)
                start_offset = curr_units[0].source_start
                end_offset = curr_units[-1].source_end
                t_count = self.tokenizer.count_tokens(chunk_text)
                if t_count > self.target_tokens and stats is not None:
                    stats.chunks_exceeding_target += 1

                chunks.append(
                    Chunk(
                        document_id=document_id,
                        chunk_id=f"{document_id}#c{chunk_idx:04d}",
                        chunk_index=chunk_idx,
                        source_start=start_offset,
                        source_end=end_offset,
                        text=chunk_text,
                        token_count=t_count,
                    )
                )
                chunk_idx += 1
                curr_units = [unit]
                curr_tokens = unit_tokens
            else:
                curr_units.append(unit)
                curr_tokens += unit_tokens

        # Flush final chunk
        if curr_units:
            chunk_text = "".join(u.text for u in curr_units)
            start_offset = curr_units[0].source_start
            end_offset = curr_units[-1].source_end
            t_count = self.tokenizer.count_tokens(chunk_text)
            if t_count > self.target_tokens and stats is not None:
                stats.chunks_exceeding_target += 1

            chunks.append(
                Chunk(
                    document_id=document_id,
                    chunk_id=f"{document_id}#c{chunk_idx:04d}",
                    chunk_index=chunk_idx,
                    source_start=start_offset,
                    source_end=end_offset,
                    text=chunk_text,
                    token_count=t_count,
                )
            )

        if stats is not None:
            stats.total_chunks += len(chunks)

        # 4. Rigorous invariant validation
        validate_chunks(document_id, text, chunks)

        return chunks
