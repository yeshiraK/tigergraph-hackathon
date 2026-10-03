"""Unit tests for production semantic unit detection, chunking, and validation."""

import unittest

from tgh.domain.models import Chunk
from tgh.ingestion.chunker import SemanticChunker
from tgh.ingestion.semantic_units import (
    SemanticUnitType,
    extract_semantic_units,
    is_heading,
    is_infobox,
    is_table,
)
from tgh.ingestion.tokenizer import ApproximateCharTokenizer
from tgh.ingestion.validator import ChunkValidationError, validate_chunks


class TestSemanticUnitDetection(unittest.TestCase):
    """Unit tests for semantic unit structural predicates."""

    def test_infobox_detection(self) -> None:
        """Verify infobox detection matches [infobox (case-insensitive)."""
        self.assertTrue(is_infobox("[Infobox Olympic event]\n  event: Sprint"))
        self.assertTrue(is_infobox("[infobox film]\n  title: Movie"))
        self.assertTrue(is_infobox("  [Infobox person]\n  name: Athlete"))
        self.assertFalse(is_infobox("Regular text about [Infobox]"))
        self.assertFalse(is_infobox("Infobox without brackets"))
        self.assertFalse(is_infobox("[…]"))

    def test_table_detection(self) -> None:
        """Verify table detection matches tables and rejects captions."""
        # Multi-line pipe table
        table_text = (
            "Rank | Athlete | Country\n"
            "1 | Jane Doe | USA\n"
            "2 | John Smith | GBR"
        )
        self.assertTrue(is_table(table_text))

        # Explicit 'Results table' header
        table_explicit = "Results table\nDate | Round\nMon | Final"
        self.assertTrue(is_table(table_explicit))

        # Single-line multi-column row
        self.assertTrue(is_table("Year | Category | Result | Rank"))

        # Prose with image caption remnants must be rejected
        prose_with_caption = (
            "Plot\n"
            "On February 1, Phil awakens in Hope, Arkansas|left.\n"
            "He begins his daily routine."
        )
        self.assertFalse(is_table(prose_with_caption))

        # Single-line prose ending with period must be rejected
        prose_single = "He was filmed near Bastrop, Texas|alt=A two-lane highway."
        self.assertFalse(is_table(prose_single))

    def test_heading_detection(self) -> None:
        """Verify heading detection rules and false-positive rejection."""
        # Valid headings with following block
        self.assertTrue(is_heading("Results", has_following_block=True))
        self.assertTrue(is_heading("Heat 1", has_following_block=True))
        self.assertTrue(is_heading("Competition format", has_following_block=True))

        # Rejected: no following block (last block in document)
        self.assertFalse(is_heading("Results", has_following_block=False))

        # Rejected: ends with punctuation
        self.assertFalse(is_heading("Results.", has_following_block=True))
        self.assertFalse(is_heading("Heat 1:", has_following_block=True))

        # Rejected: horizontal rule
        self.assertFalse(is_heading("----", has_following_block=True))

        # Rejected: numeric
        self.assertFalse(is_heading("1996", has_following_block=True))

        # Rejected: contains pipe
        self.assertFalse(is_heading("Col 1 | Col 2", has_following_block=True))

        # Rejected: bullet point
        self.assertFalse(is_heading("- Athlete name", has_following_block=True))

        # Rejected: too long (>80 chars)
        long_heading = "A" * 81
        self.assertFalse(is_heading(long_heading, has_following_block=True))

    def test_heading_attachment(self) -> None:
        """Verify that heading blocks attach to the following block."""
        doc_text = (
            "Results\n\nRank | Name\n1 | Alice\n\n"
            "Plot\n\nAlice won the gold medal."
        )
        units = extract_semantic_units(doc_text, attach_headings=True)

        # Should produce 2 units (Heading+Table, Heading+Prose) instead of 4
        self.assertEqual(len(units), 2)
        self.assertTrue(units[0].text.startswith("Results\n\nRank | Name"))
        self.assertEqual(units[0].unit_type, SemanticUnitType.TABLE)
        self.assertTrue(units[1].text.startswith("Plot\n\nAlice won"))
        self.assertEqual(units[1].unit_type, SemanticUnitType.PROSE)

        # Verify exact reconstruction
        self.assertEqual("".join(u.text for u in units), doc_text)


class TestSemanticChunker(unittest.TestCase):
    """Unit tests for chunking behavior, metadata, and invariant validation."""

    def setUp(self) -> None:
        # Use 4 chars per token approximation
        self.tokenizer = ApproximateCharTokenizer(chars_per_token=4.0)

    def test_deterministic_chunk_ids_and_ordering(self) -> None:
        """Verify chunk IDs are strictly deterministic and sequential."""
        doc_id = "test_doc_001"
        text = "Block 1 content.\n\nBlock 2 content.\n\nBlock 3 content."
        chunker = SemanticChunker(target_tokens=10, tokenizer=self.tokenizer)
        chunks = chunker.chunk_document(doc_id, text)

        self.assertGreater(len(chunks), 1)
        for idx, chunk in enumerate(chunks):
            self.assertEqual(chunk.document_id, doc_id)
            self.assertEqual(chunk.chunk_index, idx)
            self.assertEqual(chunk.chunk_id, f"{doc_id}#c{idx:04d}")

    def test_exact_reconstruction_and_offsets(self) -> None:
        """Verify source_start and source_end cleanly slice the exact text."""
        doc_id = "test_doc_002"
        text = (
            "[Infobox Olympic event]\n  event: 100m\n\n"
            "The 100m sprint was held at the Olympic Stadium.\n\n"
            "Results table\nRank | Athlete\n1 | Bolt\n2 | Gatlin"
        )
        chunker = SemanticChunker(target_tokens=20, tokenizer=self.tokenizer)
        chunks = chunker.chunk_document(doc_id, text)

        # Validate string reconstruction
        reconstructed = "".join(c.text for c in chunks)
        self.assertEqual(reconstructed, text)

        # Validate offsets
        for chunk in chunks:
            self.assertEqual(text[chunk.source_start : chunk.source_end], chunk.text)

    def test_oversized_unit_fallback_splitting(self) -> None:
        """Verify oversized unit falls back to line splitting without losing text."""
        doc_id = "test_oversized"
        # 10 lines of 20 chars = 200 chars ~ 50 tokens
        lines = [f"Table row {i:02d} | Score: {i * 10} points\n" for i in range(10)]
        table_text = "".join(lines)

        # Target size 15 tokens (~60 chars), forcing table to split
        chunker = SemanticChunker(target_tokens=15, tokenizer=self.tokenizer)
        chunks = chunker.chunk_document(doc_id, table_text)

        self.assertGreater(len(chunks), 1)
        self.assertEqual("".join(c.text for c in chunks), table_text)
        for chunk in chunks:
            self.assertEqual(
                table_text[chunk.source_start : chunk.source_end], chunk.text
            )

    def test_validator_fails_loudly_on_gap(self) -> None:
        """Verify validate_chunks raises error when offsets have a gap."""
        doc_id = "bad_doc"
        original = "Hello World"
        invalid_chunks = [
            Chunk(
                document_id=doc_id,
                chunk_id=f"{doc_id}#c0000",
                chunk_index=0,
                source_start=0,
                source_end=5,
                text="Hello",
                token_count=2,
            ),
            Chunk(
                document_id=doc_id,
                chunk_id=f"{doc_id}#c0001",
                chunk_index=1,
                source_start=6,  # Gap: skipped space at index 5
                source_end=11,
                text="World",
                token_count=2,
            ),
        ]
        with self.assertRaises(ChunkValidationError):
            validate_chunks(doc_id, original, invalid_chunks)


if __name__ == "__main__":
    unittest.main()
