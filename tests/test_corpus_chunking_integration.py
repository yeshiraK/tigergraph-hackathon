"""Integration test running production semantic chunking against data/raw/corpus.jsonl.

Validates exact reconstruction and invariants across all 2,951 documents
without generating embeddings or connecting to TigerGraph.
"""

import json
import statistics
import unittest
from pathlib import Path

from tgh.ingestion.chunker import ChunkingStats, SemanticChunker


class TestCorpusChunkingIntegration(unittest.TestCase):
    """Integration test suite executing chunking across the entire verified corpus."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.repo_root = Path(__file__).resolve().parent.parent
        cls.corpus_file = cls.repo_root / "data" / "raw" / "corpus.jsonl"
        cls.target_tokens = 768

    def test_full_corpus_chunking_and_validation(self) -> None:
        """Process all 2,951 corpus documents and verify 100% exact reconstruction."""
        self.assertTrue(
            self.corpus_file.is_file(),
            f"Corpus file must exist at {self.corpus_file}",
        )

        chunker = SemanticChunker(target_tokens=self.target_tokens)
        stats = ChunkingStats()
        chunks_per_doc: list[int] = []
        doc_count = 0

        with self.corpus_file.open("r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue
                record = json.loads(stripped)
                doc_id = record["doc_id"]
                text = record["text"]

                chunks = chunker.chunk_document(doc_id, text, stats=stats)
                chunks_per_doc.append(len(chunks))
                doc_count += 1

                # Double-check full reconstruction directly
                reconstructed = "".join(c.text for c in chunks)
                self.assertEqual(
                    reconstructed,
                    text,
                    f"Reconstruction failed for doc_id {doc_id}",
                )

        self.assertEqual(
            doc_count, 2951, f"Expected 2951 documents, got {doc_count}"
        )
        self.assertGreater(stats.total_chunks, 0)

        mean_chunks = statistics.mean(chunks_per_doc)
        min_chunks = min(chunks_per_doc)
        max_chunks = max(chunks_per_doc)

        print("\n--- Full Corpus Chunking Integration Results ---")
        print(f"Total documents processed: {doc_count}")
        print(f"Target chunk size: {self.target_tokens} tokens")
        print(f"Total chunks produced: {stats.total_chunks}")
        print(
            f"Chunks per document: min={min_chunks}, max={max_chunks}, "
            f"mean={mean_chunks:.2f}"
        )
        print(f"Chunks exceeding target: {stats.chunks_exceeding_target}")
        print(f"Oversized semantic units split: {stats.oversized_units_split}")
        print(f"Infobox blocks split: {stats.infobox_units_split}")
        print(f"Table blocks split: {stats.table_units_split}")
        print("Exact reconstruction: 100.0% (0 errors across 2,951 documents)")


if __name__ == "__main__":
    unittest.main()
