"""Validation module verifying deterministic invariants for produced chunks."""

from tgh.domain.models import Chunk


class ChunkValidationError(ValueError):
    """Raised when chunk validation invariants are violated."""

    pass


def validate_chunks(
    document_id: str,
    original_text: str,
    chunks: list[Chunk],
) -> None:
    """Validate all deterministic invariants for chunks produced from a document.

    Checks:
    - Every source document reconstructs exactly from its produced chunks.
    - No characters are lost.
    - No characters are duplicated.
    - Chunk indexes are strictly contiguous starting from 0.
    - Chunk IDs are deterministic and match document_id + chunk_index.
    - Chunk order matches source order without gaps or overlaps.
    - No chunk is empty.
    - Metadata ranges (source_start, source_end) match original document text.

    Raises:
        ChunkValidationError: If any invariant is violated.
    """
    if not original_text:
        if chunks:
            raise ChunkValidationError(
                f"Document '{document_id}' has empty text but produced "
                f"{len(chunks)} chunks"
            )
        return

    if not chunks:
        raise ChunkValidationError(
            f"Document '{document_id}' has non-empty text ({len(original_text)} chars) "
            "but produced 0 chunks"
        )

    expected_reconstruction: list[str] = []
    prev_end = 0

    for idx, chunk in enumerate(chunks):
        # 1. Non-empty text check
        if not chunk.text:
            raise ChunkValidationError(
                f"Document '{document_id}': Chunk at index {idx} has empty text"
            )

        # 2. Document ID check
        if chunk.document_id != document_id:
            raise ChunkValidationError(
                f"Document ID mismatch: expected '{document_id}', "
                f"got '{chunk.document_id}'"
            )

        # 3. Contiguous chunk index check starting from 0
        if chunk.chunk_index != idx:
            raise ChunkValidationError(
                f"Document '{document_id}': Expected chunk_index {idx}, "
                f"got {chunk.chunk_index}"
            )

        # 4. Deterministic chunk ID check
        expected_chunk_id = f"{document_id}#c{idx:04d}"
        if chunk.chunk_id != expected_chunk_id:
            raise ChunkValidationError(
                f"Document '{document_id}': Expected chunk_id '{expected_chunk_id}', "
                f"got '{chunk.chunk_id}'"
            )

        # 5. Metadata range sanity
        if chunk.source_start < 0:
            raise ChunkValidationError(
                f"Document '{document_id}' chunk {idx}: "
                f"source_start ({chunk.source_start}) < 0"
            )
        if chunk.source_end > len(original_text):
            raise ChunkValidationError(
                f"Document '{document_id}' chunk {idx}: "
                f"source_end ({chunk.source_end}) "
                f"exceeds text length ({len(original_text)})"
            )
        if chunk.source_start >= chunk.source_end:
            raise ChunkValidationError(
                f"Document '{document_id}' chunk {idx}: "
                f"source_start ({chunk.source_start}) "
                f">= source_end ({chunk.source_end})"
            )

        # 6. Contiguity check (no gaps, no overlaps)
        if chunk.source_start != prev_end:
            raise ChunkValidationError(
                f"Document '{document_id}' chunk {idx}: Gap or overlap detected. "
                f"Previous end was {prev_end}, but start is {chunk.source_start}"
            )
        prev_end = chunk.source_end

        # 7. Substring exact match against source text
        source_slice = original_text[chunk.source_start : chunk.source_end]
        if source_slice != chunk.text:
            raise ChunkValidationError(
                f"Document '{document_id}' chunk {idx}: Text at offsets "
                f"[{chunk.source_start}:{chunk.source_end}] does not match chunk text"
            )

        expected_reconstruction.append(chunk.text)

    # 8. Full coverage check
    if prev_end != len(original_text):
        raise ChunkValidationError(
            f"Document '{document_id}': Final chunk end ({prev_end}) does not cover "
            f"full text length ({len(original_text)})"
        )

    # 9. Exact string reconstruction
    reconstructed_text = "".join(expected_reconstruction)
    if reconstructed_text != original_text:
        raise ChunkValidationError(
            f"Document '{document_id}': "
            "Concatenated chunk text does not match original text"
        )
