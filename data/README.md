# Data Store & Corpus

This directory holds the raw corpus, processed chunks/entities, and evaluation benchmark datasets.

## Structure
- `data/raw/`: Original source documents and raw benchmark input files.
- `data/processed/`: Standardized text chunks, entity representations, and intermediate staging data.
- `data/benchmarks/`: Evaluation question-answer pairs and benchmark ground truth sets.

## Critical Constraint: Corpus Ground Truth
The corpus housed here is the **sole source of truth** for all benchmark questions.
External web search and model pre-trained parametric knowledge must not be used to answer evaluation questions.
Raw and processed corpus data are excluded from version control via `.gitignore`.
