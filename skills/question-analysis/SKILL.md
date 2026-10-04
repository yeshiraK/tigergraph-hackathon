---
name: question-analysis
description: Classifies question intent, isolates entity references, temporal constraints, and relational requirements for Olympic GraphRAG queries.
allowed-tools:
  - tigergraph__get_nodes
  - tigergraph__get_edges
---

# Question Analysis Skill

## Purpose
This skill provides procedural guidance to analyze incoming natural language questions against the OlympicGraphRAG domain. It identifies:
1. Target entity names (athletes, teams, venues, sports)
2. Temporal constraints (specific Olympic years, e.g. 2016, 2012, 1988)
3. Relational requirements (multi-hop traversal from venue to event, or participant to country)
4. Query intent classification:
   - `lookup`: direct fact lookup
   - `temporal`: event immediately preceding/following or specific year
   - `multi_hop`: indirect relationship requiring bridge traversal (e.g. "held at <venue> in <year>")
   - `aggregation`: counting distinct participants, nations, or events
   - `superlative`: highest/lowest competitor count

## Workflow
1. **Normalized Entity Extraction**:
   - Inspect question for venue patterns: `held at <venue_name>`.
   - Inspect question for athlete/participant mentions: `Who won the gold medal in...`.
   - Extract 4-digit years matching `\b(19\d\d|20\d\d)\b`.
2. **Constraint Formulation**:
   - If venue is detected, map to standard venue identifier candidate.
   - If nation count is requested, flag `aggregation_country` requirement.
3. **Output Structure**:
   - Return structured intent, detected entity cues, year constraint, and recommended downstream focus.
