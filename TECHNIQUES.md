# Project Techniques and Design Rationale

## 1. Introduction
This document details the key software engineering and data science techniques employed in this project. It explains the "what," "why," and "benefit" of each major design decision.

## 2. Core Pipeline Architecture
*   **Technique**: Incremental, Per-Document Processing (`knowledge_graph_builder.py`)
*   **Rationale**: The initial approach of loading all documents into memory was not scalable and led to high RAM/VRAM consumption.
*   **Benefit**: This ensures the memory footprint remains low and constant, allowing the system to process a large number of documents without crashing.

## 3. Data Extraction and Enrichment
*   **Technique**: Hybrid Entity Extraction with Enhanced Step Parsing (`entity_extractor.py`)
*   **Rationale**: Relying on a single method (either pure LLM or pure regex) is brittle. LLMs can hallucinate or miss structured data, while regex can't handle ambiguity. Additionally, robustly extracting multi-paragraph steps is crucial for complete procedure descriptions.
*   **Benefit**: Combining LLM-based extraction for semantic understanding with pattern-based matching for known entities (e.g., `AMF`, `SMF` from a whitelist) provides a robust and accurate result. This is further improved by a final validation step against the text. The enhanced step parsing ensures that even complex, multi-paragraph steps are correctly identified and extracted, leading to more complete and accurate procedure descriptions.

*   **Technique**: LLM-Powered Step Generation (`entity_extractor.py`)
*   **Rationale**: Many procedures in the source documents lack explicit, machine-readable step-by-step instructions. The initial fallback of using hardcoded steps was inaccurate.
*   **Benefit**: Using an LLM to generate plausible, context-aware steps for these cases results in a much more complete and accurate knowledge graph.

## Search Engine

The search engine is a critical component for navigating the vast 3GPP specifications. It employs a **Full Semantic Search Model** that leverages multiple semantic signals for highly accurate conceptual understanding.

### Architecture:

The search process relies entirely on semantic embeddings:

1.  **Multi-Field Semantic Embeddings**: For each procedure, the system generates and utilizes three separate semantic embeddings:
    *   An embedding for the procedure's `title`.
    *   An embedding for the `parent_title` of its containing section.
    *   An embedding for the full `description` (concatenated steps).
    These embeddings are generated using a state-of-the-art Sentence Transformer model (`Qwen/Qwen3-Embedding-8B`).

2.  **Optimized Weighted Scoring**: The final relevance score for each procedure is a weighted sum of the cosine similarities between the user's query embedding and the three procedure embeddings.
    *   The weights (`W_SEMANTIC_TITLE: 1.7`, `W_SEMANTIC_PARENT: 1.7`, `W_SEMANTIC_DESC: 16`) are defined in `codebase/config.py` and are finely tuned to achieve optimal relevance and ranking for golden queries.
    *   A similarity threshold is applied to filter out less relevant results.

### Optimization:

*   **Multi-GPU Support**: The embedding model is loaded onto a separate GPU (if available) to parallelize processing and reduce latency.
*   **Adaptive Embedding**: For very long procedure descriptions, the system intelligently chunks the text into smaller, overlapping segments, generates embeddings for each, and then averages them. This ensures that the full semantic context is captured without exceeding the embedding model's token limit or causing `CUDA out of memory` errors.

## 5. Database and Scalability
*   **Technique**: Graph Database (Neo4j) (`database_manager.py`)
*   **Rationale**: The 3GPP data is highly interconnected. Procedures, network functions, messages, and steps are all nodes in a complex graph. A relational database would struggle to represent and query these many-to-many relationships efficiently.
*   **Benefit**: A graph database is the natural choice for this domain, allowing for efficient querying of complex relationships (e.g., "find all steps involving the AMF in the 5G AKA procedure").

*   **Technique**: Fail-Fast Database Connection (`database_manager.py`)
*   **Rationale**: The application previously only failed with a database error when it first attempted a query, which could be late in the process.
*   **Benefit**: By explicitly verifying the database connection upon initialization, the application fails immediately if the database is unavailable, saving time and providing clearer error feedback.

## Diagram Identification Strategy (Phase 1 - Text-based Heuristics)

To enable multi-modal knowledge graph construction, the pipeline now includes a strategy to identify sequence diagrams within `.docx` documents. This initial phase leverages text-based heuristics without requiring complex image analysis or OCR.

### Strategy:

1.  **Section Title Analysis (High Priority):** The title of the section containing a figure is analyzed for keywords strongly associated with procedures (e.g., "procedure", "flow", "sequence", "establishment", "management", "mobility", "registration", "authentication", "session"). The presence of such keywords is a strong indicator that the associated figure is a sequence diagram.
2.  **Caption Structure Analysis:** The figure's caption is checked for a strict numbering format (e.g., "Figure X.Y-Z: [Title]"). This formal, clause-based numbering scheme is highly indicative of a detailed procedure diagram, distinguishing it from more general or illustrative figures.

If a figure meets either of these criteria, it is classified as a sequence diagram and linked to its corresponding `DocumentSection`. This refined identification ensures that only relevant diagrams are used for further procedure identification.
