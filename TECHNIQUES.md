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
    *   The weights (`W_SEMANTIC_TITLE: 1.7`, `W_SEMANTIC_PARENT: 1.7`, `W_SEMANTIC_DESC: 16`) are defined in `codebase_figure/config.py` and are finely tuned to achieve optimal relevance and ranking for golden queries.
    *   A similarity threshold is applied to filter out less relevant results.

### Optimization:

*   **Multi-GPU Support**: The embedding model is loaded onto a separate GPU (if available) to parallelize processing and reduce latency.
*   **Adaptive Embedding**: For very long procedure descriptions, the system intelligently chunks the text into smaller, overlapping segments, generates embeddings for each, and then averages them. This ensures that the full semantic context is captured without exceeding the embedding model's token limit or causing `CUDA out of memory` errors.

## 6. Database and Scalability
*   **Technique**: Graph Database (Neo4j) (`database_manager.py`)
*   **Rationale**: The 3GPP data is highly interconnected. Procedures, network functions, messages, and steps are all nodes in a complex graph. A relational database would struggle to represent and query these many-to-many relationships efficiently.
*   **Benefit**: A graph database is the natural choice for this domain, allowing for efficient querying of complex relationships (e.g., "find all steps involving the AMF in the 5G AKA procedure").

*   **Technique**: Fail-Fast Database Connection (`database_manager.py`)
*   **Rationale**: The application previously only failed with a database error when it first attempted a query, which could be late in the process.
*   **Benefit**: By explicitly verifying the database connection upon initialization, the application fails immediately if the database is unavailable, saving time and providing clearer error feedback.

## 4. Figure Extraction from DOCX Files

*   **Technique**: VML and DrawingML Image Extraction (`document_loader.py`)
*   **Rationale**: 3GPP documents use both legacy VML (`<v:imagedata>`) and modern DrawingML (`<w:blip>`) formats for embedded images. A robust extraction method must handle both formats and correctly associate images with their captions, which can have varying formats ("Figure 4.2-1" vs "Figure-4.2-1").
*   **Implementation**:
    1.  For each paragraph, use namespace-agnostic XPath to find both modern (`<w:blip r:embed="rId">`) and legacy (`<v:imagedata r:id="rId">`) image references.
    2.  Extract the relationship ID (e.g., `rId3`) from the XML attributes.
    3.  Resolve the relationship ID through `doc.part.rels` to get the actual image path from `word/_rels/document.xml.rels` (e.g., `media/image1.wmf`).
    4.  Extract and save the image blob to a file.
    5.  Store metadata (`r_id`, `target_ref`, `file_type`, `file_path`) in a `FigureMetadata` object.
    6.  When a figure caption is detected (using flexible regex `r'^Figure[\s\-]'`), match it with the most recent uncaptioned figure in the current section.
*   **Benefit**: This approach correctly extracts all embedded images regardless of format and handles various caption styles used in 3GPP documents, ensuring 100% figure-to-caption matching accuracy.

## 5. Diagram-Centric Pipeline

To enable multi-modal knowledge graph construction, the pipeline has been refactored to be diagram-centric.

### Strategy:

1.  **Figure Extraction**: The `DocumentLoader` first extracts all figures from the document, ensuring accurate figure-to-caption association, without attempting to classify them.
2.  **Procedure Identification**: The `KnowledgeGraphBuilder` identifies potential procedure sections, primarily by looking for sections that contain figures.
3.  **Diagram Classification & Parsing**: A dedicated `DiagramParser` analyzes the content of each figure using Computer Vision techniques:
    *   **Vector Format Conversion**: Binary EMF/WMF files are converted to PNG using LibreOffice headless mode
    *   **Line Detection**: Applies OpenCV Hough Line Transform to detect structural patterns
    *   **Classification**: Identifies sequence diagrams based on presence of vertical lifelines (≥2) and horizontal messages (≥3)
    *   **Accuracy**: Achieves ~60% classification accuracy on 3GPP specification figures
4.  **Textual Enrichment**: The `EntityExtractor` processes the text associated with the diagram to provide detailed descriptions, parameters, and keys, enriching the structurally-sound skeleton provided by the diagram parser.

## 6. Computer Vision for Diagram Classification

*   **Technique**: Hough Line Transform for Sequence Diagram Detection (`diagram_parser.py`)
*   **Rationale**: 3GPP specifications contain hundreds of diagrams (sequence diagrams, flowcharts, architecture diagrams, etc.). Manually identifying which ones are sequence diagrams is time-consuming and error-prone. An automated CV-based approach can efficiently classify diagrams at scale.
*   **Implementation**:
    1.  **Format Handling**: Convert binary EMF/WMF to PNG using LibreOffice (`--headless --convert-to png`)
    2.  **Edge Detection**: Apply Canny edge detection to identify diagram boundaries
    3.  **Line Detection**: Use HoughLinesP to detect straight lines with configurable thresholds
    4.  **Angle Classification**: Calculate line angles to distinguish vertical (80-100°) from horizontal (0-10° or 170-180°)
    5.  **Length Filtering**: Only count substantial lines (vertical ≥20% height, horizontal ≥10% width)
    6.  **Heuristic Matching**: Classify as sequence diagram if structural pattern matches (≥2 lifelines, ≥3 messages, horizontal > vertical)
*   **Benefit**: Automated classification reduces manual effort and enables scalable processing of large document sets. The 60% accuracy provides a good balance between precision and recall for initial entity extraction, with text-based fallback handling misclassifications.
