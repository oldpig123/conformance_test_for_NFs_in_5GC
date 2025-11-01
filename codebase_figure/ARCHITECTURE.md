# Codebase Architecture Documentation

This document provides a technical overview of the Python modules within the `codebase_figure` directory. It details the primary classes, key methods, and data structures that form the 3GPP Knowledge Graph construction pipeline.

## 1. High-Level Overview

The primary goal of this project is to parse 3GPP technical specification documents (`.docx`), extract key telecommunications entities and their relationships from both text and diagrams, and build a comprehensive, queryable knowledge graph in a Neo4j database. This graph then serves as the foundation for a semantic search engine and the automatic generation of Finite State Machines (FSMs).

## 2. Architectural Flow

The process is orchestrated by `main.py` and driven by the `KnowledgeGraphBuilder` class, which utilizes the other modules in a sequential, diagram-first pipeline.

### Data Flow Diagram

The following diagram illustrates how data flows between the core components:

```mermaid
graph TD
    A[Start: .docx Files] --> B(DocumentLoader);
    B --"List of DocumentSection"--> E{KnowledgeGraphBuilder};
    subgraph "KnowledgeGraphBuilder Pipeline"
    E --> D_P(DiagramParser);
    D_P --"Diagram is Sequence?"-->|Yes| D_E(Extract from Diagram);
    D_E --"Structural Entities"--> C(Enrich with EntityExtractor);
    D_P -->|No / Fail| C_T(Extract from Text);
    C_T --> C;
    C --"ProcedureContext (with Entities)"--> D(RelationExtractor);
    D --"ProcedureContext (with Relations)"--> E;
    end
    E --"Final Entities & Relations"--> F(DatabaseManager);
    F --> G[(Neo4j Database)];

    subgraph "Application Layer"
        G --> H(ProcedureSearchEngine);
        G --> I(FSMConverter);
    end
```

### Component Sequence

The pipeline operates in two main phases: an incremental build phase and a finalization phase.

**Build Phase (per-document loop):**
1.  **`DocumentLoader`**: Reads and parses all `.docx` files, extracting text sections and metadata for all embedded figures.
2.  **`KnowledgeGraphBuilder`**: Iterates through each document's sections.
3.  **`DiagramParser`**: For sections with figures, this new module attempts to parse the figure. It classifies the figure and, if it's a sequence diagram, extracts the core structural entities (Network Functions, Messages).
4.  **`EntityExtractor`**: Enriches the data. If a diagram was successfully parsed, it extracts detailed descriptions, parameters, and keys from the surrounding text. If diagram parsing failed or wasn't applicable, it performs a full entity extraction from the text as a fallback.
5.  **`RelationExtractor`**: Establishes relationships between all extracted entities.
6.  **`DatabaseManager`**: Writes the entities and relationships for the current document as nodes and edges into the Neo4j database. This step is repeated for every document, keeping memory usage low.

**Finalization Phase:**
7.  **`DatabaseManager`**: After all documents are processed, new methods (`get_all_entities`, `get_all_relationships`) are called to fetch the entire, complete graph from the database.
8.  **`ProcedureSearchEngine`**: Consumes the complete entity list fetched from the database to build its search indexes.
9.  **`FSMConverter`**: Uses the graph data fetched from the database to model procedures as state machines.

---

## 3. Core Data Structures (`data_structures.py`)

This module contains no executable code but is critical as it defines the core data classes that are passed between all other modules.

*   **`FigureMetadata`** (dataclass): Stores comprehensive metadata for an extracted figure:
    *   `caption`: The figure's caption text (initially empty, assigned later)
    *   `file_path`: Path to the extracted image file
    *   `file_type`: Image format (e.g., 'wmf', 'emf', 'png', 'jpg', 'vsdx', 'doc', 'docx', 'pptx')
    *   `original_index`: Sequential index of the figure in the document
    *   `r_id`: The relationship ID from the docx XML (e.g., 'rId3')
    *   `target_ref`: The target reference path from document.xml.rels (e.g., 'media/image1.wmf')
    *   `ole_prog_id`: ProgID from OLE objects (e.g., 'Visio.Drawing.15', 'Word.Picture.8')
    *   `ole_object_path`: Path to extracted OLE object file
    *   `is_ole_object`: Boolean flag indicating if this is an OLE object
    *   `nesting_level`: Recursion depth for nested documents (0=direct, 1=nested, etc., max 3)

*   **`DocumentSection`** (dataclass): Represents a clause or sub-clause parsed from a source `.docx` document:
    *   `title`: Section heading text
    *   `text`: Full text content of the section
    *   `clause`: Extracted clause number (e.g., '4.2.3')
    *   `document`: Source document filename
    *   `has_figure`: Boolean flag (True only if a figure has been successfully matched with a caption)
    *   `figures`: List of `FigureMetadata` objects
    *   `is_procedure`: Boolean flag indicating if this section describes a procedure
    *   `procedure_name`: The identified name of the procedure (if applicable)

*   **`Entity`** (dataclass): The primary representation of a node in the knowledge graph, with search capabilities:
    *   Core fields: `name`, `entity_type`, `properties`
    *   Search fields: `description`, `parent_title`, `search_keywords`, `embedding`, `title_embedding`, `parent_title_embedding`

*   **`Relationship`** (dataclass): The primary representation of an edge (relationship) in the knowledge graph:
    *   Core fields: `source_name`, `target_name`, `rel_type`, `properties`
    *   FSM fields: `is_transition`, `transition_condition`, `transition_action`

*   **`ProcedureContext`** (dataclass): A temporary data container that holds all extracted information for a single procedure during processing.

*   **FSM-Related Classes**: `FSMState`, `FSMTransition`, `FiniteStateMachine` for state machine representation.

*   **Search-Related Classes**: `SearchQuery`, `SearchResult` for search engine functionality.

---

## 4. Detailed Module Descriptions

### `main.py`

*   **Purpose**: The main entry point for the application. It initializes and runs the full knowledge graph construction pipeline and then demonstrates the search and FSM conversion functionalities.

### `config.py`

*   **Purpose**: A non-executable module that provides centralized configuration for the entire application.

### `document_loader.py`

*   **Purpose**: Handles the initial loading and parsing of raw `.docx` files. Extracts both legacy VML and modern DrawingML images from `.docx` files using XPath queries and relationship mapping.
*   **Main Class**: `DocumentLoader`
*   **Key Implementation Details**:
    *   Uses namespace-agnostic XPath to find both modern (`<w:blip>`) and legacy (`<v:imagedata>`) image references in paragraph XML.
    *   Extracts relationship IDs (e.g., `rId3`) and resolves them through `doc.part.rels` to actual image paths.
    *   Stores comprehensive metadata including `r_id`, `target_ref`, `file_type`, and `file_path` in `FigureMetadata` objects.
    *   Uses flexible regex pattern `r'^Figure[\s\-]'` to match both "Figure 4.2-1" and "Figure-4.2-1" caption formats.
*   **Method Reference**:
    *   `_extract_sections_with_figures(self, file_path)`
        *   **Description**: The main parsing method. It iterates through a document's paragraphs, reconstructing section text and extracting metadata for all embedded figures. When a figure caption is detected, it assigns the caption to the most recent uncaptioned figure in the current section and marks the section as having a valid figure.
    *   `_is_section_header(self, para, text)`
        *   **Description**: Determines if a paragraph is a section header by checking style names (Heading, H\d) and text patterns.
    *   `_extract_clause_number(self, text)`
        *   **Description**: Extracts the clause number from section header text.
    *   `identify_procedure_sections_with_llm(self, sections)`
        *   **Description**: Uses LLM analysis combined with enhanced fallback heuristics to identify which sections containing figures describe specific procedures.

### `diagram_parser.py`

*   **Purpose**: Handles the analysis, classification, and parsing of diagram files extracted from 3GPP documents. Uses Computer Vision to identify sequence diagrams and extract structural information. **Phase 1.6**: Added nested document extraction support.
*   **Main Class**: `DiagramParser`
*   **Key Implementation Details**:
    *   Supports vector (EMF/WMF), raster (PNG/JPG), Visio (VSDX), and nested documents (DOC/DOCX/PPTX)
    *   Uses LibreOffice in headless mode for format conversions
    *   Applies OpenCV Canny edge detection and Hough Line Transform for structural analysis
    *   Classifies diagrams based on detected vertical lines (lifelines) and horizontal lines (messages)
*   **Method Reference**:
    *   `parse_diagram(self, figure_metadata)`:
        *   **Description**: Main entry point. Routes to appropriate parser based on file type and OLE status.
        *   **Returns**: Dictionary with extracted data if sequence diagram, None otherwise.
    *   `_parse_visio_diagram(self, figure_metadata)`:
        *   **Description**: Handles Visio (.vsdx) files by converting to PNG and applying CV classification.
    *   `_parse_nested_document(self, figure_metadata, depth)`:
        *   **Description**: **NEW (Phase 1.6)** - Extracts diagrams from nested Word/PowerPoint objects.
        *   **Parameters**: depth tracks recursion level (max 3)
    *   `_extract_from_pptx(self, figure_metadata, depth)`:
        *   **Description**: **NEW (Phase 1.6)** - Exports PowerPoint slides as PNG images for classification.
        *   **Method**: Uses LibreOffice to convert entire slides to PNG
    *   `_extract_from_docx(self, figure_metadata, depth)`:
        *   **Description**: **NEW (Phase 1.6)** - Hybrid extraction from Word documents.
        *   **Phase 1**: Extract embedded images from relationships
        *   **Phase 2**: Export pages as PNG if no diagrams found (fallback)
    *   `_extract_from_doc(self, figure_metadata, depth)`:
        *   **Description**: **NEW (Phase 1.6)** - Handles old Word format by converting to .docx first.
    *   `_parse_vector_diagram(self, figure_metadata)`:
        *   **Description**: Handles EMF/WMF files. Attempts XML parsing first, then falls back to raster conversion.
    *   `_parse_raster_diagram(self, figure_metadata)`:
        *   **Description**: Handles PNG/JPG files using Computer Vision analysis.
    *   `_is_sequence_diagram_raster(self, figure_metadata)`:
        *   **Description**: Classifies raster diagrams using Hough Line Transform.
        *   **Algorithm**: Detects vertical lines (≥80°, length ≥20% height) and horizontal lines (≤10° or ≥170°, length ≥10% width)
        *   **Heuristic**: Requires ≥2 vertical lines, ≥3 horizontal lines, and horizontal > vertical
        *   **Returns**: bool indicating if diagram is a sequence diagram

### `entity_extractor.py`

*   **Purpose**: Extracts structured entities from text. It serves two roles: enriching diagram-extracted data with details from text (descriptions, parameters, keys) and acting as a fallback for full text-based extraction when diagram parsing is not applicable.
*   **Main Class**: `EntityExtractor`

### `relation_extractor.py`

*   **Purpose**: Extracts relationships between the entities previously identified by the `EntityExtractor` or `DiagramParser`.
*   **Main Class**: `RelationExtractor`

### `knowledge_graph_builder.py`

*   **Purpose**: The main orchestrator class. It manages the end-to-end incremental build pipeline.
*   **Main Class**: `KnowledgeGraphBuilder`
*   **Method Reference**:
    *   `build_knowledge_graph(self, file_paths)`
        *   **Description**: Orchestrates the new diagram-first pipeline. It uses the `DiagramParser` to extract structural information from figures and the `EntityExtractor` to enrich it with textual details or as a fallback.

### `database_manager.py`

*   **Purpose**: Provides a dedicated, low-level interface for all interactions with the Neo4j graph database.
*   **Main Class**: `DatabaseManager`

### `search_engine.py`

*   **Purpose**: Implements the full semantic search functionality.
*   **Main Class**: `ProcedureSearchEngine`

### `fsm_converter.py`

*   **Purpose**: Converts extracted procedures into Finite State Machine (FSM) representations for export.
*   **Main Class**: `FSMConverter`