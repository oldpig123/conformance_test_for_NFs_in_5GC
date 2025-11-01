# 3GPP Knowledge Graph and FSM Builder

This project automates the process of building a comprehensive knowledge graph from 3GPP technical specification documents (`.docx`). It uses a combination of rule-based parsing and modern NLP/LLM techniques to identify and extract key telecommunications concepts, such as procedures, network functions, messages, and their relationships.

The resulting knowledge graph is stored in a Neo4j database and can be used for various purposes, including semantic search and the automatic generation of Finite State Machines (FSMs) for conformance testing.

## Key Features

- **Automated Knowledge Graph Construction**: Parses 3GPP `.docx` files to build a graph of telecom entities and their relationships.
- **Diagram-Centric Pipeline**: A new pipeline that prioritizes extracting information from sequence diagrams for a more accurate and robust knowledge graph.
- **LLM-Powered Extraction**: Leverages Large Language Models for complex tasks like procedure identification, entity extraction, and relationship discovery.
- **Rich Entity Model**: Extracts a variety of 3GPP-specific entities, including:
    - `Procedure`
    - `NetworkFunction` (AMF, SMF, etc.)
    - `Message`
    - `Parameter` (SUCI, GUTI, etc.)
    - `Key` (Kausf, Kseaf, etc.)
    - `Step` (with detailed descriptions from the source document).
- **Graph Database Storage**: Persists the constructed knowledge graph in a Neo4j database for robust querying and analysis.
## Search Engine

The project features a powerful search engine designed to quickly retrieve relevant 3GPP procedures based on natural language queries. It employs a **Full Semantic Search Model** that leverages multiple semantic signals for highly accurate conceptual understanding.

### Key Features:

*   **Full Semantic Approach**: Relies entirely on semantic embeddings for conceptual understanding, abandoning traditional keyword-based (TF-IDF) search.
*   **Multi-Field Semantic Embeddings**: Generates and utilizes three separate semantic embeddings for each procedure:
    *   An embedding for the procedure's `title`.
    *   An embedding for the `parent_title` of its containing section.
    *   An embedding for the full `description` (concatenated steps).
*   **Optimized Weighted Scoring**: A finely tuned weighting scheme (defined in `codebase_figure/config.py` as `W_SEMANTIC_TITLE: 1.7`, `W_SEMANTIC_PARENT: 1.7`, `W_SEMANTIC_DESC: 16`) combines these semantic signals to achieve optimal relevance and ranking for golden queries.
*   **LLM-Enhanced Descriptions**: Leverages LLMs to generate rich, concise summaries of procedures, enhancing semantic search accuracy.
- **FSM Generation**: Automatically converts extracted procedures from the knowledge graph into Finite State Machines (FSMs), which are exported in both `.json` and `.dot` (for Graphviz visualization) formats.
- **GPU Accelerated**: Designed to utilize GPU for NLP/LLM model inference. Now supports multi-GPU configurations by distributing the main LLM and the embedding model across separate devices to parallelize workloads and increase processing speed.
- **Modular Architecture**: The codebase is logically structured into modules for easy maintenance and extension.

## Project Structure

The core logic is contained within the `codebase/` and `codebase_figure/` directories:

```
codebase_figure/
├── main.py                   # Main entry point to run the full pipeline.
├── knowledge_graph_builder.py# Orchestrates the entire KG construction process.
├── config.py                 # Central configuration for DB, paths, and models.
├── data_structures.py        # Defines all data classes (Entity, FSM, etc.).
├── document_loader.py        # Handles loading and parsing of .docx files.
├── diagram_parser.py         # Parses and classifies diagrams from figures.
├── entity_extractor.py       # Extracts entities (Procedures, NFs, Messages, etc.).
├── relation_extractor.py     # Extracts relationships between entities.
├── database_manager.py       # Manages all interactions with the Neo4j database.
├── search_engine.py          # Implements natural language search functionality.
└── fsm_converter.py          # Converts KG procedures into FSMs.
```

## System Requirements

- **Python**: 3.12
- **Database**: A running Neo4j database instance.
- **GPU**: An NVIDIA GPU with CUDA 11.8+ support is recommended for better performance.

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-name>
    ```

2.  **Create and activate the Conda environment:**
    The `conformance_test.yml` file contains all the necessary dependencies. Use Conda to create the environment:
    ```bash
    conda env create -f conformance_test.yml
    conda activate conformance_test
    ```
    This will install Python, all required packages, and **PyTorch with CUDA 11.8 via pip** (included in the environment file).
    
    **Note:** PyTorch is installed via pip (not conda) due to dependency conflicts with newer library versions. This is handled automatically by the environment file and provides full GPU support.

3.  **Place 3GPP Documents:**
    Put your 3GPP specification `.docx` files into the `3GPP/` directory (or the directory specified by `DOCS_PATH` in the config).

4.  **Configure the application:**
    Open `codebase_figure/config.py` and edit the following sections:
    - **Neo4j Database:** Update `NEO4J_URI`, `NEO4J_USER`, and `NEO4J_PASSWORD` to match your database credentials.
    - **Documents Path:** Ensure `DOCS_PATH` points to the directory containing your `.docx` files.

## How to Run

Execute the main script from the project root directory:

```bash
python codebase_figure/main.py
```

The script will perform the following steps:
1.  Initialize all components and check for GPU.
2.  Load and parse the 3GPP documents, extracting figure metadata.
3.  Identify procedure sections using LLM analysis.
4.  For each procedure, attempt to parse sequence diagrams to extract core entities.
5.  Enrich the graph with details from the text using the Entity and Relation Extractors.
6.  Clear the existing Neo4j database.
7.  Load the newly constructed knowledge graph into Neo4j.
8.  Run a demo that showcases the search and FSM conversion features.

The generated FSM files (`.json` and `.dot`) will be saved in the `output/` directory.

## The Pipeline Explained

The project follows a systematic pipeline to transform raw documents into a structured knowledge graph and FSMs:

1.  **Document Loading**: `.docx` files are parsed to extract text and metadata for any embedded figures.
2.  **Procedure Identification**: An LLM analyzes sections containing figures to identify which ones describe a specific, step-by-step procedure, ensuring accurate figure-to-caption association.
3.  **Diagram-First Extraction**: For identified procedures, a `DiagramParser` attempts to analyze the figure content. If it's a sequence diagram, it extracts the core structure (Network Functions, Messages, sequence).
4.  **Text-Based Enrichment**: The `EntityExtractor` then processes the surrounding text to enrich the graph with details not found in the diagram (step descriptions, parameters, keys) and serves as a fallback if diagram parsing fails.
5.  **Incremental Database Loading**: The pipeline processes one document at a time. After each document is fully analyzed, its extracted entities and relationships are immediately loaded into the Neo4j database. This incremental approach ensures that memory usage remains low and stable, allowing the system to scale to a large number of documents.
6.  **Search Indexing & FSM Conversion**: After the entire build process is complete, the `main.py` script fetches the complete knowledge graph from the database to build a globally-aware search index. It then demonstrates the search and FSM conversion features using this complete dataset.

## Diagram Classification

The system includes **Computer Vision-based sequence diagram classification** that automatically identifies sequence diagrams from 3GPP specification figures:

### Classification Algorithm
*   **Vector Format Support (EMF/WMF)**: Converts to PNG using LibreOffice headless mode
*   **Raster Format Support (PNG/JPG)**: Direct Computer Vision analysis
*   **Detection Method**: Hough Line Transform to identify diagram structure:
    *   Vertical lines → Lifelines/Actors
    *   Horizontal lines → Messages/Interactions
*   **Classification Heuristics**: 
    *   Minimum 2 vertical lines (lifelines)
    *   Minimum 3 horizontal lines (messages)
    *   More horizontal than vertical lines (typical sequence diagram pattern)
*   **Accuracy**: ~60% on test dataset (12/20 correctly classified)

### Technical Details
*   **OpenCV**: Edge detection (Canny) and line detection (HoughLinesP)
*   **LibreOffice**: Binary EMF/WMF → PNG conversion
*   **Processing Time**: 2-3 seconds per diagram including conversion
*   **Integration**: Classified diagrams are automatically associated with their parent procedures

## Future Enhancements

*   **Entity Extraction from Diagrams**: Extract Network Functions and Messages from classified sequence diagrams
*   **OCR Integration**: Add Tesseract OCR to extract text labels from diagram elements
*   **Sequence Ordering**: Determine temporal order of messages from vertical positions
*   **Enhanced Classification**: Fine-tune thresholds and add machine learning classifier
