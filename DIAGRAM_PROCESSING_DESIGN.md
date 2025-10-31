# Future Architecture: Diagram-Based Knowledge Graph Construction

This document summarizes a discussion on a potential future architecture for improving the knowledge graph construction process by parsing sequence diagrams directly from the 3GPP documents.

---

## ✅ Completed Phase: Figure Extraction (2025-10-31)

**Status:** ✅ **DONE** - Ready for next phase

### What Was Completed:
1. ✅ Figure extraction from DOCX files (both VML and DrawingML)
2. ✅ Accurate figure-to-caption association (100% - 286/286 figures)
3. ✅ Fixed regex patterns to handle "Figure-X" format
4. ✅ Removed duplicate class definitions in `data_structures.py`
5. ✅ Comprehensive metadata tracking (`r_id`, `target_ref`, `file_path`, `file_type`)
6. ✅ All documentation updated and aligned with implementation

### Files Modified:
- `codebase_figure/data_structures.py` - Clean dataclass definitions
- `codebase_figure/document_loader.py` - Robust VML/DrawingML extraction
- `README.md`, `CHANGELOG.md`, `TECHNIQUES.md`, `codebase_figure/ARCHITECTURE.md` - Updated

### Test Results:
- ✅ All 286 figures extracted with captions
- ✅ Python files compile without errors
- ✅ VML and DrawingML extraction working correctly

---

## 🚀 Next Phase: Diagram Parser Implementation (Starting Tomorrow)

**Objective:** Implement `diagram_parser.py` to read sequence diagrams and extract entities/relations for the knowledge graph.

### What's Ready:
- `FigureMetadata` objects with file paths and metadata
- `DocumentSection` objects with figures and context
- Pipeline placeholder in architecture

### What to Implement:

This document summarizes a discussion on a potential future architecture for improving the knowledge graph construction process by parsing sequence diagrams directly from the 3GPP documents.

## 1. Core Concept: Hybrid Diagram-Text Model

The proposed ideal architecture is a hybrid model that leverages the strengths of both the graphical diagrams and the descriptive text.

1.  **Diagram-First for Structure:** The sequence diagram would be parsed to create a highly accurate and unambiguous "skeleton" of the procedure. This includes:
    *   The **Network Functions** involved (from the lifelines).
    *   The **Messages** exchanged (from the text labels on arrows).
    *   The exact **sequence** of those messages (`FOLLOWED_BY` relationships, from the vertical position of arrows).
    *   The **source and target** of each message (`SENDS`/`SEND_TO` relationships, from the start and end points of arrows).

2.  **Text-Based for Enrichment:** The text paragraphs corresponding to each step in the diagram would then be parsed to enrich this skeleton with crucial low-level details:
    *   The full, detailed **description** of what happens in that step.
    *   The specific **Parameters** and **Keys** associated with each message.
    *   Conditional logic (e.g., "If X, then...") that is not present in the diagram.

## 2. Technical Implementation Challenges & Plan

Implementing this would be a complex, multi-modal data extraction task.

### 2.1. Diagram Extraction and Type Identification

The first step is to extract the diagram objects from the `.docx` file. This is complicated by the fact that the diagrams are a mix of different formats:
*   **Vector formats:** `.wmf` and `.emf` (which contain structured `DrawingML` XML).
*   **Raster formats:** `.png`, `.jpg`, `.jpeg` (which are flat grids of pixels).

The pipeline would need to identify the format of each diagram before routing it to the appropriate parser.


1. Section Title Analysis (High Priority): The title of the section containing the diagram is our strongest clue. If the section title itself contains words like "procedure", "flow", or "establishment", we can be highly confident that the associated figure is a sequence diagram.

2. Caption Structure Analysis: We can strengthen the existing _is_figure_caption_for_section logic. A caption that follows the strict format "Figure [Clause Number]-[Index]: [Title]" is a very strong indicator of a formal procedure diagram, even without specific keywords in the
      title itself.

3. Diagram Content Analysis (The Ideal Goal): This is the most powerful method, as outlined in our discussion in figure_base.md.
    * For vector diagrams (.emf/.wmf), I can parse the underlying XML to programmatically identify the classic sequence diagram structure: multiple vertical lifelines with horizontal arrows between them.
    * For raster diagrams (.png/.jpg), this would involve using computer vision (e.g., OpenCV) to detect those same patterns (vertical lines and horizontal arrows).


### 2.1.1. New Implementation Strategy: Content-First Diagram Analysis

Based on our latest discussion, we will pivot to a more direct, content-first approach, bypassing the text-based heuristics for classification. This aligns with the "Ideal Goal" and is expected to be more accurate.

don't classify right after figure finding, but process them all with cv/ocr/drawingml, if a sequence diagram, extract entity and relation for KG, and title, parent title, full procedure context for search feature. if not, just skip it.
**The New Pipeline:**

1.  **Pure Figure Extraction (`document_loader.py`):**
    *   The role of `document_loader.py` will be simplified to focus exclusively on extracting all embedded graphical objects (both modern DrawingML and legacy VML) from the `.docx` files.
    *   It will save each extracted figure to a temporary file and create a `FigureMetadata` object containing the file path and type, but it will *not* perform any classification.

2.  **New `diagram_parser.py` Module:**
    *   A new, dedicated module, `diagram_parser.py`, will be created to handle the complex task of analyzing the content of the extracted figures.
    *   This module will contain specialized parsers for different image types.

3.  **Content-Based Classification and Parsing (`diagram_parser.py`):**
    *   **Vector Diagrams (EMF/WMF):** The parser will analyze the underlying XML structure to programmatically identify the components of a sequence diagram (e.g., vertical lifelines, horizontal arrows).
    *   **Raster Diagrams (PNG/JPG):** The parser will use a Computer Vision (CV) and Optical Character Recognition (OCR) pipeline (e.g., using OpenCV and Tesseract) to detect the same structural patterns and extract text.
    *   A figure will only be classified as a sequence diagram if these structural components are successfully identified from its content.

4.  **Conditional Knowledge Graph Construction (`knowledge_graph_builder.py`):**
    *   The main builder will orchestrate the process. For each figure extracted by the `DocumentLoader`:
        *   It will pass the figure to the `DiagramParser`.
        *   **If the parser identifies it as a sequence diagram:**
            *   The parser will extract the core structural entities (Network Functions, Messages, sequence) from the diagram.
            *   The `KnowledgeGraphBuilder` will then use this structured information to build the primary skeleton of the knowledge graph.
            *   The surrounding text from the `DocumentSection` (title, parent title, and full text) will be used by the `EntityExtractor` and `RelationExtractor` to enrich this skeleton with details like parameters, keys, and full step descriptions for the search feature.
        *   **If the figure is not a sequence diagram,** it will be ignored for the purpose of KG construction.

This content-first strategy is more robust and directly tackles the core challenge of accurately identifying and utilizing the rich structural information within sequence diagrams.

### 2.1.2. LLM's Evolving Role in Procedure Identification and Enrichment

Our discussion clarified the strategic role of the LLM within this phased approach:

*   **Procedure Identification:** The LLM remains critical for identifying procedures from text, especially in sections that do not contain a sequence diagram.

*   **Text-Based Knowledge Graph Enrichment (Post-Diagram Parsing):**
    *   Crucially, the **core structural elements** of the Knowledge Graph (Network Functions, Messages, and their sequence) will be derived directly from parsing the sequence diagram content.
    *   The LLM's primary role in "Text-Based for Enrichment" will be to extract **details from the accompanying text paragraphs** that are not explicitly visible in the diagram. This includes:
        *   Full, detailed **descriptions** of what happens in each step.
        *   Specific **Parameters** and **Keys** associated with messages.
        *   **Conditional logic** (e.g., "If X, then...") not represented visually.
    *   This text-based enrichment is fundamental for enabling the semantic search feature, as these extracted details form the basis for generating embeddings and keywords.

### 2.1.3. Linking Sequence Diagrams to Procedures

A sequence diagram is inherently linked to the procedure it illustrates through the document's hierarchical structure. The process for establishing this link is as follows:

1.  **Document Section as Container:** Each `DocumentSection` object represents a specific clause or sub-clause in the 3GPP document. A sequence diagram, along with its caption, is embedded within one of these sections.
2.  **Figure Metadata within Section:** When a figure is detected and extracted, its `FigureMetadata` (containing the extracted file path, caption, type, and `is_sequence_diagram` classification) will be stored directly within the `DocumentSection.figures` list.
3.  **Procedure Identification at Section Level:** The `identify_procedure_sections_with_llm` method (or its enhanced version) operates on `DocumentSection` objects. It uses various signals, including the `figure_is_sequence_diagram` flag, to determine if a given `DocumentSection` *describes a procedure*.
4.  **Implicit Link:** If a `DocumentSection` is successfully identified as describing a procedure (i.e., `is_procedure` is `True` and `procedure_name` is populated), then any sequence diagram contained within that `DocumentSection` (as indicated by its `FigureMetadata` in `DocumentSection.figures`) is implicitly linked to that identified procedure. The procedure's name and other details are directly accessible via the parent `DocumentSection`.

**Phase 2: Diagram Content Analysis (Future Work)**

This phase, to be implemented later, will involve analyzing the visual content of the diagrams themselves.

*   **Vector Diagrams (EMF/WMF):** Parse the underlying `DrawingML` XML to identify structural elements characteristic of sequence diagrams (vertical lifelines, horizontal arrows).
*   **Raster Diagrams (PNG/JPG):** Employ Computer Vision (e.g., OpenCV) and OCR (e.g., Tesseract) to detect visual patterns (lifelines, arrows) and extract text labels within the diagram).

### 2.2. Parsing Methodologies

A two-pronged approach would be required:

#### A) Vector Diagram Parsing (for EMF/WMF)

This is the preferred method due to the structured nature of the data.
*   **Tooling:** Use a library like `lxml` to parse the `DrawingML` XML.
*   **Logic:**
    1.  **Identify Lifelines (NFs):** Parse XML elements for tall rectangles. The text within these shapes defines the `NetworkFunction` entities.
    2.  **Identify Messages & Parameters:** For each horizontal arrow element:
        *   The text block spatially **above** the arrow is the `Message` name.
        *   The text block spatially **below** the arrow contains the `Parameter`/`Key` list.
    3.  **Parse Parameters:** The parameter string (e.g., `(param1, [param2])`) would be parsed to extract individual parameters and their optionality (based on `[]` brackets).
    4.  **Determine Sequence:** The vertical (`y`) coordinate of each arrow in the XML determines the message order.

#### B) Raster Diagram Parsing (for PNG/JPG)

This is a fallback for "flat" images.
*   **Tooling:** A Computer Vision pipeline using libraries like OpenCV and an OCR engine like Tesseract.
*   **Logic:**
    1.  **Shape Detection:** Use algorithms to find all rectangles (lifelines) and horizontal lines (arrows).
    2.  **Text Recognition (OCR):** Extract all text and its bounding box coordinates from the image.
    3.  **Correlation:** Spatially correlate the recognized text with the detected shapes. Text above an arrow is a message; text below is parameters; text inside a vertical box is a Network Function.

### 2.3. Handling Inconsistencies

As noted in the discussion, the document authors are not always consistent. The use of **solid vs. dashed arrows** cannot be reliably used to determine if a message is a "request" or "reply". Therefore, this information would be ignored, and message type would have to be inferred from the message name itself.

## 3. Conclusion

This hybrid, diagram-first approach would produce a significantly more accurate and reliable knowledge graph. However, due to the technical complexity of parsing mixed-format embedded objects, it represents a major future development effort. The immediate priority remains to fix and stabilize the current text-only extraction pipeline.

# Full roadmap for Implementation (temporary)
implement the figure extraction and classification into sequence diagram in the current codebase in `codebase_figure/`, the following steps will be taken:

## logic of figure extraction from .docx file
To ensure all figures are extracted, especially legacy vector images (VML), a hybrid approach will be used:

1.  **High-Level Extraction with `python-docx`:** The primary method will be to iterate through the document using the `python-docx` library to find modern image formats embedded as `InlineShape` objects. This is robust for most images.

2.  **Low-Level XML Parsing for VML:** To handle legacy Vector Markup Language (VML) images, which `python-docx` may miss, a secondary, low-level parsing method will be implemented. This method will:
    *   Open the `.docx` file as a zip archive.
    *   Parse `word/_rels/document.xml.rels` to map relationship IDs (rIds) to image file paths.
    *   Parse `word/document.xml` to find `<v:imagedata>` tags.
    *   Use the `r:embed` attribute from the tag to look up the image path in the relationship map and extract the image.

This dual strategy ensures a comprehensive extraction of all embedded figures, regardless of their format.
  ---

  Overall Plan (High-Level):

   1. Data Structures Update (`data_structures.py`)
   2. Document Loading & Figure Identification (`document_loader.py`)
   3. Procedure Identification Refinement (`document_loader.py`)
   4. Knowledge Graph Construction Adaptation (Future - `entity_extractor.py`, `relation_extractor.py`, `knowledge_graph_builder.py`)
   5. Testing and Verification

  ---

  Detailed Step-by-Step Plan:

  Step 1: Data Structures Update (`codebase_figure/data_structures.py`)

   * Action 1.1: Add from pathlib import Path to imports.
   * Action 1.2: Define the FigureMetadata dataclass. This will store detailed information about each extracted figure.
```python
   @dataclass
   class FigureMetadata:
       """Metadata for an extracted figure."""
       caption: str
       file_path: Path
       file_type: str # e.g., 'png', 'emf', 'wmf', 'drawingml'
       original_index: int # Its order in the document
       is_sequence_diagram: bool = False
```
   * Action 1.3: Modify DocumentSection dataclass:
       * Change figures: List[str] to figures: List['FigureMetadata'].
       * Add figure_is_sequence_diagram: bool = False. This flag will indicate if any figure within the section is classified as a sequence diagram.

  Step 2: Document Loading & Figure Identification (`codebase_figure/document_loader.py`)

   * Action 2.1: Add necessary imports: `zipfile`, `xml.etree.ElementTree`.
   * Action 2.2: Implement a new helper method `_extract_vml_images(self, file_path: Path) -> Dict[int, List[FigureMetadata]]`. This method will:
       * Open the `.docx` file using `zipfile`.
       * Parse `word/_rels/document.xml.rels` to build a relationship map (rId -> image path).
       * Parse `word/document.xml` to find `<v:imagedata>` tags within paragraphs.
       * For each tag, extract the image using the relationship map.
       * Return a dictionary mapping the paragraph index to a list of `FigureMetadata` objects for the VML images found in that paragraph.
   * Action 2.3: Modify the main `_extract_sections_with_figures(self, file_path: Path) -> List[DocumentSection]` method:
       * Call `_extract_vml_images` at the beginning to get all legacy VML images.
       * As it iterates through paragraphs with `python-docx`, it will also process modern `InlineShape` objects.
       * It will merge the results from both extraction methods.
       * When a figure caption is identified, it will associate the caption with the correct `FigureMetadata` object (from either VML or InlineShape extraction).
       * It will then call `_classify_figure_as_sequence_diagram` to set the flag on the `FigureMetadata`.
       * Finally, it will add the complete `FigureMetadata` to `current_section.figures` and update the `figure_is_sequence_diagram` flag on the `DocumentSection`.

  Step 3: Procedure Identification Refinement (`codebase_figure/document_loader.py`)

   * Action 3.1: Modify identify_procedure_sections_with_llm(self, sections: List[DocumentSection]) -> List[DocumentSection]:
       * Instead of filtering sections_with_figures = [s for s in sections if s.has_figure], prioritize sections where s.figure_is_sequence_diagram is True. These are our high-confidence procedure candidates.
       * Adjust the scoring in _enhanced_fallback_identification to give a much higher boost (or even direct classification) if section.figure_is_sequence_diagram is True.
       * The LLM (_query_llm_for_procedure_identification) will still be used for sections that don't have a sequence diagram but might still be procedures, or as a secondary check.

  Step 4: Knowledge Graph Construction Adaptation (Future - `entity_extractor.py`, `relation_extractor.py`, `knowledge_graph_builder.py`)

   * Action 4.1 (Phase 2 Integration): Once Phase 1 is stable, we will introduce a new module (e.g., diagram_parser.py) or extend entity_extractor.py to:
       * Iterate through DocumentSection objects where figure_is_sequence_diagram is True.
       * For each such section, retrieve the FigureMetadata objects.
       * Based on FigureMetadata.file_type, call the appropriate parser (DrawingML parser for vector, CV/OCR pipeline for raster).
       * Extract Network Functions, Messages, and their sequence directly from the diagram.
   * Action 4.2: Adapt entity_extractor.py and relation_extractor.py to consume the diagram-extracted entities and relationships as the primary source of truth for structure.
   * Action 4.3: Use the LLM (via entity_extractor.py) to perform text-based enrichment (detailed step descriptions, parameters, keys, conditional logic) from the DocumentSection.text for the diagram-derived structure.

  Step 5: Testing and Verification

   * Action 5.1: Develop unit tests for each new method and modified logic.
   * Action 5.2: Perform integration tests to ensure the entire pipeline works as expected.

  ---

   1. Figure Extraction and Saving (`document_loader.py`):
       * The document_loader.py module will be responsible for the initial task of extracting the raw embedded figure data (PNG, JPG, EMF, WMF, DrawingML XML) from the .docx file.
       * It will then save these raw figures to temporary files (e.g., in an output/figures directory).
       * The FigureMetadata object (which we'll add to data_structures.py) will store the file_path to these extracted temporary files, along with their file_type. This makes the figure content accessible for later processing.

   2. Diagram Content Processing (New Module: `diagram_parser.py`):
       * The actual parsing of the diagram content (i.e., applying CV/OCR for raster images or lxml for DrawingML/vector graphics to extract Network Functions, Messages, and their sequence) will be handled by a new, dedicated module, which I propose naming diagram_parser.py.
       * This module would contain specialized parsers:
           * A DrawingMLParser for vector formats (EMF, WMF, DrawingML XML).
           * A RasterDiagramParser for raster formats (PNG, JPG) using OpenCV and Tesseract.
       * These parsers would take the file_path and file_type from a FigureMetadata object and return a structured representation of the diagram's content (e.g., a list of identified NFs, messages, and their sequential order).

   3. Integration into the Pipeline (`entity_extractor.py` and `knowledge_graph_builder.py`):
       * The entity_extractor.py module, which is currently responsible for extracting entities and relationships from text, would be adapted. For DocumentSection objects that contain identified sequence diagrams (figure_is_sequence_diagram is True), the entity_extractor would call
         the appropriate parser within diagram_parser.py.
       * The knowledge_graph_builder.py orchestrates the entire process, ensuring that for each relevant DocumentSection, the diagram parsing is invoked at the correct stage.
  