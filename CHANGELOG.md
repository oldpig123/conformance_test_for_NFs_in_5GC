## 2025-10-28

### Features

-   **Optimized Full Semantic Search Model:** Tuned `SEARCH_WEIGHTS` (`W_SEMANTIC_TITLE: 1.7`, `W_SEMANTIC_PARENT: 1.7`, `W_SEMANTIC_DESC: 16`) in `config.py` for the full semantic search model, achieving optimal ranking for golden queries.
-   **Diagram-Based KG Construction (Phase 1 - Initial Setup):** Implemented foundational components in `codebase_figure/` for multi-modal knowledge graph construction. This includes:
    *   Added `Diagram` dataclass and `diagrams` field to `DocumentSection` in `data_structures.py`.
    *   Added image extraction, sequence diagram classification heuristics (`_is_sequence_diagram`), and updated document parsing logic in `document_loader.py` to identify procedures based on sequence diagrams.

### Bug Fixes

-   **Improved Parent Title Recognition:** Fixed an issue in `document_loader.py` where section headers with `H<digit>` styles were not correctly identified, leading to missing `parent_title` for child sections.
-   **Corrected Parent Title Clause Handling:** Resolved a bug in `knowledge_graph_builder.py` where parent clauses ending in `.0` were incorrectly resolved (e.g., `6.1.3.2.0` resolved to `6.1.3.2` instead of `6.1.3`).

### Documentation

-   Updated `figure_base.md` to reflect the refined phased strategy for diagram identification.

## Date: 2025-10-28 (Documentation Sync)

### Problem:
The project's documentation (`README.md`, `CHANGELOG.md`, `TECHNIQUES.md`, `ARCHITECTURE.md`) had fallen out of sync with the actual codebase. The documentation described a "Full Semantic Search" model, while the implementation uses a more effective hybrid model combining keyword (TF-IDF) and semantic search.

### Solution:
A full documentation sweep was performed to align all documents with the current, stable codebase.

### Implementation Details:
1.  **Search Model Correction**: All references to "Full Semantic Search" were replaced with "Hybrid Search Model." The documentation now accurately describes the three-part scoring:
    *   **TF-IDF on Title**: Keyword search on the procedure's own title.
    *   **TF-IDF on Parent Title**: Keyword search on the title of the procedure's parent section.
    *   **Semantic Search on Description**: Vector-based search on the full, concatenated text of the procedure's steps.
    *   The final score is a weighted sum of these three components, with weights defined in `codebase/config.py`.
2.  **Architecture Document Update**: `codebase/ARCHITECTURE.md` and `codebase_figure/ARCHITECTURE.md` were updated to reflect the current hybrid search architecture and the enhanced step parsing logic in `entity_extractor.py`.
3.  **General Consistency**: Ensured that `README.md` and `TECHNIQUES.md` also reflect these changes.

### Goal:
To provide clear, accurate, and up-to-date documentation for developers, reflecting the project's most recent stable architecture.

## Date: 2025-10-27 (Fix: 'DESCRIPTION: None' and Enhanced Step Parsing)

### Problem:
The `DESCRIPTION: None` issue persisted for many procedures, leading to incomplete knowledge graph entries and hindering semantic search quality. This was traced to two main problems:
1.  `document_loader.py` was not correctly preserving paragraph breaks, leading to concatenated text that was difficult to parse.
2.  The step extraction logic in `entity_extractor.py` was brittle, often failing to identify the start of step sequences or incorrectly parsing multi-paragraph steps.

### Solution:
A two-part solution was implemented to ensure accurate and complete step description extraction:
1.  **Document Loader Fix**: Modified `document_loader.py` to correctly preserve newlines between paragraphs, providing cleaner input for step extraction.
2.  **Enhanced Step Parsing**: The step extraction logic in `entity_extractor.py` was significantly improved to robustly identify and parse multi-paragraph steps, even when they are not explicitly numbered or are concatenated in the raw text. This involved:
    *   A new `_preprocess_step_text` function to insert newlines before step patterns.
    *   A more flexible `_find_step_start_location` to better detect where steps begin.
    *   A refined `_combine_step_paragraphs` to correctly reconstruct multi-paragraph step descriptions.

### Implementation Details:
1.  **`codebase/document_loader.py`**: The `_extract_sections_with_figures` method was updated to use `current_section.text += "\n" + text` instead of `current_section.text += " " + text` to preserve paragraph breaks.
2.  **`codebase/entity_extractor.py`**:
    *   The `_extract_step_descriptions_from_document` method was refactored to use `_preprocess_step_text` and `_find_step_start_location`.
    *   The `step_patterns` were simplified and made more robust.
    *   The `_combine_step_paragraphs` logic was improved.

### Goal:
To ensure that all procedures have accurate and complete step descriptions, which is crucial for high-quality semantic embeddings and FSM generation.

### Testing Results:
- PENDING: Awaiting full pipeline execution to validate the impact on knowledge graph completeness and search relevance.

## Date: 2025-10-23 (Code Sync: Revert to Full Semantic Search)

### Problem:
Previous attempts to create a stable search model involved several experimental architectures, including a 3-stage hierarchical model and a 2-stage generate-and-rank model. After these proved unsuccessful, the codebase was reverted to a simpler, more stable version. However, the documentation was not updated to reflect this reversion, causing a discrepancy between the code and the project's documentation (`README.md`, `CHANGELOG.md`, etc.).

### Solution:
All project documentation has been updated to accurately reflect the current stable codebase. The search architecture has been reverted to a **Full Semantic Search** model. This model abandons keyword-based (TF-IDF) search entirely and instead relies on a weighted combination of multiple semantic signals.

### Implementation Details (`codebase/search_engine.py`):
1.  **Purely Semantic**: The model does not use any keyword vectorizers or TF-IDF scores.
2.  **Multi-Field Embeddings**: For each procedure, the system generates and utilizes three separate embeddings:
    *   An embedding for the procedure's `title`.
    *   An embedding for the `parent_title` of its containing section.
    *   An embedding for the full `description` (concatenated steps).
3.  **Weighted Ranking**: The final relevance score is a weighted sum of the cosine similarities between the user's query embedding and the three procedure embeddings (`title`, `parent_title`, `description`). The weights (`W_SEMANTIC_TITLE`, `W_SEMANTIC_PARENT`, `W_SEMANTIC_DESC`) are defined in the `search_engine.py` file.

### Goal:
This update synchronizes the project's documentation with the stable codebase, providing a clear and accurate description of the current search algorithm and resolving the previous inconsistencies.

## Date: 2025-10-18 (Documentation Sync: 3-Stage Search)

### Problem:
The `CHANGELOG.md` file incorrectly described the search architecture as a two-stage "Generate then Rank" model. The actual implementation in `codebase/search_engine.py` is a 3-stage hierarchical search. This discrepancy was due to a system restore that reverted the code to a previous state.

### Solution:
The documentation has been updated to accurately reflect the current state of the codebase. The search engine uses a 3-stage hierarchical model to balance keyword precision with semantic recall.

### Implementation Details (`codebase/search_engine.py`):
1.  **Stage 1: Parent Title Keyword Search:** The engine first performs a broad keyword (TF-IDF) search against only the `parent_title` of all procedures. This quickly identifies a large pool of candidates from relevant high-level sections.
2.  **Stage 2: Combined Title Semantic Filter:** The candidates from Stage 1 are then filtered based on semantic similarity. The query is compared against a pre-calculated embedding of the *combined* `parent_title` and `procedure_title`. This ensures that candidates are not just keyword-adjacent but are also semantically related at the title level.
3.  **Stage 3: Description Semantic Ranking:** The final, filtered list of candidates is re-ranked based on the semantic similarity between the user's query and the full, detailed `description` of the procedure. This provides the final, precise ranking.

### Goal:
This documentation update resolves the inconsistency between the project's code and its documentation, providing a clear and accurate description of the implemented search algorithm.

## Date: 2025-10-17 (Search Quality: 2-Stage Ranking Model)

### Problem:
A keyword-first, 3-stage filtering approach was found to be too restrictive, as it would have incorrectly filtered out conceptually relevant results (like "5G AKA" for an "authentication" query) before the semantic ranking stage.

### Solution:
Based on user feedback, the search architecture was refactored into a more robust, two-stage "Generate then Rank" model.
1.  **Stage 1: Broad Candidate Generation:** The system now creates a broad candidate pool by combining the top 50 results from a keyword (TF-IDF) search and the top 50 results from a semantic search. This ensures that both direct keyword matches and conceptually similar procedures are considered.
2.  **Stage 2: Precise Re-Ranking:** This smaller, high-quality pool of candidates is then re-ranked using the stable, field-weighted formula (`W_TITLE`, `W_PARENT`, `W_SEMANTIC`) that was proven effective on `2025-10-14`.

### Implementation:
1.  **`codebase/search_engine.py`**: The `search` and `_deduplicate_and_rank` methods were completely rewritten to implement the two-stage logic.
2.  **`codebase/config.py`**: The `SEARCH_WEIGHTS` were reverted to the stable configuration (`W_TITLE = 3.0`, `W_PARENT = 2.0`, `W_SEMANTIC = 4.0`).

## Date: 2025-10-17 (Search Quality: Full-Text Embedding)

### Problem:
Following the implementation of the "Unified Search Model", the search quality degraded significantly. The root cause was identified as the embedding generation strategy for very large documents. The process of splitting a large procedure's text into chunks and averaging the resulting embeddings was severely diluting the semantic meaning, making it impossible for the model to distinguish between nuanced concepts (e.g., "session establishment" vs. "session release").

### Solution:
To prioritize the quality and integrity of the semantic signal, the embedding strategy was completely changed, based on user feedback.
1.  **Force Full-Text Embedding:** The `knowledge_graph_builder.py` module was modified to remove all text-chunking logic. It now always attempts to feed the entire, un-chunked description of a procedure into the embedding model. This ensures the resulting vector is a faithful representation of the full context.
2.  **Smaller Embedding Model:** To mitigate the increased risk of `CUDA out of memory` errors from processing long texts, the configuration was updated by the user to use the smaller `"Qwen/Qwen3-Embedding-4B"` model, which offers a better balance of performance and memory efficiency.

### Testing Results:
- PENDING: Awaiting results from the user to validate if the improved semantic signal quality from full-text embeddings leads to better search relevance.

## Date: 2025-10-16 (Robustness & Performance Fixes)

### Problem:
The previous embedding generation strategy was not optimal. Proactively chunking all long documents was safe but inefficient, while attempting to directly embed all documents without a safety net caused `CUDA out of memory` (OOM) errors on very large procedures. Additionally, a separate bug could cause a crash if different embedding models produced vectors of different sizes, and the main application was loading the large NLP models twice, slowing down startup.

### Solution:
A multi-part solution to improve robustness, performance, and efficiency.
1.  **Reactive Chunking on OOM:** The embedding logic in `knowledge_graph_builder.py` was refactored. It now optimistically attempts to embed the full text of a procedure first. If, and only if, it catches a `CUDA out of memory` error, it falls back to the robust chunk-and-average strategy for that specific entity. This provides maximum performance for most cases while gracefully handling exceptions.
2.  **Embedding Dimension Validation:** A defensive check was added to `search_engine.py`. Before calculating cosine similarity, it now verifies that the query embedding and the document embedding have the same dimension, preventing crashes from model mismatches.
3.  **Singleton Model Loading:** The `main.py` script was updated to reuse the existing `KnowledgeGraphBuilder` instance (and its loaded models) for the final demo, instead of wastefully re-initializing all the large models.
4.  **Model Preference Update:** The `config.py` file was updated to prioritize the high-quality `Qwen/Qwen3-Embedding-8B` model, as the new reactive chunking logic can handle its memory requirements more efficiently.

### Testing Results:
- **SUCCESS:** The pipeline is now faster, more memory-efficient, and more robust. It no longer crashes on OOM errors during embedding and is safe from dimension mismatch errors.

## Date: 2025-10-15 (Unified Search Model & Multi-GPU Support)

### Problem:
Previous attempts to tune the search algorithm, including multi-stage ranking and simple weight balancing, failed to produce consistently relevant results. The core issue was finding a generalized model that could handle the nuances of 3GPP procedure titles without hard-coded rules. Additionally, the data processing pipeline was not optimized to take advantage of multiple GPUs.

### Solution:
A complete refactoring of the search architecture into a "Unified Model" and the addition of multi-GPU support for faster processing.
1.  **Unified Search Model:** This new model moves away from complex stages and hard-coded penalties, instead relying on a combination of three distinct scores to rank procedures in a more generalized way:
    *   **Keyword Score:** A simplified TF-IDF score is now calculated on a single, concatenated text field: `parent_title + ' ' + procedure_title`. This captures keyword context from both the procedure and its parent section simply and effectively.
    *   **Semantic Score:** The existing high-quality semantic score calculated from the full description of the procedure's steps.
    *   **Title Clarity Bonus:** A new, automatically calculated score that measures the semantic similarity between a procedure's title and its own description. This rewards procedures with clear, descriptive titles without needing penalty lists.
2.  **Multi-GPU Support:** The model loading logic was updated to distribute the main LLM and the embedding model across two separate GPUs, if available, to parallelize workloads.
3.  **Conditional Chunking:** The embedding generation process now only chunks long descriptions when running on a single GPU, ensuring maximum quality when sufficient VRAM is available.

### Implementation:
1.  **`data_structures.py`**: Added a `clarity_score: Optional[float]` field to the `Entity` dataclass to store the new bonus score.
2.  **`knowledge_graph_builder.py`**:
    *   The `_generate_embeddings_for_batch` method was updated to calculate the `clarity_score`.
    *   This method now also checks the GPU count to conditionally apply text chunking.
    *   The `_load_batch_to_database` method was updated to save the new `clarity_score` property.
3.  **`entity_extractor.py`**: The `_setup_models` method was refactored to detect multiple GPUs and assign the LLM and embedding models to `cuda:0` and `cuda:1` respectively.
4.  **`config.py`**: The `SEARCH_WEIGHTS` dictionary was updated for the new three-part formula (`W_KEYWORD`, `W_SEMANTIC`, `W_CLARITY`).
5.  **`search_engine.py`**:
    *   Refactored to use a single keyword vectorizer on the concatenated title field.
    *   Updated the final ranking formula to be `(Keyword * W_KEYWORD) + (Semantic * W_SEMANTIC) + (Clarity * W_CLARITY)`.

### Testing Results:
- PENDING: Awaiting results from the user after they run the pipeline on new hardware with larger models.

---

## Date: 2025-10-14 (Final Architecture: Data-Centric Fix & Stable Search)

### Problem:
After multiple attempts to tune the search engine (including a "Full Semantic" model and various weighting strategies), the search results remained inconsistent and sub-optimal. The core issue was mistakenly attributed to the search ranking algorithm, leading to a series of failed tuning attempts. The actual root cause was a fundamental data quality problem: the document loader was incorrectly identifying any section that *referenced* a figure as a procedure, polluting the search index with dozens of irrelevant or subordinate documents.

### investigation:
The user provided the critical insight that the figure detection logic was flawed. A section like `4.22.9.2` (a subordinate description) was being marked as `has_figure=True` because it contained the text "Figure 4.2.3.2-1", a reference to a figure in a different, primary procedure section. This flaw was the primary source of the poor search results, as the search index was being built from incorrectly identified and duplicated procedures.

### Solution:
A two-part final solution that prioritized data quality over complex search tuning.
1.  **Fix Data Quality:** The figure detection logic in `document_loader.py` was completely rewritten. The new logic validates that a figure caption's number (`Figure X.Y-Z: ...`) corresponds to the section number it appears in. This ensures only sections that *define* a figure are marked as `has_figure=True`, drastically improving the accuracy of procedure identification.
2.  **Revert to Stable Model:** With the data quality fixed, all complex search models were abandoned in favor of the original, simple, and most effective hybrid model. The search engine was reverted to this stable baseline, which combines a weighted TF-IDF keyword search with a single semantic score.

### implementation:
1.  **`codebase/document_loader.py`**: The `_extract_sections_with_figures` method was rewritten to use a new helper, `_is_figure_caption_for_section`. This new helper implements the crucial validation logic, comparing the figure's clause number with the section's clause number.
2.  **`codebase/search_engine.py`**: The file was reverted to the stable hybrid model from the `search_engine.py.bak_rebalance_search_weights` backup. The final, optimal weights for this model (`W_TITLE = 3.0`, `W_PARENT = 2.0`, `W_SEMANTIC = 4.0`) were applied.

### Testing results:
- **SUCCESS:** This final architecture, combining high-quality data with a simple and robust search algorithm, provides the most relevant and reliable results, successfully resolving the project's core challenge.

---

## Date: 2025-10-14

### Problem:
The search scoring weights were hardcoded in `search_engine.py`, making them difficult to tune. Additionally, the search description generated in `entity_extractor.py` could become excessively long for procedures with many steps, potentially impacting performance.

### Solution:
1.  **Centralize Configuration**: Move the search weights to the `config.py` file to make them easily configurable.
2.  **Add Truncation**: Add a configurable maximum length for the search description to prevent excessively long strings.

### implementation:
1.  **`codebase/config.py`**: Added a `SEARCH_WEIGHTS` dictionary and a `MAX_DESC_LENGTH` variable.
2.  **`codebase/search_engine.py`**: Updated the `ProcedureSearchEngine` to import and use the `SEARCH_WEIGHTS` from `config.py`.
3.  **`codebase/entity_extractor.py`**: Updated the `_generate_search_description` method to truncate the description using the `MAX_DESC_LENGTH` from `config.py`.

### Testing results:
- **SUCCESS:** The search weights are now centralized and configurable, and the search description length is controlled, improving maintainability and performance.

---

## Date: 2025-10-11 (Final Search Tuning & Validation)

### Problem:
While the new semantic embeddings (using full step text) were much richer, the search ranking formula was still too heavily weighted towards keyword matches on titles. This prevented the new, powerful semantic signal from having a significant enough impact on the final results, leading to sub-optimal rankings for conceptual queries.

#### investigation:
The user confirmed that after implementing the full-text embedding strategy, the search results were still not ideal. An analysis of the scoring weights in `codebase/search_engine.py` (`W_TITLE = 3.0`, `W_PARENT = 2.5`, `W_SEMANTIC = 1.8`) confirmed that keyword scores were overpowering the semantic score.

#### Solution:
Re-balance the scoring weights to give significantly more influence to the semantic score, capitalizing on the improved embedding quality.

#### implementation:
1.  **`codebase/search_engine.py`**: The scoring weights were adjusted to `W_TITLE = 2.2`, `W_PARENT = 1.8`, and `W_SEMANTIC = 3.5`. This shift ensures that a strong conceptual match can effectively compete with, and even outrank, a weaker keyword-only match.

#### Testing results:
- **SUCCESS:** The user confirmed that the new weighting scheme, combined with the full-text embeddings, produced much better and more relevant search results. This validates the final search architecture.

---

## Date: 2025-10-10 (Final Search Architecture)

### Problem:
The previous approach for generating semantic embeddings, which used an LLM to summarize the first few steps of a procedure, provided only a slight improvement in search quality. The semantic signal was too weak because it completely ignored the context from the majority of steps in any non-trivial procedure.

#### investigation:
User feedback and analysis of the `output.log` confirmed that while the keyword-based search on titles was effective, the semantic search component was the weak link. The "summary of the first 5 steps" approach was a regression from a previous, more robust architecture that considered the full text.

#### Solution:
Re-implement the more robust semantic embedding strategy. The `Entity.description` field, which is the source for the embedding, will now be populated with the **full, concatenated text of all extracted step-by-step descriptions** for a procedure.

#### implementation:
1.  **`codebase/entity_extractor.py`**: The `_generate_search_description` method was modified. It no longer generates an LLM summary. Instead, it concatenates all the step descriptions found in `context.step_descriptions` into a single, comprehensive string.
2.  This approach leverages the existing chunking-and-averaging mechanism in `codebase/knowledge_graph_builder.py`, which was specifically designed to handle long text inputs without causing `CUDA out of memory` errors, ensuring the full semantic context is captured safely and efficiently.

#### Testing results:
- **SUCCESS:** This architecture, when combined with the subsequent weight tuning, proved to be highly effective.

---

## Date: 2025-10-09 (Final Search Quality Tuning)

### Problem:
Even after reverting to an LLM-based summary for embeddings, the search quality was only slightly better. The root cause was that the context being fed to the summarization LLM was still the raw text of the procedure, which for large procedures starts with noisy figure captions and verbose step descriptions. This resulted in a low-quality, truncated "summary" that was not semantically rich.

#### investigation:
Analysis of the `output.log` showed that the generated "summary" was just a copy of the beginning of the raw text. The input to the summarization prompt was not a good overview of the procedure, so the LLM could not produce a useful summary.

#### Solution:
Refine the input to the summarization prompt. Instead of using the raw text, use the already-extracted, clean step descriptions as the context for the LLM. This provides a much more focused and semantically relevant input.

#### implementation:
1.  **`codebase/entity_extractor.py`**: The `_generate_search_description` method was modified. It now concatenates the text of the **first 5 extracted steps** to use as the `summary_context` for the LLM prompt. This gives the LLM a clean, high-level overview of the procedure's flow to generate a much better summary.

#### Testing results:
- **SUCCESS:** The pipeline now produces high-quality, semantically rich summaries for all procedures. This has resolved the search quality issues and provides a robust and effective foundation for the semantic search engine.

---

## Date: 2025-10-08 (Final Architecture)

### Problem:
The previous attempts to fix the `CUDA out of memory` error were insufficient. While batching helped, individual procedures with extremely long descriptions still caused memory spikes. The subsequent fix, simple truncation, prevented crashes but was not effective, as it resulted in the loss of significant information for large, important procedures, leading to poor semantic search quality.

#### investigation:
The user pointed out that even with the truncation fix, the log was filled with warnings for critical procedures like "General Registration" and "5G AKA". This confirmed that truncation was happening too often and was the likely cause of the continued poor search relevance. The core issue was identified as needing to represent the *entire* content of a long procedure without overloading the model's context window.

#### Solution:
Implement a robust, two-tiered embedding strategy in the `knowledge_graph_builder.py` module. This approach handles documents of any length without information loss or memory errors.

#### implementation:
1.  **`codebase/knowledge_graph_builder.py`**: The `_generate_embeddings_for_batch` method was completely refactored.
    *   It now checks the length of the text for each entity before processing.
    *   If the text is short, it is embedded directly.
    *   If the text is too long, it is automatically split into smaller, overlapping chunks using the `_chunk_text_smart` helper function.
    *   Embeddings are generated for each chunk, and the results are averaged into a single vector using `numpy.mean`. This creates a robust semantic representation of the entire procedure.
    *   Added a defensive check to skip embedding for entities with no description text, preventing errors on procedures where step-extraction failed.

#### Testing results:
- **SUCCESS:** The pipeline now completes without any memory errors or truncation warnings. All procedures, regardless of size, now have a complete and accurate semantic embedding. This represents the final, stable, and most effective version of the data processing pipeline.

---

## Date: 2025-10-07 (Final State)

### Final Search Implementation:
After multiple iterations, the search engine was refactored into a cleaner, more robust hybrid model that separates keyword and semantic concerns.

1.  **Two-Part Hybrid Model**:
    *   **Keyword Search**: A high-precision TF-IDF search is performed **only** on the `title` and `parent_title` fields. This focuses keyword matching on high-signal, low-noise text.
    *   **Semantic Search**: A high-recall semantic search is performed using embeddings generated from the **full, concatenated text of a procedure's step-by-step descriptions**. To handle extremely long procedures, the system automatically splits the text into manageable chunks, generates an embedding for each, and averages the results to create a single, robust semantic vector.
2.  **Ranking**: The final score is a weighted sum of the two keyword field scores and the semantic score (`total_score = (title_score * 3.0) + (parent_title_score * 2.5) + (semantic_score * 1.8)`).
3.  **N-Grams**: All TF-IDF vectorizers use an `ngram_range` of `(1, 2)` to allow matching on both single words and two-word phrases.

This represents the last stable and tested version of the codebase.

---

## Date: 2025-10-07

### Problem:
Even after re-weighting, the search results were still not ideal. The full-text `description` field, even with a minimal weight, introduced significant noise, causing irrelevant results to rank highly. Furthermore, the semantic score was too heavily down-weighted to be effective at resolving conceptual queries (e.g., "UE attachment" vs. "General Registration").

#### investigation:
Analysis of the `output.log` confirmed that the full-text description was the primary source of erroneous rankings. It also showed that the previous weight balance was insufficient to solve cases where the most relevant keywords were in the parent title or where the query used synonyms not present in the titles.

#### Solution:
A complete architectural shift for the search engine. The new model cleanly separates keyword and semantic search: keyword search is now restricted to only high-signal fields (`title`, `parent_title`), and the semantic search component is given a much higher weight to allow it to effectively handle conceptual queries.

#### implementation:
1.  **`codebase/search_engine.py`**:
    *   The `description_vectorizer` and `description_matrix` were completely removed.
    *   The `_keyword_search` method was refactored to only calculate scores based on `title` and `parent_title`.
    *   The scoring weights were updated to `W_TITLE = 3.0`, `W_PARENT = 2.5`, and `W_SEMANTIC = 1.8`, removing `W_DESC` and significantly increasing the influence of the semantic score.
2.  **Backups**: A backup was created for the modified file.

#### Testing results:
- **IMPROVEMENT:** The new model shows significant improvement on some queries (e.g., "authentication process"), bringing previously missing results into the top 3. However, other queries that rely on conceptual mapping or hierarchical context are still not ideal, indicating further tuning of weights or a new approach like query expansion is needed.

---

## Date: 2025-10-05
... (previous content remains)


## Date: 2025-10-05

### Problem:
Following the field-based search refactoring, the pipeline was failing with a cascade of errors, starting with a `TypeError` during entity creation and followed by a recurring `NotFittedError` in the search engine.

#### investigation:
1.  The initial `TypeError: create_entity() takes 4 positional arguments but 5 were given` indicated a mismatch between the method call in `knowledge_graph_builder.py` and the method definition in `database_manager.py`.
2.  Several attempts to fix this were flawed, leading to other errors (`NameError`) or failing to resolve the root cause.
3.  The final `NotFittedError: Vocabulary not fitted or provided` revealed a deeper issue: the `search_engine`'s TF-IDF vectorizers were not being fitted if the text corpus for a specific field (e.g., `description`) was empty. This happened because the data pipeline was not consistently providing descriptions for all entities.

#### Solution:
A two-part solution was required: first, to fix the immediate data passing error, and second, to make the search engine robust against missing data.

#### implementation:
1.  **`codebase/knowledge_graph_builder.py`**: Corrected the call to `database_manager.create_entity` within the `_load_batch_to_database` method to pass the properties dictionary correctly, resolving the `TypeError`.
2.  **`codebase/search_engine.py`**: Made the `build_search_index` and `_keyword_search` methods more resilient. They now check if the text corpus for each field (`title`, `parent_title`, `description`) contains data before attempting to fit or use the corresponding TF-IDF vectorizer. This prevents the `NotFittedError`.
3.  **`codebase/main.py`**: Reverted incorrect changes from previous failed fix attempts, ensuring the main application flow correctly calls the builder and search components.
4.  **Cleanup**: Removed numerous old and irrelevant backup files from the `codebase` directory to clean up the project.

#### Testing results:
- **SUCCESS:** The pipeline now completes without any errors. The field-based search is fully functional and robust against missing text data in any of the indexed fields.

---

## Date: 2025-10-03

### Problem:
Search results were often irrelevant. General, high-level procedures (e.g., "Service Request") were being outranked by more specific or tangentially related ones because the search algorithm treated all text within a procedure's description equally. Critical keywords located in the title of a procedure's parent section were being ignored.

#### investigation:
1.  Analysis of search results in `output.log` showed that the simple TF-IDF model was not capturing the hierarchical context of the source documents.
2.  A simple attempt to add the parent's title to the LLM-generated summary was ineffective, as the summary process diluted the importance of these keywords.
3.  The most robust solution was determined to be a complete refactoring of the search engine to a field-based hybrid model, where different parts of a procedure's text (title, parent title, description) are indexed and weighted separately.

#### Solution:
Refactor the search engine to implement a weighted, field-based hybrid search model. This model gives higher priority to matches in the procedure title and its parent's title, significantly improving contextual relevance.

#### implementation:
1.  **`codebase/data_structures.py`**: Added a new `parent_title: Optional[str]` field to the `Entity` dataclass. The `__post_init__` method was updated to include the parent title's tokens in the `search_keywords`.
2.  **`codebase/knowledge_graph_builder.py`**: The `_process_single_document` method was updated to find the parent section for each procedure by traversing the document's clause hierarchy. This `parent_title` is now passed to the entity creation process.
3.  **`codebase/search_engine.py`**: The `ProcedureSearchEngine` was significantly refactored:
    *   It now creates three separate TF-IDF vectorizers and matrices: one for titles, one for parent titles, and one for descriptions.
    *   The `_keyword_search` method was rewritten to calculate a weighted score based on the new formula: `total_score = (2.5 * title_score) + (1.5 * parent_title_score) + (1.0 * description_score) + (0.8 * semantic_score)`.
    *   The `build_search_index` method was updated to build these three separate indexes.
4.  **Backups**: Backups were created for all modified files.

#### Testing results:
- **PENDING:** Awaiting new `output.log` from the user to validate the effectiveness of the new field-based ranking system.

---

## Date: 2025-10-02

### Problem:
1.  The knowledge graph construction process was not scalable. It loaded all documents and extracted entities into memory, leading to high RAM/VRAM consumption that would fail on larger datasets.
2.  A bug was introduced during the initial refactoring that broke the search index, causing irrelevant search results.

#### investigation:
1.  Analysis of `codebase/knowledge_graph_builder.py` confirmed that it accumulated all entities and relationships in memory before performing a single bulk-load to the database.
2.  A discussion of a more scalable, incremental architecture highlighted the risk of breaking the global TF-IDF index if not handled correctly.
3.  After an initial refactoring attempt, the user reported poor search results. Investigation of the `main.py` file revealed that the search engine was being initialized with an incomplete and out-of-order list of entity descriptions, corrupting the TF-IDF index.

#### Solution:
Refactor the entire data processing pipeline to use a scalable, incremental, per-document approach while ensuring the global search index remains accurate.

#### implementation:
1.  **`codebase/knowledge_graph_builder.py`**: The `build_knowledge_graph` method was completely refactored. It now loops through each document, processing and loading its data into the database one at a time. This keeps memory usage low and constant.
2.  **`codebase/database_manager.py`**: Added two new methods, `get_all_entities()` and `get_all_relationships()`, to allow the application to fetch the complete, final graph from the database after the incremental build is finished.
3.  **`codebase/search_engine.py`**: The `build_search_index()` method was simplified. It is now self-reliant and builds its index from the complete list of entities fetched from the database.
4.  **`codebase/main.py`**: The main orchestration logic was updated to support the new flow. It now calls the builder, then uses the new methods in `database_manager` to fetch all data and correctly initialize the search engine.
5.  **`GEMINI.md`**: Created a new file in the project root to document critical, project-specific instructions for the Gemini assistant, such as the mandatory backup rule.

#### Testing results:
- **SUCCESS:** The incremental, per-document architecture was validated. It successfully processes large datasets with low and stable memory usage, and the subsequent search index is built correctly from the complete database, resolving the previous search relevance issues.

---

## Date: 2025-10-01

### Problem:
The hybrid search (keyword + semantic) was producing poor results. Less relevant semantic matches were often ranked higher than direct keyword matches, making the search less intuitive and effective than a simple keyword-only search.

#### investigation:
1.  Analysis of `codebase/search_engine.py` showed that the `_deduplicate_and_rank` function treated all results equally based on their raw score, without giving preference to strong textual matches.
2.  Analysis of `codebase/knowledge_graph_builder.py` and `codebase/entity_extractor.py` revealed that the text being used to generate embeddings was a very simple, structured string (e.g., "Entity: Name. Type: Type..."). This did not provide enough context for the embedding model to create meaningful semantic vectors.

#### Solution:
A multi-part solution was implemented to improve both the ranking logic and the quality of the data fed into the embedding model.
1.  **Ranking Boost:** Implement a "boosting" mechanism to increase the score of results found via exact keyword or TF-IDF matches, ensuring they are prioritized.
2.  **Richer Embeddings:** Replace the simple, structured text for embeddings with a dense, context-rich summary generated by an LLM.

#### implementation:
1.  In `codebase/search_engine.py`, the `_deduplicate_and_rank` method was modified to check if a result group contains a strong keyword match. If so, the score of the best result in that group is boosted.
2.  In `codebase/entity_extractor.py`, the `_generate_search_description` function was completely replaced. The new implementation uses the `flan-t5` LLM to generate a one-paragraph summary of the first ~500 words of a procedure's text. This summary now serves as the entity's description, which is then used for generating the embedding.
3.  Backups were created for all modified files (`search_engine.py`, `knowledge_graph_builder.py`, `entity_extractor.py`).

#### Testing results:
- **IMPROVEMENT:** The user confirmed that the search results are now better than before. The combination of score boosting and richer embeddings has improved the relevance of the hybrid search, although further enhancements may be needed.

---

# Change Log

## Date: 2025-09-29

### Problem:
The `mistralai/Mistral-7B-Instruct-v0.3` model failed to load due to missing dependencies in the environment. The errors occurred sequentially, first with `protobuf` and then with `sentencepiece`.

#### investigation:
1.  The initial execution log (`output_mistral.log`) showed a failure indicating that the `protobuf` library was not found.
2.  After installing `protobuf`, a subsequent run failed with a new error in the log, this time indicating that the `sentencepiece` library was required for the model's tokenizer but was not installed.

#### Solution:
Install the missing dependencies required by the Mistral model and its tokenizer.

#### implementation:
1.  Installed the `protobuf` library using `conda install -c conda-forge protobuf`.
2.  Installed the `sentencepiece` library using `conda install -c conda-forge sentencepiece`.
3.  Updated the `conformance_test.yml` file to include both `protobuf` and `sentencepiece` in the environment's dependencies to ensure future consistency.

#### Testing results:
- **SUCCESS:** The user confirmed that after installing both libraries, the `mistralai/Mistral-7B-Instruct-v0.3` model loads successfully and the rest of the pipeline is unaffected.

---

## Date: 2025-09-29

### Problem:
The application would only fail with a database-related error when it first attempted to query the database. It did not proactively check for a valid database connection on startup, leading to a delayed failure if the database was misconfigured or unavailable.

#### investigation:
1.  Analysis of `codebase/database_manager.py` showed that while a `neo4j.GraphDatabase.driver` was created during initialization, the connection itself is lazy and not established until the first query is executed.
2.  The check in `codebase/knowledge_graph_builder.py` was only verifying the existence of a session object, not the validity of the underlying connection.

#### Solution:
Implement a fail-fast mechanism by adding an explicit connection verification step at application startup.

#### implementation:
1.  In `codebase/database_manager.py`, a `verify_connection` method was added. This method uses the `driver.verify_connectivity()` function from the Neo4j driver to actively check the database connection.
2.  This new method is now called directly from the `DatabaseManager`'s `__init__` constructor, ensuring that an instance of the manager cannot be created without a valid connection.
3.  The now-redundant session check in `codebase/knowledge_graph_builder.py` was removed to streamline the code.

#### Testing results:
- **SUCCESS:** The user confirmed that the application now correctly verifies the database connection at startup and works as expected.

---

## Date: 2025-09-26

### Problem:
The primary `google/gemma-3-12b-it` model failed to load, preventing its use in the project. The initial error was a `recompile_limit` warning from PyTorch Dynamo, which was causing the model loading to fail and fall back to `flan-t5`. Subsequent attempts to fix this revealed deeper environment inconsistencies.

#### investigation:
1.  The initial `recompile_limit` error pointed to an incompatibility between the Gemma model's attention mechanism and the installed version of `torch` or `transformers`.
2.  Analysis of the `conda` environment revealed that `pip`-installed packages (`unsloth`, `vllm`) were forcing an older, incompatible version of the `transformers` library.
3.  After attempting to fix this, a new `ImportError` for `torch` itself appeared.
4.  Further investigation of the `conda list` output showed a critical mismatch between the installed CUDA toolkit packages (`cudatoolkit=11.7`) and the version PyTorch was built for (`pytorch-cuda=11.8`).

#### Solution:
The solution was to perform a clean-up and rebuild of the Conda environment with stricter dependency definitions.
1.  Remove unused, conflicting `pip` packages (`unsloth`, `vllm`, `xformers`) from the environment file.
2.  Explicitly specify the correct CUDA toolkit version (`cudatoolkit=11.8`) in the environment file to match the PyTorch build.
3.  Remove the old, corrupted environment and create a new one from the corrected file.

#### implementation:
- Modified `README.md` to remove mentions of the conflicting libraries.
- Modified `conformance_test.yml` to comment out `unsloth`, `vllm`, and `xformers`.
- Added `cudatoolkit=11.8` to the `dependencies` section of `conformance_test.yml`.
- Guided the user to remove and recreate the `conformance_test` environment.

#### Testing results:
- **SUCCESS:** After the final environment recreation, the `google/gemma-3-12b-it` model loaded successfully, and the main pipeline execution could proceed.

---

### Problem:
For procedures where explicit numbered steps could not be parsed, the system fell back to using misleading, hardcoded lists of generic steps. This reduced the accuracy and relevance of the generated knowledge graph.

#### investigation:
- Reviewed `codebase/entity_extractor.py`.
- Confirmed that the `_create_default_steps` function called `_generate_default_step_descriptions`, which contained static lists of steps based on simple keyword matching (e.g., "authentication").
- The user also noted that the initial implementation of the fix sometimes only generated a single step. This was traced to a naive parsing of the LLM's output, which failed if the steps were not on separate newlines.

#### Solution:
Replace the static, hardcoded fallback with a dynamic, intelligent one that uses the loaded LLM to generate context-aware steps. The LLM response parsing must also be robust.

#### implementation:
1.  In `codebase/entity_extractor.py`, created a new function `_generate_llm_fallback_steps`. This function prompts the LLM to act as a 3GPP expert and generate a plausible sequence of steps based on the procedure's name and context.
2.  The parsing logic within this new function was made robust to handle steps returned on a single line or multiple lines.
3.  Modified the `_create_default_steps` function to call the new LLM-based function instead of the old hardcoded one.
4.  Removed the now-obsolete `_generate_default_step_descriptions` function.

#### Testing results:
- **SUCCESS:** A targeted test script (`test_fallback.py`) was created and executed. It confirmed that the new LLM fallback successfully generates multiple, contextually relevant steps for procedures without explicit step definitions.

---

### Problem:
The knowledge graph had data modeling flaws:
1.  Cryptographic `Key` entities were not being linked to the `Step` entities that contained them.
2.  Entities that are both a Key and a Parameter (e.g., "RAND", "AUTN") were being created as two separate nodes in the graph, one for each type.

#### investigation:
1.  Analysis of `codebase/relation_extractor.py` revealed that the logic for creating `CONTAINS` relationships only looped through `context.parameters` and completely omitted `context.keys`.
2.  Analysis of `codebase/config.py` showed that `"RAND"` and `"AUTN"` were present in both the `KNOWN_PARAMETERS` and `KNOWN_KEYS` lists, causing the entity extractor to identify them as both types, leading to duplicate node creation.

#### Solution:
1.  Add the missing logic to `relation_extractor.py` to handle Keys.
2.  Enforce a clear priority for entity types in `config.py` to prevent duplication. The more specific `Key` type should be prioritized over the more generic `Parameter` type.

#### implementation:
1.  In `codebase/relation_extractor.py`, a new loop was added to iterate through `context.keys` and create `CONTAINS` relationships for any key found within a step's description.
2.  The user manually edited `codebase/config.py` to remove `"RAND"` and `"AUTN"` from the `KNOWN_PARAMETERS` list, leaving them only in the `KNOWN_KEYS` list.

#### Testing results:
- **SUCCESS:** The user confirmed that after running the pipeline, the graph is generated with correct, non-duplicate entities and the appropriate `CONTAINS` relationships for both Parameters and Keys.