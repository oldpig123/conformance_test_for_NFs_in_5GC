from pathlib import Path
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
from tqdm import tqdm
import warnings
import re
import gc
import torch
warnings.filterwarnings("ignore")

from data_structures import Entity, Relationship, ProcedureContext
from document_loader import DocumentLoader
from entity_extractor import EntityExtractor
from relation_extractor import RelationExtractor
from database_manager import DatabaseManager

class KnowledgeGraphBuilder:
    """Main knowledge graph construction pipeline with incremental processing."""
    
    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str):
        print("Initializing Knowledge Graph Builder...")
        self.entity_extractor = EntityExtractor()
        self.document_loader = DocumentLoader(self.entity_extractor.text_generator)
        self.relation_extractor = RelationExtractor(self.entity_extractor.text_generator)
        self.database_manager = DatabaseManager(neo4j_uri, neo4j_user, neo4j_password)
        self.procedure_contexts: List[ProcedureContext] = [] # Still useful for context
        print("✓ Knowledge Graph Builder initialized.")

    def build_knowledge_graph(self, file_paths: List[Path]):
        """
        Main pipeline with incremental processing.
        Processes one document at a time to manage memory.
        """
        print("\n" + "="*70)
        print("  3GPP KNOWLEDGE GRAPH CONSTRUCTION PIPELINE (Incremental Mode)")
        print("="*70)
        
        self.database_manager.clear_database()
        print("✓ Database cleared for new build.")
        
        try:
            all_sections = self.document_loader.load_documents(file_paths)

            for file_path in file_paths:
                print(f"\n--- Processing Document: {file_path.name} ---")
                
                # 1. Process a single document to get its entities and relationships
                doc_entities, doc_relationships = self._process_single_document(all_sections, file_path.stem)

                if not doc_entities:
                    print(f"  No entities found in {file_path.name}, skipping.")
                    continue

                # 2. Generate embeddings for this document's entities only
                self._generate_embeddings_for_batch(doc_entities)

                # 3. Load this document's data into the database
                self._load_batch_to_database(doc_entities, doc_relationships)

                # 4. Free up memory before the next document
                del doc_entities, doc_relationships
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            self._print_final_statistics() # Print stats at the end
            
        except Exception as e:
            print(f"Error in pipeline: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # The database manager connection is now closed by the caller (main.py)
            pass

    def _process_single_document(self, all_sections: List, document_name: str) -> Tuple[List[Entity], List[Relationship]]:
        """Processes a single document and returns its entities and relationships."""
        document_sections = [s for s in all_sections if s.document == document_name]
        procedure_sections = self.document_loader.identify_procedure_sections_with_llm(document_sections)
        
        if not procedure_sections:
            return [], []

        doc_entities: Dict[tuple, Entity] = {}
        doc_relationships: List[Relationship] = []

        # Create a quick lookup for sections by clause for parent lookup
        sections_by_clause = {s.clause: s for s in all_sections}

        for section in procedure_sections:
            print(f"  Procedure: {section.procedure_name}")

            # Find parent title using the clause hierarchy
            parent_title = None
            if '.' in section.clause:
                parent_clause = '.'.join(section.clause.split('.')[:-1])
                parent_section = sections_by_clause.get(parent_clause)
                if parent_section:
                    parent_title = parent_section.title
                    print(f"    Found parent section for '{section.clause}': '{parent_clause}' ({parent_title})")

            # Pass the temporary collections and the found parent_title
            self._process_procedure_section(section, doc_entities, doc_relationships, parent_title)
        
        return list(doc_entities.values()), doc_relationships

    def _process_procedure_section(self, section, doc_entities: Dict, doc_relationships: List, parent_title: Optional[str] = None):
        """Processes a single procedure, adding its data to the provided collections."""
        context = ProcedureContext(section.procedure_name, section)
        
        # Add procedure entity to the document's entity collection, now with parent_title
        self._add_entity(context.procedure_name, "Procedure", {
            "clause": section.clause, "document": section.document, "has_figure": section.has_figure
        }, None, doc_entities, parent_title=parent_title)
        
        entity_result = self.entity_extractor.extract_entities_for_procedure(context, parent_title=parent_title)
        
        if entity_result.success:
            # Update the procedure entity with its summary
            procedure_key = (context.procedure_name, "Procedure")
            if procedure_key in doc_entities and context.search_description:
                proc_entity = doc_entities[procedure_key]
                proc_entity.description = context.search_description
                proc_entity.__post_init__() # Regenerate keywords

            # Add extracted entities to the document's collection
            self._add_entities_from_context(context, doc_entities)
            
            # Extract and add relationships to the document's collection
            relationships = self.relation_extractor.extract_relationships_for_procedure(context)
            doc_relationships.extend(relationships)
            
            self.procedure_contexts.append(context) # Keep for final stats
        else:
            print(f"    ✗ Entity extraction failed: {entity_result.error_message}")

    def _add_entities_from_context(self, context: ProcedureContext, doc_entities: Dict):
        """Adds entities from a procedure context to the document's entity collection."""
        common_props = {"procedure": context.procedure_name, "clause": context.section.clause}
        for nf in context.network_functions:
            self._add_entity(nf, "NetworkFunction", common_props, None, doc_entities)
        for msg in context.messages:
            self._add_entity(msg, "Message", common_props, None, doc_entities)
        for param in context.parameters:
            self._add_entity(param, "Parameter", common_props, None, doc_entities)
        for key in context.keys:
            self._add_entity(key, "Key", common_props, None, doc_entities)
        
        for step_name in context.steps:
            step_match = re.search(r'_step_([\w-]+)$', step_name)
            step_number_str = step_match.group(1) if step_match else "1"
            step_description = context.step_descriptions.get(step_name, "")
            step_props = {**common_props, "step_number": step_number_str}
            self._add_entity(step_name, "Step", step_props, step_description, doc_entities)

    def _add_entity(self, name: str, entity_type: str, properties: Dict, description: Optional[str], entity_collection: Dict, parent_title: Optional[str] = None):
        """Adds a single entity to a specified collection, avoiding duplicates."""
        key = (name, entity_type)
        if key not in entity_collection:
            properties['parent_title'] = parent_title
            entity_collection[key] = Entity(name=name, entity_type=entity_type, properties=properties, description=description, parent_title=parent_title)

    def _generate_embeddings_for_batch(self, entities_list: List[Entity]):
        """Generates embeddings for a given list (batch) of entities."""
        print(f"\n=== Generating Embeddings for batch of {len(entities_list)} entities ===")
        if not self.entity_extractor.embedding_model or not entities_list:
            return

        entity_texts = [f"Entity: {e.name}. Type: {e.entity_type}. Keywords: {' '.join(e.search_keywords)}. Description: {e.description or ''}" for e in entities_list]
        
        try:
            embeddings = self.entity_extractor.embedding_model.encode(
                entity_texts, batch_size=8, show_progress_bar=True, convert_to_tensor=False
            )
            for entity, embedding in zip(entities_list, embeddings):
                entity.embedding = embedding.tolist() # This updates the object in the list
            print(f"  ✓ Successfully generated embeddings for {len(entities_list)} entities.")
        except Exception as e:
            print(f"  ❌ Error during embedding generation: {e}")

    def _load_batch_to_database(self, entities_list: List[Entity], relationships_list: List[Relationship]):
        """Loads a batch of entities and relationships into the database."""
        print(f"\n=== Loading batch of {len(entities_list)} entities and {len(relationships_list)} relationships to DB ===")
        if not entities_list:
            return

        # Create entities
        for entity in tqdm(entities_list, desc="Creating entities"):
            # The embedding is now a property of the Entity object
            props_with_embedding = entity.properties.copy()
            if entity.embedding:
                props_with_embedding['embedding'] = entity.embedding
            self.database_manager.create_entity(entity.name, entity.entity_type, props_with_embedding)
        
        # Create relationships
        if relationships_list:
            for rel in tqdm(relationships_list, desc="Creating relationships"):
                self.database_manager.create_relationship(rel.source_name, rel.target_name, rel.rel_type, rel.properties)
        
        # Create indexes
        entity_types = set(entity.entity_type for entity in entities_list)
        self.database_manager.create_indexes(list(entity_types))
        print("✓ Batch loaded to Neo4j.")

    def _print_final_statistics(self):
        """Print comprehensive statistics at the end of the build process."""
        print("\n" + "="*70)
        print("  FINAL KNOWLEDGE GRAPH STATISTICS")
        print("="*70)
        
        # Since we don't hold all entities in memory, we can query the DB for stats
        # or accumulate them during the build. For now, we'll rely on procedure contexts.
        
        entity_counts = defaultdict(int)
        for ctx in self.procedure_contexts:
            entity_counts["Procedure"] += 1
            entity_counts["NetworkFunction"] += len(ctx.network_functions)
            entity_counts["Message"] += len(ctx.messages)
            entity_counts["Parameter"] += len(ctx.parameters)
            entity_counts["Key"] += len(ctx.keys)
            entity_counts["Step"] += len(ctx.steps)

        print("Total Entities Processed (approximate):")
        for entity_type, count in sorted(entity_counts.items()):
            print(f"  {entity_type:15}: {count:4d}")
        
        print(f"\nProcedures Processed: {len(self.procedure_contexts)}")
        print(f"Documents Processed: {len(set(ctx.section.document for ctx in self.procedure_contexts))}")
        print("="*70)