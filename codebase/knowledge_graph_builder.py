from pathlib import Path
from typing import List, Dict
from collections import defaultdict
from tqdm import tqdm
import warnings
import re
warnings.filterwarnings("ignore")

from data_structures import Entity, Relationship, ProcedureContext
from document_loader import DocumentLoader
from entity_extractor import EntityExtractor
from relation_extractor import RelationExtractor
from database_manager import DatabaseManager

class KnowledgeGraphBuilder:
    """Main knowledge graph construction pipeline following refined requirements."""
    
    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str):
        print("Initializing Knowledge Graph Builder...")
        
        # Initialize components in order
        self.entity_extractor = EntityExtractor()
        self.document_loader = DocumentLoader(self.entity_extractor.text_generator)
        self.relation_extractor = RelationExtractor(self.entity_extractor.text_generator)
        self.database_manager = DatabaseManager(neo4j_uri, neo4j_user, neo4j_password)
        
        # Storage
        self.all_entities: Dict[tuple, Entity] = {}
        self.all_relationships: List[Relationship] = []
        self.procedure_contexts: List[ProcedureContext] = []
        
        print("✓ Knowledge Graph Builder initialized")
    
    def build_knowledge_graph(self, file_paths: List[Path]):
        """Main pipeline execution following refined steps 0-6."""
        print("\n" + "="*70)
        print("  3GPP KNOWLEDGE GRAPH CONSTRUCTION PIPELINE")
        print("="*70)
        
        try:
            # Step 1: Load 3GPP documents
            all_sections = self.document_loader.load_documents(file_paths)
            
            # Step 4a-g: Process each document
            for file_path in file_paths:
                print(f"\n--- Processing Document: {file_path.name} ---")
                self._process_single_document(all_sections, file_path.stem)
            
            # Step 4h: Merge all knowledge graphs and load to database
            self._merge_and_load_to_database()
            
            # Final statistics
            self._print_final_statistics()
            
        except Exception as e:
            print(f"Error in pipeline: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Step 6: Close connection
            self.database_manager.close()
    
    def _process_single_document(self, all_sections: List, document_name: str):
        """Step 4a-g: Process a single document."""
        # Filter sections for this document
        document_sections = [s for s in all_sections if s.document == document_name]
        
        # Step 4b: Select sections with figures (procedure identification)
        procedure_sections = self.document_loader.identify_procedure_sections_with_llm(document_sections)
        
        if not procedure_sections:
            print(f"  No procedure sections found in {document_name}")
            return
        
        # Step 4c-f: Process each procedure section
        for section in procedure_sections:
            print(f"\n  Procedure: {section.procedure_name}")
            self._process_procedure_section(section)
    
    def _process_procedure_section(self, section):
        """Step 4c-e: Process a single procedure section."""
        # Create procedure context
        context = ProcedureContext(section.procedure_name, section)
        
        # Add procedure entity (Step 2a)
        self._add_entity(section.procedure_name, "Procedure", {
            "clause": section.clause,
            "document": section.document,
            "has_figure": section.has_figure,
            "extraction_method": "llm_procedure_identification"
        })
        
        # Step 2b-e: Extract entities
        entity_result = self.entity_extractor.extract_entities_for_procedure(context)
        
        if entity_result.success:
            # CRITICAL: Add entities using procedure-specific step names
            self._add_entities_from_context(context)
            
            # Step 3: Extract relationships
            relationships = self.relation_extractor.extract_relationships_for_procedure(context)
            self.all_relationships.extend(relationships)
            
            # Store context
            self.procedure_contexts.append(context)
            
            print(f"    ✓ Extracted: {len(context.network_functions)} NFs, {len(context.messages)} msgs, "
                  f"{len(context.parameters)} params, {len(context.keys)} keys, {len(context.steps)} steps")
            print(f"    ✓ Relationships: {len(relationships)}")
        else:
            print(f"    ✗ Entity extraction failed: {entity_result.error_message}")
    
    def _add_entities_from_context(self, context: ProcedureContext):
        """CRITICAL FIX: Add entities using procedure-specific step names."""
        # Network Functions (Step 2b)
        for nf in context.network_functions:
            self._add_entity(nf, "NetworkFunction", {
                "procedure": context.procedure_name,
                "clause": context.section.clause,
                "extraction_method": "nlp_llm_extraction"
            })
        
        # Messages (Step 2c)
        for msg in context.messages:
            self._add_entity(msg, "Message", {
                "procedure": context.procedure_name,
                "clause": context.section.clause,
                "extraction_method": "nlp_llm_extraction"
            })
        
        # Parameters (Step 2d)
        for param in context.parameters:
            self._add_entity(param, "Parameter", {
                "procedure": context.procedure_name,
                "clause": context.section.clause,
                "extraction_method": "nlp_llm_extraction"
            })
        
        # Keys (Step 2d)
        for key in context.keys:
            self._add_entity(key, "Key", {
                "procedure": context.procedure_name,
                "clause": context.section.clause,
                "extraction_method": "nlp_llm_extraction"
            })
        
        # Steps (Step 2e) - CRITICAL FIX: Use procedure-specific step names from context
        for step_name in context.steps:
            # Extract step number from procedure-specific name
            step_match = re.search(r'_step_(\d+)$', step_name)
            step_number = int(step_match.group(1)) if step_match else 1
            
            self._add_entity(step_name, "Step", {
                "procedure": context.procedure_name,
                "clause": context.section.clause,
                "step_number": step_number,
                "extraction_method": "nlp_llm_extraction"
            })
    
    def _merge_and_load_to_database(self):
        """Step 4h & 5: Merge knowledge graphs and load to database."""
        print(f"\n=== STEP 5: Loading Knowledge Graph to Database ===")
        
        # Clear database
        self.database_manager.clear_database()
        
        # Create entities
        entities_list = list(self.all_entities.values())
        self.database_manager.create_entities(entities_list)
        
        # Create relationships
        self.database_manager.create_relationships(self.all_relationships)
        
        # Create indexes
        entity_types = set(entity.entity_type for entity in entities_list)
        self.database_manager.create_indexes(list(entity_types))
        
        print("✓ Knowledge graph loaded to Neo4j")
        
        # Verify critical relationships
        self._verify_critical_relationships()
    
    def _verify_critical_relationships(self):
        """Verify required relationships exist."""
        print("\n=== Verifying Required Relationships ===")
        
        critical_relations = ['FOLLOWED_BY', 'PART_OF', 'INVOKE', 'INVOLVE', 'CONTAINS', 'SEND_BY', 'SEND_TO']
        
        for rel_type in critical_relations:
            count = self.database_manager.verify_relationships(rel_type)
            if count > 0:
                print(f"✓ {rel_type}: {count}")
                samples = self.database_manager.get_sample_relationships(rel_type, 2)
                for sample in samples:
                    print(f"    {sample['source']} -> {sample['target']}")
            else:
                print(f"⚠️  {rel_type}: 0 relationships")
    
    def _print_final_statistics(self):
        """Print comprehensive statistics."""
        print("\n" + "="*70)
        print("  KNOWLEDGE GRAPH CONSTRUCTION STATISTICS")
        print("="*70)
        
        # Entity statistics
        entity_counts = defaultdict(int)
        for entity in self.all_entities.values():
            entity_counts[entity.entity_type] += 1
        
        print("Entity Counts:")
        for entity_type, count in sorted(entity_counts.items()):
            print(f"  {entity_type:15}: {count:4d}")
        
        # Relationship statistics
        relationship_counts = defaultdict(int)
        for rel in self.all_relationships:
            relationship_counts[rel.rel_type] += 1
        
        print(f"\nRelationship Counts (Total: {len(self.all_relationships)}):")
        for rel_type, count in sorted(relationship_counts.items()):
            print(f"  {rel_type:15}: {count:4d}")
        
        print(f"\nProcedures Processed: {len(self.procedure_contexts)}")
        print(f"Documents Processed: {len(set(ctx.section.document for ctx in self.procedure_contexts))}")
        
        # Procedure-specific step analysis
        print(f"\nProcedure-Specific Steps Analysis:")
        for ctx in self.procedure_contexts[:5]:  # Show first 5
            print(f"  {ctx.procedure_name}: {len(ctx.steps)} steps")
            for step in ctx.steps[:3]:  # Show first 3 steps
                print(f"    - {step}")
        
        print("="*70)
    
    def _add_entity(self, name: str, entity_type: str, properties: Dict):
        """Add entity (avoid duplicates)."""
        key = (name, entity_type)
        if key not in self.all_entities:
            self.all_entities[key] = Entity(name, entity_type, properties)
    
    # Incremental update methods (Requirement 6)
    def add_new_document(self, file_path: Path):
        """Add new document to existing knowledge graph."""
        print(f"Adding new document: {file_path.name}")
        sections = self.document_loader.load_documents([file_path])
        self._process_single_document(sections, file_path.stem)
        self._merge_and_load_to_database()
    
    def update_document(self, file_path: Path):
        """Update existing document in knowledge graph."""
        print(f"Updating document: {file_path.name}")
        # Remove existing entities for this document
        entities_to_remove = [
            key for key, entity in self.all_entities.items()
            if entity.properties.get('document') == file_path.stem
        ]
        for key in entities_to_remove:
            del self.all_entities[key]
        
        # Remove relationships for this document
        self.all_relationships = [
            rel for rel in self.all_relationships
            if not any(ctx.section.document == file_path.stem 
                      for ctx in self.procedure_contexts
                      if ctx.procedure_name in [rel.source_name, rel.target_name])
        ]
        
        # Re-add updated document
        self.add_new_document(file_path)
    
    def remove_document(self, document_name: str):
        """Remove document from knowledge graph."""
        print(f"Removing document: {document_name}")
        # Implementation for document removal
        # Remove entities and relationships associated with the document
        pass
    
    def build_knowledge_graph(self, document_files: List[Path]):
        """Enhanced knowledge graph building with long context embeddings."""
        print(f"\n=== BUILDING KNOWLEDGE GRAPH WITH LONG CONTEXT ===")
        
        # Step 1: Load documents
        all_sections = self.document_loader.load_documents(document_files)
        
        # Step 2: Process each document
        for file_path in document_files:
            print(f"\n--- Processing Document: {file_path.name} ---")
            self._process_single_document(all_sections, file_path.stem)
        
        # Step 5: Enhanced embedding generation with full context
        print(f"\n=== STEP 5: GENERATING LONG CONTEXT EMBEDDINGS ===")
        
        entity_extractor = EntityExtractor()
        
        for context in self.procedure_contexts:
            procedure_name = context.procedure_name
            print(f"\n📄 Generating long context embeddings for: {procedure_name}")
            
            # Create comprehensive procedure text
            full_procedure_text = f"""
            PROCEDURE: {procedure_name}
            
            SECTION TITLE: {context.section.title}
            CLAUSE: {context.section.clause}
            
            FULL CONTENT:
            {context.section.text}
            
            EXTRACTED ENTITIES:
            Network Functions: {', '.join(context.network_functions)}
            Messages: {', '.join(context.messages)}
            Parameters: {', '.join(context.parameters)}
            Keys: {', '.join(context.keys)}
            
            PROCEDURE STEPS:
            {chr(10).join([f"{i+1}. {step}" for i, step in enumerate(context.steps)])}
            
            RELATIONSHIPS:
            {chr(10).join([f"- {rel.source_name} --{rel.rel_type}--> {rel.target_name}" 
                          for rel in context.relationships[:10]])}  # First 10 relationships
            """
            
            # Generate long context embedding
            embedding_result = entity_extractor.generate_long_context_embeddings(
                full_procedure_text, procedure_name
            )
            
            if embedding_result["embedding"]:
                # Store embedding with procedure entity
                procedure_entity = self._find_entity_by_name(procedure_name, "Procedure")
                if procedure_entity:
                    procedure_entity.properties["long_context_embedding"] = embedding_result["embedding"]
                    procedure_entity.properties["embedding_metadata"] = embedding_result["metadata"]
                    print(f"  ✓ Stored long context embedding: {embedding_result['metadata']['total_tokens']} tokens")
        
        print(f"\n✅ Long context embeddings generated for {len(self.procedure_contexts)} procedures")
    
    def _find_entity_by_name(self, name: str, entity_type: str):
        """Find entity by name and type."""
        key = (name, entity_type)
        return self.all_entities.get(key)