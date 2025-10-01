from pathlib import Path
from typing import List, Dict, Optional
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
        
        print("✓ Knowledge Graph Builder initialized.")
    
    def build_knowledge_graph(self, file_paths: List[Path]):
        """Main pipeline execution with memory management."""
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
            
            # Step 5: Generate embeddings in batches to avoid OOM
            self._generate_embeddings_in_batches()
            
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
            step_match = re.search(r'_step_([\w-]+)$', step_name)
            step_number_str = step_match.group(1) if step_match else "1"
            
            # Get the description for this step from the context
            step_description = context.step_descriptions.get(step_name, "")
            
            self._add_entity(
                name=step_name,
                entity_type="Step",
                properties={
                    "procedure": context.procedure_name,
                    "clause": context.section.clause,
                    "step_number": step_number_str,
                    "extraction_method": "nlp_llm_extraction"
                },
                description=step_description  # Pass the description here
            )
            print(f"      📝 Added Step entity: {step_name}")
    
    def _merge_and_load_to_database(self):
        """Step 4h & 5: Merge knowledge graphs and load to database."""
        print(f"\n=== STEP 5: Loading Knowledge Graph to Database ===")
        
        # DEBUG: Check what entities we have
        entity_counts = self._debug_entity_collection()
        
        # If no steps, something is wrong with entity building
        if entity_counts.get("Step", 0) == 0:
            print("❌ CRITICAL: No step entities found! Check entity building logic.")
            return
        
        # Clear database
        self.database_manager.clear_database()
        
        # Create entities with better error handling
        entities_list = list(self.all_entities.values())
        print(f"Creating {len(entities_list)} entities...")
        
        created_entities = 0
        step_entities_created = 0
        
        for entity in entities_list:
            try:
                result = self.database_manager.create_entity(
                    entity.name, 
                    entity.entity_type, 
                    entity.properties
                )
                
                if result:
                    created_entities += 1
                    if entity.entity_type == "Step":
                        step_entities_created += 1
                        # Only print every 10th step to reduce noise
                        if step_entities_created % 10 == 0:
                            print(f"SUCCESS: Created {step_entities_created} step entities so far...")
                else:
                    print(f"WARNING: Failed to create entity {entity.name}")
                    
            except Exception as e:
                print(f"ERROR: Failed to create entity {entity.name}: {e}")
    
        print(f"✓ Created {created_entities}/{len(entities_list)} entities (including {step_entities_created} steps)")
        
        # ADD THE RELATIONSHIP DEBUGGING HERE - BEFORE RELATIONSHIP CREATION
        rel_counts = self._debug_relationship_collection()
        
        if len(self.all_relationships) == 0:
            print("❌ CRITICAL: No relationships to create! Check relationship extraction.")
            return
        
        # Create relationships
        print(f"Creating {len(self.all_relationships)} relationships...")
        
        created_relationships = 0
        failed_relationships = []
        relationship_counts = {}
        
        for i, relationship in enumerate(self.all_relationships):
            try:
                # Track relationship types
                rel_type = relationship.rel_type
                relationship_counts[rel_type] = relationship_counts.get(rel_type, 0) + 1
                
                # Print progress every 1000 relationships
                if (i + 1) % 1000 == 0:
                    print(f"    Processing relationship {i + 1}/{len(self.all_relationships)}")
                
                # Verify both entities exist first
                source_exists = self._verify_entity_exists(relationship.source_name)
                target_exists = self._verify_entity_exists(relationship.target_name)
                
                if not source_exists:
                    print(f"ERROR: Source entity '{relationship.source_name}' does not exist")
                    failed_relationships.append(f"{relationship.source_name} (source missing)")
                    continue
                    
                if not target_exists:
                    print(f"ERROR: Target entity '{relationship.target_name}' does not exist")
                    failed_relationships.append(f"{relationship.target_name} (target missing)")
                    continue
                
                # Create the relationship
                result = self.database_manager.create_relationship(
                    relationship.source_name,
                    relationship.target_name,
                    relationship.rel_type,
                    relationship.properties or {}
                )
                
                if result:
                    created_relationships += 1
                    # Print success for critical relationship types
                    if relationship.rel_type in ["PART_OF", "FOLLOWED_BY", "INVOKE", "INVOLVE"]:
                        # Only print first few of each type to reduce noise
                        if relationship_counts[rel_type] <= 3:
                            print(f"SUCCESS: Created {relationship.rel_type} relationship: {relationship.source_name} -> {relationship.target_name}")
                else:
                    failed_relationships.append(f"{relationship.source_name} -[{relationship.rel_type}]-> {relationship.target_name}")
                    
            except Exception as e:
                print(f"ERROR: Failed to create relationship: {e}")
                failed_relationships.append(f"{relationship.source_name} -[{relationship.rel_type}]-> {relationship.target_name} (exception)")
        
        print(f"✓ Created {created_relationships}/{len(self.all_relationships)} relationships")
        
        # Print relationship type summary
        print(f"\nRelationship Creation Summary:")
        for rel_type, count in sorted(relationship_counts.items()):
            print(f"  {rel_type}: {count} attempted")
        
        if failed_relationships:
            print(f"❌ Failed relationships: {len(failed_relationships)}")
            # Show first 5 failures
            for failed in failed_relationships[:5]:
                print(f"   {failed}")
            if len(failed_relationships) > 5:
                print(f"   ... and {len(failed_relationships) - 5} more")
        
        # Create indexes
        entity_types = set(entity.entity_type for entity in entities_list)
        self.database_manager.create_indexes(list(entity_types))
        
        print("✓ Knowledge graph loaded to Neo4j")
        
        # Verify critical relationships
        self._verify_critical_relationships()

    def _generate_embeddings_in_batches(self):
        """Generate embeddings for entities in batches to avoid OOM."""
        print("\n=== STEP 6: Generating Embeddings ===")
        
        try:
            # For now, skip embeddings to focus on core functionality
            print("  Embeddings generation skipped for now...")
            print("  ✓ Can be implemented later with sentence transformers")
            
            # Future implementation:
            # from sentence_transformers import SentenceTransformer
            # model = SentenceTransformer('all-MiniLM-L6-v2')
            # 
            # # Get all entity texts for embedding
            # entity_texts = []
            # for entity in self.all_entities.values():
            #     text = f"{entity.name} {entity.entity_type}"
            #     entity_texts.append(text)
            # 
            # # Generate embeddings in batches
            # batch_size = 32
            # for i in range(0, len(entity_texts), batch_size):
            #     batch = entity_texts[i:i+batch_size]
            #     embeddings = model.encode(batch)
            #     # Store embeddings in database
            
        except Exception as e:
            print(f"  Error in embedding generation: {e}")
            print("  Continuing without embeddings...")

    def _debug_entity_collection(self):
        """Debug what entities are being collected."""
        print("\n=== DEBUGGING ENTITY COLLECTION ===")
        
        entity_counts = {}
        step_entities = []
        
        for entity_name, entity in self.all_entities.items():
            entity_type = entity.entity_type
            entity_counts[entity_type] = entity_counts.get(entity_type, 0) + 1
            
            if entity_type == "Step":
                step_entities.append(entity_name)
        
        print("Entity counts by type:")
        for entity_type, count in entity_counts.items():
            print(f"  {entity_type}: {count}")
        
        print(f"\nStep entities ({len(step_entities)}):")
        for step in step_entities[:10]:  # Show first 10
            print(f"  - {step}")
        if len(step_entities) > 10:
            print(f"  ... and {len(step_entities) - 10} more")
        
        return entity_counts

    def _debug_relationship_collection(self):
        """Debug what relationships are being collected."""
        print("\n=== DEBUGGING RELATIONSHIP COLLECTION ===")
        
        if len(self.all_relationships) == 0:
            print("❌ NO RELATIONSHIPS COLLECTED!")
            return {}
        
        relationship_counts = {}
        critical_relationships = []
        contains_relationships = []
        
        for relationship in self.all_relationships:
            rel_type = relationship.rel_type
            relationship_counts[rel_type] = relationship_counts.get(rel_type, 0) + 1
            
            if rel_type in ["PART_OF", "FOLLOWED_BY", "INVOKE", "INVOLVE", "CONTAINS", "SEND", "SEND_BY", "SEND_TO"]:
                critical_relationships.append(f"{relationship.source_name} -[{rel_type}]-> {relationship.target_name}")
            
            if rel_type == "CONTAINS":
                contains_relationships.append(f"{relationship.source_name} -[CONTAINS]-> {relationship.target_name}")
        
        print("Relationship counts by type:")
        for rel_type, count in sorted(relationship_counts.items()):
            print(f"  {rel_type}: {count}")
        
        print(f"\nCritical relationships ({len(critical_relationships)}):")
        for rel in critical_relationships[:15]:  # Show first 15
            print(f"  - {rel}")
        if len(critical_relationships) > 15:
            print(f"  ... and {len(critical_relationships) - 15} more")
        
        # Specifically debug CONTAINS relationships
        if contains_relationships:
            print(f"\n✅ CONTAINS relationships ({len(contains_relationships)}):")
            for rel in contains_relationships[:10]:  # Show first 10
                print(f"  - {rel}")
            if len(contains_relationships) > 10:
                print(f"  ... and {len(contains_relationships) - 10} more")
        else:
            print(f"\n❌ NO CONTAINS RELATIONSHIPS FOUND!")
            
            # Debug why CONTAINS relationships are missing
            print("Debug info for CONTAINS relationships:")
            print(f"  - Total procedures: {len(set(r.properties.get('procedure', 'unknown') for r in self.all_relationships))}")
            
            # Check if we have parameters and steps
            parameters_found = set()
            steps_found = set()
            for r in self.all_relationships:
                if "step_" in r.source_name:
                    steps_found.add(r.source_name)
                if r.target_name in ['SUPI', 'GUTI', 'TMSI', 'IMEI', 'TAI', 'PLMN ID', 'S-NSSAI', 'DNN']:
                    parameters_found.add(r.target_name)
            
            print(f"  - Steps found: {len(steps_found)} (examples: {list(steps_found)[:3]})")
            print(f"  - Parameters found: {len(parameters_found)} (examples: {list(parameters_found)})")
        
        # Check for missing required relationship types
        required_types = ["PART_OF", "FOLLOWED_BY", "INVOKE", "INVOLVE", "CONTAINS", "SEND", "SEND_BY", "SEND_TO"]
        missing_types = [rt for rt in required_types if rt not in relationship_counts]
        
        if missing_types:
            print(f"\n❌ MISSING REQUIRED RELATIONSHIP TYPES: {missing_types}")
        else:
            print(f"\n✅ All required relationship types present!")
        
        return relationship_counts
    
    def _verify_entity_exists(self, entity_name: str) -> bool:
        """Verify if an entity exists in the database."""
        try:
            query = "MATCH (n {name: $name}) RETURN count(n) as count"
            result = self.database_manager.session.run(query, name=entity_name)
            count = result.single()["count"]
            return count > 0
        except Exception as e:
            print(f"ERROR: Could not verify entity {entity_name}: {e}")
            return False
    
    def _verify_critical_relationships(self):
        """Verify that critical relationships were actually created."""
        print("\n=== VERIFYING RELATIONSHIP CREATION ===")
        
        required_types = ["FOLLOWED_BY", "PART_OF", "INVOKE", "INVOLVE", "CONTAINS", "SEND_BY", "SEND_TO", "SEND"]
        
        for rel_type in required_types:
            try:
                query = f"MATCH ()-[r:{rel_type}]->() RETURN count(r) as count"
                result = self.database_manager.session.run(query)
                count = result.single()["count"]
                
                if count > 0:
                    print(f"✓ {rel_type}: {count} relationships")
                    
                    # Show example relationships
                    example_query = f"""
                    MATCH (a)-[r:{rel_type}]->(b) 
                    RETURN a.name as source, b.name as target, type(r) as rel_type
                    LIMIT 3
                    """
                    examples = self.database_manager.session.run(example_query)
                    for record in examples:
                        print(f"    Example: {record['source']} -[{record['rel_type']}]-> {record['target']}")
                else:
                    print(f"⚠️  {rel_type}: {count} relationships - MISSING!")
                    
            except Exception as e:
                print(f"❌ {rel_type}: Error checking - {e}")
        
        # Overall statistics
        try:
            total_query = "MATCH ()-[r]->() RETURN count(r) as total"
            total_result = self.database_manager.session.run(total_query)
            total_count = total_result.single()["total"]
            print(f"\n📊 Total relationships in database: {total_count}")
            
            entity_query = "MATCH (n) RETURN count(n) as total"
            entity_result = self.database_manager.session.run(entity_query)
            entity_count = entity_result.single()["total"]
            print(f"📊 Total entities in database: {entity_count}")
            
        except Exception as e:
            print(f"❌ Error getting statistics: {e}")
    
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
        
        # # Procedure-specific step analysis
        # print(f"\nProcedure-Specific Steps Analysis:")
        # for ctx in self.procedure_contexts[:5]:  # Show first 5
        #     print(f"  {ctx.procedure_name}: {len(ctx.steps)} steps")
        #     for step in ctx.steps[:3]:  # Show first 3 steps
        #         print(f"    - {step}")
        
        print("="*70)
    
    def _add_entity(self, name: str, entity_type: str, properties: Dict, description: Optional[str] = None):
        """Add entity (avoid duplicates)."""
        key = (name, entity_type)
        if key not in self.all_entities:
            self.all_entities[key] = Entity(name=name, entity_type=entity_type, properties=properties, description=description)
    
    def _find_entity_by_name(self, name: str, entity_type: str):
        """Find entity by name and type."""
        key = (name, entity_type)
        return self.all_entities.get(key)
    
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
    
    def create_step_entity(self, step_name: str, procedure_name: str, step_descriptions: Dict[str, str] = None):
        """Create step entity with description field."""
        
        # Get description from context if available
        description = ""
        if step_descriptions and step_name in step_descriptions:
            description = step_descriptions[step_name]
            # Normalize paragraph breaks (as requested - single line with spaces)
            description = description.replace('  ', ' ').replace('\n', ' ').strip()
        
        step_entity = Step(
            name=step_name,
            procedure_name=procedure_name,
            description=description,  # Store multi-paragraph content here
            step_order=self._extract_step_order(step_name)
        )
        
        self.session.add(step_entity)
        print(f"SUCCESS: Created Step entity '{step_name}' with description ({len(description)} chars)")
        
        return step_entity

    def _extract_step_order(self, step_name: str) -> int:
        """Extract step order from step name for proper sequencing."""
        # Extract number from step name like "43221_Procedure_step_7a" → 7
        match = re.search(r'step_(\d+)', step_name)
        if match:
            return int(match.group(1))
        
        # Fallback to position-based ordering
        return 1
    
    """Build complete knowledge graph with proper isolation."""
    total_stats = {
        'procedures': 0,
        'entities': {},
        'relationships': 0,
        'errors': []
    }
