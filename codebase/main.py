import os
import warnings
import torch
import re
from natsort import natsorted
from pathlib import Path

# Suppress warnings early
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, DOCS_PATH, DOC_PATTERN
from knowledge_graph_builder import KnowledgeGraphBuilder
from search_engine import ProcedureSearchEngine
from fsm_converter import FSMConverter
from data_structures import SearchQuery

def main():
    """Main execution function with search and FSM conversion demo."""
    print("="*70)
    print("  3GPP KNOWLEDGE GRAPH BUILDER - COMPLETE SOLUTION")
    print("="*70)
    
    # Step 0: Configuration and initialization
    print("Step 0: Configuration and initialization...")
    
    # GPU check (Requirement 1)
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        current_gpu = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(current_gpu)
        gpu_memory = torch.cuda.get_device_properties(current_gpu).total_memory / 1e9
        print(f"✓ GPU detected: {gpu_name}")
        print(f"✓ VRAM available: {gpu_memory:.0f} GB")
    else:
        print("⚠️  No GPU detected - using CPU (will be slower)")
    
    # Find documents
    doc_files = list(DOCS_PATH.glob(DOC_PATTERN))
    if not doc_files:
        print(f"❌ No documents found in {DOCS_PATH} with pattern {DOC_PATTERN}")
        return
    
    print(f"\nFound {len(doc_files)} documents to process:")
    for doc_file in doc_files:
        print(f"  - {doc_file.name}")
    
    builder = None
    try:
        # Initialize Knowledge Graph Builder
        print(f"\nInitializing Knowledge Graph Builder...")
        builder = KnowledgeGraphBuilder(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
        
        # Build knowledge graph incrementally
        print(f"\n=== BUILDING KNOWLEDGE GRAPH ===")
        builder.build_knowledge_graph(doc_files)
        
        # Demo: Search and FSM Conversion
        print(f"\n=== SEARCH AND FSM CONVERSION DEMO ===")
        # Re-initialize builder to get a fresh DB connection for the demo
        builder = KnowledgeGraphBuilder(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
        demo_search_and_fsm(builder)
        
    except KeyboardInterrupt:
        print("\n⚠️  Process interrupted by user")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if builder:
            builder.database_manager.close()
    
    print("\n" + "="*70)
    print("  PIPELINE EXECUTION COMPLETED")
    print("="*70)

def demo_search_and_fsm(builder: KnowledgeGraphBuilder):
    """Demo search functionality and FSM conversion using data from the database."""
    
    # Initialize search engine and FSM converter
    print("Initializing search engine and FSM converter...")
    search_engine = ProcedureSearchEngine(builder.entity_extractor.embedding_model)
    fsm_converter = FSMConverter()
    
    # Get all entities and relationships from the database
    print("Loading all entities and relationships from database...")
    all_entities = builder.database_manager.get_all_entities()
    all_relationships = builder.database_manager.get_all_relationships()
    print(f"Found {len(all_entities)} total entities and {len(all_relationships)} relationships in the database.")
    
    # Index all entities for search
    search_engine.build_search_index(all_entities)
    
    # Demo queries
    demo_queries = [
        "authentication process between UE and 5G core network",
        "registration procedure for UE attachment",
        "5G AKA authentication protocol",
        "PDU session establishment",
        "service request procedure"
    ]
    
    print(f"\n--- NATURAL LANGUAGE SEARCH DEMO ---")
    
    for query_text in demo_queries:
        print(f"\n🔍 Query: '{query_text}'")
        
        try:
            # Search for procedures using simplified interface
            results = search_engine.search_procedures(query_text, max_results=3)
            
            if results:
                print(f"   Found {len(results)} procedures:")
                
                for i, result in enumerate(results, 1):
                    procedure_name = result['name']
                    score = result['score']
                    match_type = result['match_type']
                    
                    print(f"   {i}. {procedure_name} (score: {score:.3f}, type: {match_type})")
                    
                    # Convert first result to FSM
                    if i == 1:
                        print(f"   🔄 Converting '{procedure_name}' to FSM...")
                        try:
                            # Get the full Step entities for the procedure by filtering all_entities
                            procedure_steps = [e for e in all_entities if e.entity_type == 'Step' and e.properties.get('procedure') == procedure_name]
                            step_entities = natsorted(procedure_steps, key=lambda s: s.properties.get('step_number', '0'))

                            # Get all relationships involving the procedure and its steps
                            all_relevant_entity_names = {s.name for s in step_entities}
                            all_relevant_entity_names.add(procedure_name)
                            
                            relationships = [
                                {'source_name': r.source_name, 'target_name': r.target_name, 'rel_type': r.rel_type}
                                for r in all_relationships
                                if r.source_name in all_relevant_entity_names or r.target_name in all_relevant_entity_names
                            ]

                            # Convert to FSM
                            fsm = fsm_converter.convert_procedure_to_fsm(
                                procedure_name, step_entities, relationships
                            )
                            
                            if fsm:
                                is_valid, errors = fsm_converter.validate_fsm(fsm)
                                print(f"   ✓ Validation: {'PASSED' if is_valid else 'FAILED'}")
                                if errors:
                                    for error in errors[:3]:
                                        print(f"     - {error}")
                                
                                safe_name = fsm_converter._sanitize_filename(procedure_name)
                                json_file = f"{safe_name}_fsm.json"
                                dot_file = f"{safe_name}_fsm.dot"
                                
                                if fsm_converter.export_fsm_to_json(fsm, json_file):
                                    print(f"   📁 FSM exported to: output/{json_file}")
                                
                                if fsm_converter.export_fsm_to_dot(fsm, dot_file):
                                    print(f"   📁 DOT file for visualization: output/{dot_file}")
                                
                                show_fsm_details(fsm)
                                
                            else:
                                print("   ❌ FSM conversion failed: No FSM generated")

                        except Exception as e:
                            print(f"   ❌ FSM conversion failed: {e}")
                            import traceback
                            traceback.print_exc()
            else:
                print(f"   No procedures found for query.")
        except Exception as e:
            print(f"   ❌ Search failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Demo: Direct procedure search
    print(f"\n--- DIRECT PROCEDURE SEARCH ---")
    try:
        authentication_procedures = search_engine.search_authentication_procedures()
        
        if authentication_procedures:
            print(f"Found {len(authentication_procedures)} authentication procedures:")
            for i, result in enumerate(authentication_procedures[:5], 1):  # Show top 5
                print(f"  {i}. {result.entity.name} (score: {result.similarity_score:.3f})")
        else:
            print("No authentication procedures found.")
    except Exception as e:
        print(f"Authentication search failed: {e}")

def show_fsm_details(fsm):
    """Show detailed FSM information."""
    print(f"   📋 FSM Details for '{fsm.procedure_name}':")  # Use procedure_name from data_structures
    print(f"      Initial State: {fsm.initial_state}")
    print(f"      Final States: {', '.join(fsm.final_states) if fsm.final_states else 'None'}")
    
    print(f"      States:")
    for state in fsm.states[:3]:  # Show first 3 states
        desc = state.step_entity.description[:60] + "..." if state.step_entity.description and len(state.step_entity.description) > 60 else (state.step_entity.description or "")
        print(f"        - {state.name}: {desc}")
    
    print(f"      Transitions:")
    for transition in fsm.transitions[:3]:  # Show first 3 transitions
        print(f"        - {transition.source_state} → {transition.target_state} (trigger: {transition.trigger})")

if __name__ == "__main__":
    main()