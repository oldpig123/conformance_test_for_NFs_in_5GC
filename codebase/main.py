import os
import warnings
import torch
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
    
    try:
        # Initialize Knowledge Graph Builder
        print(f"\nInitializing Knowledge Graph Builder...")
        builder = KnowledgeGraphBuilder(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
        
        # Build knowledge graph
        print(f"\n=== BUILDING KNOWLEDGE GRAPH ===")
        builder.build_knowledge_graph(doc_files)
        
        # Demo: Search and FSM Conversion
        print(f"\n=== SEARCH AND FSM CONVERSION DEMO ===")
        demo_search_and_fsm(builder)
        
    except KeyboardInterrupt:
        print("\n⚠️  Process interrupted by user")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*70)
    print("  PIPELINE EXECUTION COMPLETED")
    print("="*70)

def demo_search_and_fsm(builder: KnowledgeGraphBuilder):
    """Demo search functionality and FSM conversion."""
    
    # Initialize search engine
    print("Initializing search engine...")
    search_engine = ProcedureSearchEngine(builder.entity_extractor.embedding_model)
    
    # Index all entities for search
    all_entities = list(builder.all_entities.values())
    search_engine.index_entities(all_entities)
    
    # Initialize FSM converter
    fsm_converter = FSMConverter()
    
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
        
        # Create search query
        query = SearchQuery(
            query_text=query_text,
            entity_types=["Procedure"],
            max_results=3,
            similarity_threshold=0.3
        )
        
        # Search for procedures
        results = search_engine.search(query)
        
        if results:
            print(f"   Found {len(results)} procedures:")
            
            for i, result in enumerate(results, 1):
                procedure_name = result.entity.name
                score = result.similarity_score
                match_type = result.match_type
                
                print(f"   {i}. {procedure_name} (score: {score:.3f}, type: {match_type})")
                
                # Convert first result to FSM
                if i == 1:
                    print(f"   🔄 Converting '{procedure_name}' to FSM...")
                    
                    try:
                        # Get procedure context
                        procedure_context = None
                        for ctx in builder.procedure_contexts:
                            if ctx.procedure_name == procedure_name:
                                procedure_context = ctx
                                break
                        
                        # Convert to FSM
                        fsm = fsm_converter.convert_procedure_to_fsm(
                            procedure_name=procedure_name,
                            entities=all_entities,
                            relationships=builder.all_relationships,
                            context=procedure_context
                        )
                        
                        # Validate FSM
                        validation = fsm_converter.validate_fsm(fsm)
                        
                        print(f"   ✓ FSM created: {len(fsm.states)} states, {len(fsm.transitions)} transitions")
                        print(f"   ✓ Validation: {'PASSED' if validation['overall_valid'] else 'FAILED'}")
                        
                        # Export FSM
                        output_dir = Path("output")
                        output_dir.mkdir(exist_ok=True)
                        
                        # Export to JSON
                        json_file = output_dir / f"{procedure_name.replace(' ', '_')}_fsm.json"
                        fsm_converter.export_fsm_to_json(fsm, json_file)
                        
                        # Export to DOT for visualization
                        dot_file = output_dir / f"{procedure_name.replace(' ', '_')}_fsm.dot"
                        fsm_converter.export_fsm_to_dot(fsm, dot_file)
                        
                        print(f"   📁 FSM exported to: {json_file}")
                        print(f"   📁 DOT file for visualization: {dot_file}")
                        
                        # Show FSM details
                        show_fsm_details(fsm)
                        
                    except Exception as e:
                        print(f"   ❌ FSM conversion failed: {e}")
                        import traceback
                        traceback.print_exc()
        else:
            print(f"   No procedures found for query.")
    
    # Demo: Direct procedure search
    print(f"\n--- DIRECT PROCEDURE SEARCH ---")
    authentication_procedures = search_engine.search_authentication_procedures()
    
    if authentication_procedures:
        print(f"Found {len(authentication_procedures)} authentication procedures:")
        for result in authentication_procedures[:5]:  # Show top 5
            print(f"  - {result.entity.name} (score: {result.similarity_score:.3f})")

def show_fsm_details(fsm):
    """Show detailed FSM information."""
    print(f"   📋 FSM Details for '{fsm.procedure_name}':")
    print(f"      Initial State: {fsm.initial_state}")
    print(f"      Final States: {', '.join(fsm.final_states)}")
    
    print(f"      States:")
    for state in fsm.states[:3]:  # Show first 3 states
        desc = state.description[:60] + "..." if len(state.description) > 60 else state.description
        print(f"        - {state.name}: {desc}")
    
    print(f"      Transitions:")
    for transition in fsm.transitions[:3]:  # Show first 3 transitions
        print(f"        - {transition.source_state} → {transition.target_state} (trigger: {transition.trigger})")

def search_specific_procedure(builder: KnowledgeGraphBuilder, query: str) -> str:
    """Search for a specific procedure and return its name."""
    print(f"🔍 Searching for: '{query}'")
    
    # Initialize search engine
    search_engine = ProcedureSearchEngine(builder.entity_extractor.embedding_model)
    all_entities = list(builder.all_entities.values())
    search_engine.index_entities(all_entities)
    
    # Create search query
    search_query = SearchQuery(
        query_text=query,
        entity_types=["Procedure"],
        max_results=1,
        similarity_threshold=0.3
    )
    
    # Search
    results = search_engine.search(search_query)
    
    if results:
        procedure_name = results[0].entity.name
        print(f"✓ Found procedure: {procedure_name}")
        return procedure_name
    else:
        print(f"❌ No procedure found for query: {query}")
        return None

def convert_procedure_to_fsm(builder: KnowledgeGraphBuilder, procedure_name: str):
    """Convert a specific procedure to FSM."""
    print(f"🔄 Converting '{procedure_name}' to FSM...")
    
    # Initialize FSM converter
    fsm_converter = FSMConverter()
    
    # Get procedure context
    procedure_context = None
    for ctx in builder.procedure_contexts:
        if ctx.procedure_name == procedure_name:
            procedure_context = ctx
            break
    
    if not procedure_context:
        print(f"❌ Procedure context not found for: {procedure_name}")
        return None
    
    # Convert to FSM
    try:
        fsm = fsm_converter.convert_procedure_to_fsm(
            procedure_name=procedure_name,
            entities=list(builder.all_entities.values()),
            relationships=builder.all_relationships,
            context=procedure_context
        )
        
        # Validate FSM
        validation = fsm_converter.validate_fsm(fsm)
        print(f"✓ FSM Validation: {'PASSED' if validation['overall_valid'] else 'FAILED'}")
        
        # Export FSM
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        
        json_file = output_dir / f"{procedure_name.replace(' ', '_')}_fsm.json"
        fsm_converter.export_fsm_to_json(fsm, json_file)
        
        print(f"✓ FSM exported to: {json_file}")
        
        return fsm
        
    except Exception as e:
        print(f"❌ FSM conversion failed: {e}")
        return None

if __name__ == "__main__":
    main()