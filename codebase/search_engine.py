from typing import List, Dict, Any, Optional
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings("ignore")

# FIXED: Add missing imports
from data_structures import Entity, SearchQuery, SearchResult, ProcedureContext
from config import SEARCH_CONFIG

class ProcedureSearchEngine:
    """Search engine for finding procedures by natural language queries (Requirement 8)."""
    
    def __init__(self, embedding_model=None):
        self.embedding_model = embedding_model
        self.entities_index: Dict[str, Entity] = {}
        self.entity_names = []

        # NEW: Weights for combining multiple semantic scores
        self.W_SEMANTIC_TITLE = 2.0
        self.W_SEMANTIC_PARENT = 2.0
        self.W_SEMANTIC_DESC = 1.0
        
        print("ProcedureSearchEngine initialized with Full Semantic Search model")
    
    def search(self, query: SearchQuery) -> List[SearchResult]:
        """Search for procedures using a multi-field semantic scoring model."""
        
        # 1. Get semantic scores from title, parent_title, and description fields
        semantic_scores = self._semantic_search(query)
        
        # 2. Combine, rank, and produce final results
        results = self._deduplicate_and_rank(semantic_scores)
        
        # Filter by entity types if specified
        if query.entity_types:
            results = [r for r in results if r.entity.entity_type in query.entity_types]
        
        # Apply similarity threshold and limit
        results = [r for r in results if r.similarity_score >= query.similarity_threshold]
        results = results[:query.max_results]
        
        print(f"Search query: '{query.query_text}' -> {len(results)} results")
        
        return results
    
    def build_search_index(self, entities: List[Entity]):
        """Builds the search index by loading entities with their embeddings."""
        print(f"Indexing {len(entities)} entities for semantic search...")
        if not entities:
            print("Warning: No entities provided for indexing")
            return

        self.entities_index = {entity.name: entity for entity in entities}
        self.entity_names = [entity.name for entity in entities]
        print(f"✓ Semantic search index built with {len(entities)} documents")
            
    def _semantic_search(self, query: SearchQuery) -> Dict[str, Dict[str, float]]:
        """Performs semantic search across multiple fields and returns a dict of scores."""
        scores = {name: {'title': 0.0, 'parent': 0.0, 'desc': 0.0} for name in self.entity_names}
        if not self.embedding_model:
            return scores

        try:
            query_embedding = self.embedding_model.encode(query.query_text, convert_to_numpy=True)
            
            for entity_name, entity in self.entities_index.items():
                # Title embedding
                if entity.title_embedding:
                    scores[entity_name]['title'] = self._calculate_cosine_similarity(query_embedding, entity.title_embedding)
                
                # Parent title embedding
                if entity.parent_title_embedding:
                    scores[entity_name]['parent'] = self._calculate_cosine_similarity(query_embedding, entity.parent_title_embedding)

                # Description embedding
                if entity.embedding:
                    scores[entity_name]['desc'] = self._calculate_cosine_similarity(query_embedding, entity.embedding)

        except Exception as e:
            print(f"Semantic search error: {e}")
        
        return scores

    def _deduplicate_and_rank(self, semantic_scores: Dict[str, Dict[str, float]]) -> List[SearchResult]:
        """Combines semantic scores from different fields and ranks the results."""
        final_scores = {}
        for entity_name in self.entity_names:
            scores = semantic_scores.get(entity_name, {})
            title_score = scores.get('title', 0.0)
            parent_score = scores.get('parent', 0.0)
            desc_score = scores.get('desc', 0.0)

            # Combine scores with new weights
            final_score = (title_score * self.W_SEMANTIC_TITLE) + \
                          (parent_score * self.W_SEMANTIC_PARENT) + \
                          (desc_score * self.W_SEMANTIC_DESC)

            if final_score > 0.1: # Apply a threshold
                final_scores[entity_name] = final_score

        # Create SearchResult objects
        results = []
        for name, score in final_scores.items():
            results.append(SearchResult(
                entity=self.entities_index[name],
                similarity_score=score,
                match_type="semantic_hybrid"
            ))
        
        # Sort by final score
        results.sort(key=lambda x: x.similarity_score, reverse=True)
        return results

        # Create SearchResult objects
        results = []
        for name, score in final_scores.items():
            results.append(SearchResult(
                entity=self.entities_index[name],
                similarity_score=score,
                match_type="hybrid"
            ))
        
        # Sort by final score
        results.sort(key=lambda x: x.similarity_score, reverse=True)
        return results
    
    def search_procedures_by_description(self, description: str) -> List[SearchResult]:
        """Specialized search for procedures by description."""
        # Create search query for procedures only
        query = SearchQuery(
            query_text=description,
            entity_types=["Procedure"],
            max_results=10,
            similarity_threshold=0.3
        )
        
        return self.search(query)
    
    def search_authentication_procedures(self) -> List[SearchResult]:
        """Find all authentication-related procedures."""
        query = SearchQuery(
            query_text="authentication process between UE and 5G core network",
            entity_types=["Procedure"],
            max_results=20,
            similarity_threshold=0.2
        )
        
        return self.search(query)
    
    def get_procedure_suggestions(self, partial_query: str) -> List[str]:
        """Get procedure name suggestions for auto-complete."""
        suggestions = []
        partial_lower = partial_query.lower()
        
        for entity in self.entities_index.values():
            if entity.entity_type == "Procedure":
                # Check if procedure name starts with or contains query
                if (entity.name.lower().startswith(partial_lower) or 
                    partial_lower in entity.name.lower()):
                    suggestions.append(entity.name)
                
                # Check search keywords if they exist
                if hasattr(entity, 'search_keywords') and entity.search_keywords:
                    for keyword in entity.search_keywords:
                        if keyword.lower().startswith(partial_lower):
                            suggestions.append(entity.name)
                            break
        
        # Remove duplicates and limit
        suggestions = list(set(suggestions))[:10]
        suggestions.sort()
        
        return suggestions
    
    def _calculate_cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        if vec1 is None or vec2 is None:
            return 0.0
        
        # Ensure numpy arrays
        vec1 = np.asarray(vec1)
        vec2 = np.asarray(vec2)
        
        # Calculate cosine similarity
        similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        
        return float(similarity) if not np.isnan(similarity) else 0.0
    
    def get_all_entities(self) -> List[Entity]:
        """Get all indexed entities."""
        return list(self.entities_index.values())
    
    def search_procedures(self, query_text: str, max_results: int = 10) -> List[Dict]:
        """Simplified procedure search that returns dict results."""
        query = SearchQuery(
            query_text=query_text,
            entity_types=["Procedure"],
            max_results=max_results,
            similarity_threshold=0.2
        )
        
        results = self.search(query)
        
        # Convert to simple dict format for demo
        simple_results = []
        for result in results:
            simple_results.append({
                'name': result.entity.name,
                'score': result.similarity_score,
                'match_type': result.match_type,
                'description': getattr(result.entity, 'description', "No description available")
            })
        
        return simple_results
    
    def get_procedure_details(self, procedure_name: str) -> Optional[Dict]:
        """Get detailed information about a specific procedure."""
        entity = self.entities_index.get(procedure_name)
        
        if not entity or entity.entity_type != "Procedure":
            return None
        
        # Extract steps from related entities or create default ones
        steps = []
        step_descriptions = {}
        relationships = []
        
        # Look for step-related information in entity properties
        if hasattr(entity, 'related_entities'):
            for related_entity in entity.related_entities:
                if related_entity.entity_type == "Step":
                    steps.append({
                        'name': related_entity.name,
                        'type': 'Step'
                    })
                    if hasattr(related_entity, 'description') and related_entity.description:
                        step_descriptions[related_entity.name] = related_entity.description
        
        # If no steps found, create default steps
        if not steps:
            # Generate default steps based on procedure name
            procedure_clean = procedure_name.replace(' ', '_').replace('/', '_').replace('-', '_')
            for i in range(1, 4):  # Create 3 default steps
                step_name = f"{procedure_clean}_step_{i}"
                steps.append({
                    'name': step_name,
                    'type': 'Step'
                })
                step_descriptions[step_name] = f"Execute step {i} of {procedure_name}"
        
        return {
            'name': procedure_name,
            'description': getattr(entity, 'description', f"3GPP procedure: {procedure_name}"),
            'steps': steps,
            'step_descriptions': step_descriptions,
            'relationships': relationships,
            'properties': getattr(entity, 'properties', {})
        }