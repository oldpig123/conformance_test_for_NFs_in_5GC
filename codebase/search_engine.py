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

        # Field-specific vectorizers and matrices for keyword search (titles only)
        self.title_vectorizer = TfidfVectorizer(stop_words='english', max_features=1000, ngram_range=(1, 2))
        self.parent_title_vectorizer = TfidfVectorizer(stop_words='english', max_features=1000, ngram_range=(1, 2))
        self.title_matrix = None
        self.parent_title_matrix = None

        # Weights for combining scores
        self.W_TITLE = 2.2
        self.W_PARENT = 1.8
        self.W_SEMANTIC = 3.5
        
        print("ProcedureSearchEngine initialized with field-based ranking")
    
    def search(self, query: SearchQuery) -> List[SearchResult]:
        """Search for procedures using a hybrid, field-weighted scoring model."""
        
        # 1. Get keyword scores from title and parent_title fields
        keyword_scores = self._keyword_search(query)
        
        # 2. Get semantic scores from the LLM-generated summary (in entity.description)
        semantic_scores = self._semantic_search(query)
        
        # 3. Combine, rank, and produce final results
        results = self._deduplicate_and_rank(keyword_scores, semantic_scores)
        
        # Filter by entity types if specified
        if query.entity_types:
            results = [r for r in results if r.entity.entity_type in query.entity_types]
        
        # Apply similarity threshold and limit
        results = [r for r in results if r.similarity_score >= query.similarity_threshold]
        results = results[:query.max_results]
        
        print(f"Search query: '{query.query_text}' -> {len(results)} results")
        
        return results
    
    def build_search_index(self, entities: List[Entity]):
        """Builds separate search indexes for title and parent_title fields."""
        print(f"Indexing {len(entities)} entities for field-based search...")
        if not entities:
            print("Warning: No entities provided for indexing")
            return

        # Prepare texts for each field
        self.entities_index = {entity.name: entity for entity in entities}
        self.entity_names = [entity.name for entity in entities]
        
        titles = [entity.name for entity in entities]
        parent_titles = [entity.parent_title if entity.parent_title else '' for entity in entities]

        # Build TF-IDF matrix for each field
        try:
            if any(titles):
                self.title_matrix = self.title_vectorizer.fit_transform(titles)
            if any(parent_titles):
                self.parent_title_matrix = self.parent_title_vectorizer.fit_transform(parent_titles)
            print(f"✓ Field-based search index built with {len(entities)} documents")
        except Exception as e:
            print(f"Warning: TF-IDF indexing failed: {e}")
            
    def _keyword_search(self, query: SearchQuery) -> Dict[str, float]:
        """Performs TF-IDF search across title and parent_title fields and returns a dict of scores."""
        scores = {name: 0.0 for name in self.entity_names}
        if not hasattr(self, 'title_matrix'):
            return scores

        sim_title = np.zeros(len(self.entity_names))
        sim_parent = np.zeros(len(self.entity_names))

        if hasattr(self, 'title_matrix') and self.title_matrix is not None:
            query_vec_title = self.title_vectorizer.transform([query.query_text])
            sim_title = cosine_similarity(query_vec_title, self.title_matrix).flatten()

        if hasattr(self, 'parent_title_matrix') and self.parent_title_matrix is not None:
            query_vec_parent = self.parent_title_vectorizer.transform([query.query_text])
            sim_parent = cosine_similarity(query_vec_parent, self.parent_title_matrix).flatten()

        # Calculate weighted score for each entity
        for i, entity_name in enumerate(self.entity_names):
            weighted_score = (
                sim_title[i] * self.W_TITLE +
                sim_parent[i] * self.W_PARENT
            )
            scores[entity_name] = weighted_score
        
        return scores

    def _semantic_search(self, query: SearchQuery) -> Dict[str, float]:
        """Performs semantic search and returns a dict of scores."""
        scores = {name: 0.0 for name in self.entity_names}
        if not self.embedding_model:
            return scores

        try:
            query_embedding = self.embedding_model.encode(query.query_text)
            for entity_name, entity in self.entities_index.items():
                if entity.embedding:
                    entity_emb = np.array(entity.embedding)
                    similarity = np.dot(query_embedding, entity_emb) / (np.linalg.norm(query_embedding) * np.linalg.norm(entity_emb))
                    scores[entity_name] = float(similarity)
        except Exception as e:
            print(f"Semantic search error: {e}")
        
        return scores

    def _deduplicate_and_rank(self, keyword_scores: Dict, semantic_scores: Dict) -> List[SearchResult]:
        """Combines keyword and semantic scores and ranks the results."""
        final_scores = {}
        for entity_name in self.entity_names:
            keyword_score = keyword_scores.get(entity_name, 0.0)
            semantic_score = semantic_scores.get(entity_name, 0.0)

            # Combine scores
            final_score = keyword_score + (semantic_score * self.W_SEMANTIC)

            if final_score > 0.1: # Apply a threshold
                final_scores[entity_name] = final_score

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