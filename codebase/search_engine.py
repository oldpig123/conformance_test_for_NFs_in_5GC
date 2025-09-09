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
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        self.tfidf_matrix = None
        self.entity_descriptions = []
        self.entity_names = []
        self.entity_embeddings = {}
        self.procedure_contexts = {}  # Store full context embeddings
        self.chunk_index = {}  # For chunk-level search
        
        print("ProcedureSearchEngine initialized")
    
    def build_search_index(self, entities: List[Entity]):
        """Build search index from entities (alias for index_entities)."""
        return self.index_entities(entities)
    
    def index_entities(self, entities: List[Entity]):
        """Build search index from entities with proper embedding handling."""
        print(f"Indexing {len(entities)} entities for search...")
        
        if not entities:
            print("Warning: No entities provided for indexing")
            return
        
        # Store entities
        for entity in entities:
            self.entities_index[entity.name] = entity
        
        # Prepare text for TF-IDF and embeddings
        self.entity_descriptions = []
        self.entity_names = []
        
        for entity in entities:
            # Combine name, description, and search keywords for indexing
            text_components = [entity.name]
            
            if hasattr(entity, 'description') and entity.description:
                text_components.append(entity.description)
            
            if hasattr(entity, 'search_keywords') and entity.search_keywords:
                text_components.extend(entity.search_keywords)
            
            # Add procedure name if available
            if hasattr(entity, 'properties') and 'procedure' in entity.properties:
                text_components.append(entity.properties['procedure'])
            
            combined_text = " ".join(text_components)
            
            # CRITICAL: Ensure text is within embedding limits
            if len(combined_text) > 1000:  # Conservative limit
                combined_text = combined_text[:1000] + "..."
            
            self.entity_descriptions.append(combined_text)
            self.entity_names.append(entity.name)
        
        # Build TF-IDF matrix
        if self.entity_descriptions:
            try:
                self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.entity_descriptions)
                print(f"✓ Search index built with {len(self.entity_descriptions)} documents")
            except Exception as e:
                print(f"Warning: TF-IDF indexing failed: {e}")
                self.tfidf_matrix = None
    
    def search(self, query: SearchQuery) -> List[SearchResult]:
        """Search for procedures using natural language query."""
        results = []
        
        # Method 1: Exact keyword matching
        exact_results = self._exact_keyword_search(query)
        results.extend(exact_results)
        
        # Method 2: TF-IDF similarity search
        tfidf_results = self._tfidf_search(query)
        results.extend(tfidf_results)
        
        # Method 3: Semantic search with embeddings (if available)
        if self.embedding_model:
            semantic_results = self._semantic_search(query)
            results.extend(semantic_results)
        
        # Deduplicate and rank results
        results = self._deduplicate_and_rank(results)
        
        # Filter by entity types if specified
        if query.entity_types:
            results = [r for r in results if r.entity.entity_type in query.entity_types]
        
        # Apply similarity threshold and limit
        results = [r for r in results if r.similarity_score >= query.similarity_threshold]
        results = results[:query.max_results]
        
        print(f"Search query: '{query.query_text}' -> {len(results)} results")
        
        return results
    
    def _exact_keyword_search(self, query: SearchQuery) -> List[SearchResult]:
        """Exact keyword matching search."""
        results = []
        query_words = set(query.query_text.lower().split())
        
        for entity_name, entity in self.entities_index.items():
            matched_keywords = []
            score = 0.0
            
            # Check name matching
            entity_words = set(entity.name.lower().split())
            name_matches = query_words.intersection(entity_words)
            if name_matches:
                matched_keywords.extend(list(name_matches))
                score += 0.8 * len(name_matches) / len(query_words)
            
            # Check search keywords if they exist
            if hasattr(entity, 'search_keywords') and entity.search_keywords:
                entity_keywords = set([kw.lower() for kw in entity.search_keywords])
                keyword_matches = query_words.intersection(entity_keywords)
                if keyword_matches:
                    matched_keywords.extend(list(keyword_matches))  # FIXED: Was keywordMatches
                    score += 0.6 * len(keyword_matches) / len(query_words)
            
            # Check description matching if it exists
            if hasattr(entity, 'description') and entity.description:
                desc_words = set(entity.description.lower().split())
                desc_matches = query_words.intersection(desc_words)
                if desc_matches:
                    matched_keywords.extend(list(desc_matches))
                    score += 0.4 * len(desc_matches) / len(query_words)
            
            if score > 0:
                results.append(SearchResult(
                    entity=entity,
                    similarity_score=min(score, 1.0),
                    match_type="exact",
                    matched_keywords=list(set(matched_keywords))
                ))
        
        return results
    
    def _tfidf_search(self, query: SearchQuery) -> List[SearchResult]:
        """TF-IDF based similarity search."""
        if self.tfidf_matrix is None:
            return []
        
        results = []
        
        try:
            # Transform query
            query_vector = self.tfidf_vectorizer.transform([query.query_text])
            
            # Calculate similarities
            similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
            
            # Create results
            for i, similarity in enumerate(similarities):
                if similarity > 0.1:  # Minimum threshold
                    entity_name = self.entity_names[i]
                    entity = self.entities_index[entity_name]
                    
                    results.append(SearchResult(
                        entity=entity,
                        similarity_score=float(similarity),
                        match_type="tfidf",
                        matched_keywords=[]
                    ))
        
        except Exception as e:
            print(f"TF-IDF search error: {e}")
        
        return results
    
    def _semantic_search(self, query: SearchQuery) -> List[SearchResult]:
        """Semantic search using embeddings."""
        if not self.embedding_model:
            return []
        
        results = []
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode(query.query_text)
            
            # Compare with entity embeddings
            for entity_name, entity in self.entities_index.items():
                if hasattr(entity, 'embedding') and entity.embedding:
                    # Calculate cosine similarity
                    entity_emb = np.array(entity.embedding)
                    similarity = np.dot(query_embedding, entity_emb) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(entity_emb)
                    )
                    
                    if similarity > 0.3:  # Minimum threshold for semantic similarity
                        results.append(SearchResult(
                            entity=entity,
                            similarity_score=float(similarity),
                            match_type="semantic",
                            matched_keywords=[]
                        ))
        
        except Exception as e:
            print(f"Semantic search error: {e}")
        
        return results
    
    def _deduplicate_and_rank(self, results: List[SearchResult]) -> List[SearchResult]:
        """Remove duplicates and rank results by score."""
        # Group by entity name, keep highest score
        entity_scores = {}
        entity_results = {}
        
        for result in results:
            entity_name = result.entity.name
            if entity_name not in entity_scores or result.similarity_score > entity_scores[entity_name]:
                entity_scores[entity_name] = result.similarity_score
                entity_results[entity_name] = result
        
        # Sort by score (descending)
        sorted_results = list(entity_results.values())
        sorted_results.sort(key=lambda x: x.similarity_score, reverse=True)
        
        return sorted_results
    
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