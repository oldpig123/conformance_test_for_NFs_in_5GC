from typing import List, Dict, Any, Optional
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings("ignore")

from data_structures import Entity, SearchQuery, SearchResult
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
    
    def index_entities(self, entities: List[Entity]):
        """Build search index from entities."""
        print(f"Indexing {len(entities)} entities for search...")
        
        # Store entities
        for entity in entities:
            self.entities_index[entity.name] = entity
        
        # Prepare text for TF-IDF
        self.entity_descriptions = []
        self.entity_names = []
        
        for entity in entities:
            # Combine name, description, and search keywords for indexing
            text_components = [entity.name]
            
            if entity.description:
                text_components.append(entity.description)
            
            if entity.search_keywords:
                text_components.extend(entity.search_keywords)
            
            # Add procedure name if available
            if 'procedure' in entity.properties:
                text_components.append(entity.properties['procedure'])
            
            combined_text = " ".join(text_components)
            self.entity_descriptions.append(combined_text)
            self.entity_names.append(entity.name)
        
        # Build TF-IDF matrix
        if self.entity_descriptions:
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.entity_descriptions)
            print(f"✓ Search index built with {len(self.entity_descriptions)} documents")
        
    def index_entities_with_long_context(self, entities: List[Entity], procedure_contexts: List[ProcedureContext]):
        """Index entities with full procedure context embeddings."""
        print("🔍 Indexing entities with long context embeddings...")
        
        # Index procedure contexts with full text
        for context in procedure_contexts:
            full_text = f"""
            Procedure: {context.procedure_name}
            Section: {context.section.title}
            Content: {context.section.text}
            
            Network Functions: {', '.join(context.network_functions)}
            Messages: {', '.join(context.messages)}
            Parameters: {', '.join(context.parameters)}
            Keys: {', '.join(context.keys)}
            Steps: {'; '.join(context.steps)}
            """
            
            # Use EntityExtractor's long context method
            entity_extractor = EntityExtractor()
            embedding_result = entity_extractor.generate_long_context_embeddings(
                full_text, context.procedure_name
            )
            
            if embedding_result["embedding"]:
                self.procedure_contexts[context.procedure_name] = embedding_result
                print(f"  ✓ Indexed '{context.procedure_name}': {embedding_result['metadata']['total_tokens']} tokens")
        
        # Index individual entities with enhanced context
        for entity in entities:
            if entity.entity_type == "Procedure":
                # Use the full context embedding for procedures
                if entity.name in self.procedure_contexts:
                    self.entity_embeddings[entity.id] = {
                        "entity": entity,
                        "embedding": self.procedure_contexts[entity.name]["embedding"],
                        "metadata": self.procedure_contexts[entity.name]["metadata"]
                    }
            else:
                # Generate context-aware embeddings for other entities
                context_text = f"""
                Entity: {entity.name}
                Type: {entity.entity_type}
                Properties: {entity.properties}
                """
                
                try:
                    embedding = self.embedding_model.encode(context_text).tolist()
                    self.entity_embeddings[entity.id] = {
                        "entity": entity,
                        "embedding": embedding,
                        "metadata": {"tokens": len(context_text.split())}
                    }
                except Exception as e:
                    print(f"  ❌ Failed to embed {entity.name}: {e}")

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
    
    def semantic_search_with_long_context(self, query: SearchQuery) -> List[SearchResult]:
        """Enhanced semantic search using long context embeddings."""
        if not self.entity_embeddings:
            return []
        
        # Generate query embedding with expanded context
        expanded_query = f"""
        Query: {query.query_text}
        Looking for: {', '.join(query.entity_types) if query.entity_types else 'any entity'}
        Context: 5G telecommunications procedures, network functions, messages, and protocols
        """
        
        try:
            query_embedding = self.embedding_model.encode(expanded_query)
        except Exception as e:
            print(f"Query embedding failed: {e}")
            return []
        
        results = []
        
        for entity_id, data in self.entity_embeddings.items():
            entity = data["entity"]
            entity_embedding = data["embedding"]
            
            if not entity_embedding:
                continue
            
            # Filter by entity type if specified
            if query.entity_types and entity.entity_type not in query.entity_types:
                continue
            
            # Calculate similarity
            similarity = self._calculate_cosine_similarity(query_embedding, entity_embedding)
            
            if similarity >= query.similarity_threshold:
                # Enhanced match type determination
                match_type = self._determine_enhanced_match_type(
                    query.query_text, entity, similarity, data.get("metadata", {})
                )
                
                results.append(SearchResult(
                    entity=entity,
                    similarity_score=similarity,
                    match_type=match_type,
                    metadata=data.get("metadata", {})
                ))
        
        # Sort by similarity score
        results.sort(key=lambda x: x.similarity_score, reverse=True)
        
        return results[:query.max_results]

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
            
            # Check search keywords
            entity_keywords = set([kw.lower() for kw in entity.search_keywords])
            keyword_matches = query_words.intersection(entity_keywords)
            if keyword_matches:
                matched_keywords.extend(list(keyword_matches))
                score += 0.6 * len(keyword_matches) / len(query_words)
            
            # Check description matching
            if entity.description:
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
                if entity.embedding:
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
                
                # Check search keywords
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
    
    def _determine_enhanced_match_type(self, query: str, entity: Entity, similarity: float, metadata: Dict) -> str:
        """Determine match type with enhanced context awareness."""
        query_lower = query.lower()
        entity_name_lower = entity.name.lower()
        
        # Check for exact matches
        if entity_name_lower in query_lower or query_lower in entity_name_lower:
            return "exact_match"
        
        # High similarity with long context
        if similarity > 0.8 and metadata.get("total_tokens", 0) > 10000:
            return "high_context_match"
        
        # Semantic matches
        if similarity > 0.7:
            return "semantic_match"
        elif similarity > 0.5:
            return "contextual_match"
        else:
            return "weak_match"