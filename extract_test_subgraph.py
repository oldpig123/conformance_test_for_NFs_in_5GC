import argparse
from sentence_transformers import SentenceTransformer
from neo4j import GraphDatabase
import json
import spacy

# configurations
NEO4J_URI = "bolt://localhost:7687"  # Neo4j
NEO4J_USER = "neo4j"  # Neo4j username
NEO4J_PASSWORD = "12345678"  # Neo4j password
VECTOR_INDEX_NAME = "requirement_embeddings"
K_NUM = 5
SEMANTIC_FACTOR = 20  # Factor to scale the number of semantic results

class subgrapph_extractor:
    def __init__(self, neo4j_uri=NEO4J_URI, neo4j_user=NEO4J_USER, neo4j_password=NEO4J_PASSWORD):
        """model and database driver initialization."""
        
        self.model = SentenceTransformer('Qwen/Qwen3-Embedding-8B')  # Embedding model
        # self.model_1 = SentenceTransformer('Linq-AI-Research/Linq-Embed-Mistral')
        # self.model_2 = SentenceTransformer('Qwen/Qwen3-Embedding-8B')
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        self.nlp = spacy.load("en_core_web_trf")
        print("Subgraph Extractor initialized.")
        
    def close(self):
        """Close the Neo4j driver connection."""
        if self.driver:
            self.driver.close()
            print("Neo4j connection closed.")

    def extract_query_entity(self, query_text):
        """Extract entities from the user query using NLP."""
        doc = self.nlp(query_text)
        
        # Extract different types of entities
        entities = set()
        
        # 1. Named entities (using spaCy's NER)
        for ent in doc.ents:
            entities.add(ent.text)
        
        # 2. Noun chunks (similar to what we do in KG building)
        for chunk in doc.noun_chunks:
            if len(chunk.text.split()) <= 3:  # Keep shorter phrases
                entities.add(chunk.text)
        return list(entities)

    @staticmethod
    def _find_and_expand_hybrid(tx, query_vector, query_entities, k):
        """
        Hybrid search: combines vector similarity with entity matching.
        """
        query = """
        // Stage 1: Find starting nodes using vector search
        CALL db.index.vector.queryNodes($index_name, 3, $query_vector) YIELD node AS candidateNode, score AS vectorScore

        // Stage 2: Boost score if the node contains entities from the query
        MATCH (candidateNode)-[:CONTAINS_ENTITY]->(entity:Entity)
        WHERE entity.name IN $query_entities
        WITH candidateNode, vectorScore, COUNT(entity) AS entityMatches, COLLECT(entity.name) AS matchedEntities

        // Stage 3: Calculate hybrid score (vector similarity + entity bonus)
        WITH candidateNode, vectorScore, entityMatches, matchedEntities,
             (vectorScore + (entityMatches * 0.2)) AS hybridScore

        // Stage 4: Select the best starting node
        ORDER BY hybridScore DESC
        LIMIT 1
        WITH candidateNode AS startNode

        // Stage 5: Find all entities in the starting node
        MATCH (startNode)-[:CONTAINS_ENTITY]->(entity:Entity)

        // Stage 6: Find related requirements through shared entities
        MATCH (relatedReq:Requirement)-[:CONTAINS_ENTITY]->(entity)
        WHERE relatedReq <> startNode
        WITH startNode, COLLECT(DISTINCT relatedReq) AS uniqueRelatedReq

        // Stage 7: Calculate similarity and sort
        UNWIND uniqueRelatedReq AS req
        WITH req, vector.similarity.cosine(startNode.embedding, req.embedding) AS similarity, startNode
        ORDER BY similarity DESC
        LIMIT $k

        // Stage 8: Return results
        WITH COLLECT(req.text) AS ordered_text, startNode, COLLECT(DISTINCT req.title) AS ordered_titles
        MATCH (startNode)-[:CONTAINS_ENTITY]->(e:Entity)
        RETURN ordered_text AS requirements, COLLECT(DISTINCT e.name) AS entities, ordered_titles AS titles
        """
        
        result = tx.run(query, 
                       index_name=VECTOR_INDEX_NAME, 
                       query_vector=query_vector, 
                       query_entities=query_entities,
                       k=k)
        # print hybrid search method alert
        print("Hybrid search method used: vector similarity combined with entity matching.")
        return result.single()

    @staticmethod
    def _find_and_expand(tx, query_vector, k):
        """
        a single transaction that use a vector index to find starting node
        and then traverse the graph to find related information
        """
        
        query = """
        // Stage 1: Find the single most relevant section using the vector index.
        CALL db.index.vector.queryNodes($index_name, 1, $query_vector) YIELD node AS startNode

        // Stage 2: Find all entities within that starting section.
        MATCH (startNode)-[:CONTAINS_ENTITY]->(entity:Entity)

        // Stage 3: Find all other sections that are related via those same entities.
        // We use DISTINCT to handle each related section only once.
        MATCH (relatedReq:Requirement)-[:CONTAINS_ENTITY]->(entity)
        WHERE relatedReq <> startNode // Exclude the start node itself from the list
        WITH startNode, COLLECT(DISTINCT relatedReq) AS uniqueRelatedReq

        // Stage 4: Calculate the similarity score for each related section against the startNode.
        // UNWIND turns the collection back into individual rows to process them.
        UNWIND uniqueRelatedReq AS req
        WITH req, vector.similarity.cosine(startNode.embedding, req.embedding) AS similarity, startNode

        // Stage 5: Order the results by the calculated similarity, highest first.
        ORDER BY similarity DESC
        LIMIT $k  // Limit to the top k results

        // Stage 6: Collect the ordered titles and all related entities for the final output.
        // We also include the startNode's title at the top of the list.
        WITH COLLECT(req.text) AS ordered_text, startNode, COLLECT (req.title) AS ordered_titles
        MATCH (startNode)-[:CONTAINS_ENTITY]->(e:Entity)
        RETURN [startNode.text] + ordered_text AS requirements,
               COLLECT(DISTINCT e.name) AS entities,
               [startNode.title] + ordered_titles AS titles
        """
        
        result = tx.run(query, index_name=VECTOR_INDEX_NAME, query_vector=query_vector, k=k)
        # print(result.single())
        return result.single()
    
    @staticmethod
    def _score_by_entity_overlap(tx, candidate_titles, candidate_texts, query_entities):
        """
        Score candidate requirements by how many query entities they contain in the knowledge graph
        """
        query = """
        UNWIND $candidates AS candidate
        
        // Find the requirement node by title
        MATCH (req:Requirement {title: candidate.title})
        
        // Count how many query entities this requirement contains
        OPTIONAL MATCH (req)-[:CONTAINS_ENTITY]->(e:Entity)
        WHERE e.name IN $query_entities
        
        WITH req, candidate, COUNT(DISTINCT e) AS entityMatches, COLLECT(DISTINCT e.name) AS matchedEntities
        
        RETURN candidate.title AS title,
            candidate.text AS text,
            entityMatches AS entity_score,
            matchedEntities AS matched_entities
        """
        
        # Prepare candidates data
        candidates = [{"title": title, "text": text} for title, text in zip(candidate_titles, candidate_texts)]
        
        result = tx.run(query, candidates=candidates, query_entities=query_entities)
        return [{"title": record["title"], 
                "text": record["text"], 
                "entity_score": record["entity_score"],
                "matched_entities": record["matched_entities"]} 
                for record in result]    
        
    def get_focused_kg(self,query_text, top_k=5):
        """finds a starting point via sementc search and extracts a related subgraph"""
        print(f"\n target to {query_text}")
        
        # 1. encode the user query
        query_embedding = self.model.encode(query_text, convert_to_tensor=True).tolist()
        query_entities = self.extract_query_entity(query_text)
        print(f"extracted entities: {query_entities}")

        # 2. find the most relavant requirement and extract the subgraph
        with self.driver.session(database="neo4j") as session:
            result = session.execute_read(self._find_and_expand, query_vector=query_embedding, k=top_k)

            # result = session.execute_read(self._find_and_expand_hybrid, query_vector=query_embedding, query_entities=query_entities, k=top_k)

        # 3. Print the results
        if not result or not result['requirements']:
            print("could not find a match starting point in the knowledge graph")
            return
        
        print("\n--- Conformance Test Knowledge Graph Subgraph ---")
        print(f"Found {len(result['requirements'])} related requirements for the test")

        print("\n✅ Relevant Requirements:")
        # for req in result['requirements']:
        #     print(f"- {req}")
        for title, text in zip(result['titles'], result['requirements']):
            print(f"* {title} * :")
            print(f"  {text}")

        print("\n🔗 Related Entities:")
        # use set to show unique entities
        print(f"- {', '.join(set(result['entities']))}")
        print("\n--------------------------------------")

        return result



    def get_focus_kg_hybrid(self, query_text, top_k = 5, semantic_factor = 20):
        print(f"\nTarget: {query_text}")
        query_embedding = self.model.encode(query_text, convert_to_tensor=True).tolist()
        query_entities = self.extract_query_entity(query_text)
        print(f"Extracted entities: {query_entities}")

        with self.driver.session(database="neo4j") as session:
            # Step 1: Get top 20*k semantic results
            result_sem = session.execute_read(
                self._find_and_expand, query_vector=query_embedding, k=top_k * semantic_factor
            )

        # Step 2: Score semantic results by NLP entity overlap
        scored = []
        for title, text in zip(result_sem['titles'], result_sem['requirements']):
            score = sum(1 for ent in query_entities if ent.lower() in text.lower())
            scored.append((score, title, text))
        # Sort by NLP score (descending), then by original order
        scored.sort(key=lambda x: (-x[0], result_sem['titles'].index(x[1])))

        # Step 3: Select top k
        hybrid_titles = [item[1] for item in scored[:top_k]]
        hybrid_texts = [item[2] for item in scored[:top_k]]

        print("\n--- Hybrid Conformance Test Knowledge Graph Subgraph ---")
        print(f"Found {len(hybrid_titles)} requirements for the test")
        for title, text in zip(hybrid_titles, hybrid_texts):
            print(f"* {title} * :")
            print(f"  {text}")  # Show first 200 chars

        return {
            "titles": hybrid_titles,
            "requirements": hybrid_texts
        }
        
    def get_focus_kg_hybrid_2(self, query_text, top_k=5, semantic_factor=20):
        print(f"\nTarget: {query_text}")
        query_embedding = self.model.encode(query_text, convert_to_tensor=True).tolist()
        query_entities = self.extract_query_entity(query_text)
        print(f"Extracted entities: {query_entities}")

        with self.driver.session(database="neo4j") as session:
            # Step 1: Get top semantic_factor*k semantic results
            result_sem = session.execute_read(
                self._find_and_expand, query_vector=query_embedding, k=top_k * semantic_factor
            )
            
            # Step 2: Use knowledge graph entity search to score the semantic results
            result_with_scores = session.execute_read(
                self._score_by_entity_overlap, 
                candidate_titles=result_sem['titles'],
                candidate_texts=result_sem['requirements'], 
                query_entities=query_entities
            )

        # Step 3: Sort by entity overlap score and select top k
        sorted_results = sorted(result_with_scores, key=lambda x: x['entity_score'], reverse=True)
        
        # Step 4: Select top k
        hybrid_titles = [item['title'] for item in sorted_results[:top_k]]
        hybrid_texts = [item['text'] for item in sorted_results[:top_k]]

        print("\n--- Hybrid Conformance Test Knowledge Graph Subgraph ---")
        print(f"Found {len(hybrid_titles)} requirements for the test")
        for i, (title, text) in enumerate(zip(hybrid_titles, hybrid_texts)):
            score = sorted_results[i]['entity_score']
            print(f"* {title} * (Entity matches: {score}):")
            print(f"  {text}...")

        return {
            "titles": hybrid_titles,
            "requirements": hybrid_texts
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract a focused Knowledge Graph for a specific conformance test.")
    parser.add_argument("target_function", type=str, help="A description of the function or procedure you want to test.")
    args = parser.parse_args()

    extractor = subgrapph_extractor(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    # result = extractor.get_focused_kg(args.target_function, top_k=K_NUM)
    # result = extractor.get_focus_kg_hybrid(args.target_function, top_k=K_NUM, semantic_factor=SEMANTIC_FACTOR)
    result = extractor.get_focus_kg_hybrid_2(args.target_function, top_k=K_NUM, semantic_factor=SEMANTIC_FACTOR)
    # print result in json format
    print("\n--- Result ---")
    # print the result as following json format
    # {
    #     {
    #         "title": result['titles'],
    #         "text": result['requirements']
    #     },
    #     ...
    # }
    result_dict = []
    for title, text in zip(result['titles'], result['requirements']):
        # convert result as dic
        result_dict.append({
            "title": title,
            "text": text
        })
    print(json.dumps(result_dict, indent=2, ensure_ascii=False))

    print(json.dumps(result['requirements'], indent=2, ensure_ascii=False))
    extractor.close()