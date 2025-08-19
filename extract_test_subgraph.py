import argparse
from sentence_transformers import SentenceTransformer
from neo4j import GraphDatabase
import json

# configurations
NEO4J_URI = "bolt://localhost:7687"  # Neo4j
NEO4J_USER = "neo4j"  # Neo4j username
NEO4J_PASSWORD = "12345678"  # Neo4j password
VECTOR_INDEX_NAME = "requirement_embeddings"

class subgrapph_extractor:
    def __init__(self, neo4j_uri=NEO4J_URI, neo4j_user=NEO4J_USER, neo4j_password=NEO4J_PASSWORD):
        """model and database driver initialization."""
        # self.model = SentenceTransformer('Linq-AI-Research/Linq-Embed-Mistral')
        self.model = SentenceTransformer('Qwen/Qwen3-Embedding-8B')
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        print("Subgraph Extractor initialized.")
        
    def close(self):
        """Close the Neo4j driver connection."""
        if self.driver:
            self.driver.close()
            print("Neo4j connection closed.")
            
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
        RETURN ordered_text AS requirements,
               COLLECT(DISTINCT e.name) AS entities,
               ordered_titles AS titles
        """
        
        result = tx.run(query, index_name=VECTOR_INDEX_NAME, query_vector=query_vector, k=k)
        # print(result.single())
        return result.single()
    
    # @staticmethod
    # def _find_entity_and_expand(tx, entity_name):
        # """
        # Finds an entity by its name and returns all sections that contain it.
        # This is better for specific, known keywords.
        # """
        # query = """
        # // Find the starting entity by its name
        # MATCH (e:Entity)
        # WHERE e.name CONTAINS $entity_name
        # // Find all sections that contain this entity
        # MATCH (s:Requirement)-[:CONTAINS_ENTITY]->(e)
        # // Collect all unique section titles
        # WITH COLLECT(DISTINCT s.title) AS sections
        # RETURN sections, [$entity_name] AS entities // Return the searched entity name
        # """
        # result = tx.run(query, entity_name=entity_name)
        # return result.single()
    
    def get_focused_kg(self,query_text, top_k=5):
        """finds a starting point via sementc search and extracts a related subgraph"""
        print(f"\n target to {query_text}")
        
        # 1. encode the user query
        query_embedding = self.model.encode(query_text, convert_to_tensor=True).tolist()
        
        # 2. find the most relavant requirement and extract the subgraph
        with self.driver.session(database="neo4j") as session:
            result = session.execute_read(self._find_and_expand, query_vector=query_embedding, k=top_k)
            print(result)

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract a focused Knowledge Graph for a specific conformance test.")
    parser.add_argument("target_function", type=str, help="A description of the function or procedure you want to test.")
    args = parser.parse_args()

    extractor = subgrapph_extractor(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    result = extractor.get_focused_kg(args.target_function)
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