from typing import List
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

try:
    from neo4j import GraphDatabase
except ImportError:
    print("Error: neo4j not installed. Run: pip install neo4j")
    exit(1)

from data_structures import Entity, Relationship

class DatabaseManager:
    """Handles Neo4j database operations (Step 5-6)."""
    
    def __init__(self, uri: str, user: str, password: str):
        try:
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
            # Test connection
            with self.driver.session() as session:
                session.run("RETURN 1")
            print(f"✓ Connected to Neo4j at {uri}")
        except Exception as e:
            print(f"✗ Failed to connect to Neo4j: {e}")
            raise
    
    def clear_database(self):
        """Clear existing data."""
        print("  Clearing existing database...")
        with self.driver.session(database="neo4j") as session:
            session.run("MATCH (n) DETACH DELETE n")
    
    def create_entities(self, entities: List[Entity]):
        """Create entities in database."""
        print(f"  Creating {len(entities)} entities...")
        with self.driver.session(database="neo4j") as session:
            for entity in tqdm(entities, desc="Creating entities", leave=False):
                query = f"""
                MERGE (n:{entity.entity_type} {{name: $name}})
                SET n += $props
                """
                session.run(query, name=entity.name, props=entity.properties)
    
    def create_relationships(self, relationships: List[Relationship]):
        """Create relationships in database."""
        print(f"  Creating {len(relationships)} relationships...")
        with self.driver.session(database="neo4j") as session:
            for rel in tqdm(relationships, desc="Creating relationships", leave=False):
                query = f"""
                MATCH (a {{name: $source}}), (b {{name: $target}})
                MERGE (a)-[r:{rel.rel_type}]->(b)
                SET r += $props
                """
                session.run(query, 
                           source=rel.source_name, 
                           target=rel.target_name, 
                           props=rel.properties)
    
    def create_indexes(self, entity_types: List[str]):
        """Create database indexes."""
        print(f"  Creating indexes for {len(entity_types)} entity types...")
        with self.driver.session(database="neo4j") as session:
            for entity_type in entity_types:
                query = f"CREATE INDEX IF NOT EXISTS FOR (n:{entity_type}) ON (n.name)"
                session.run(query)
    
    def verify_relationships(self, rel_type: str) -> int:
        """Verify relationship count."""
        with self.driver.session(database="neo4j") as session:
            result = session.run(f"MATCH ()-[r:{rel_type}]->() RETURN count(r) as count")
            return result.single()["count"]
    
    def get_entity_count(self, entity_type: str) -> int:
        """Get entity count."""
        with self.driver.session(database="neo4j") as session:
            result = session.run(f"MATCH (n:{entity_type}) RETURN count(n) as count")
            return result.single()["count"]
    
    def get_sample_relationships(self, rel_type: str, limit: int = 5) -> List[dict]:
        """Get sample relationships."""
        with self.driver.session(database="neo4j") as session:
            result = session.run(f"""
                MATCH (a)-[r:{rel_type}]->(b) 
                RETURN a.name as source, b.name as target
                LIMIT {limit}
            """)
            return [record.data() for record in result]
    
    def close(self):
        """Close connection (Step 6)."""
        if self.driver:
            self.driver.close()
            print("✓ Neo4j connection closed")