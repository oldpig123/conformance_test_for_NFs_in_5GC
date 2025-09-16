from typing import List
from tqdm import tqdm
import warnings
import re
warnings.filterwarnings("ignore")

try:
    from neo4j import GraphDatabase
except ImportError:
    print("Error: neo4j not installed. Run: pip install neo4j")
    exit(1)

from data_structures import Entity, Relationship

class DatabaseManager:
    """Handles Neo4j database operations (Step 5-6)."""
    
    def __init__(self, uri: str, username: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        self.session = self.driver.session()  # ← ADD THIS LINE
    
    def close(self):
        if hasattr(self, 'session'):
            self.session.close()
        self.driver.close()

    def clear_database(self):
        """Clear all nodes and relationships in the database."""
        try:
            self.session.run("MATCH (n) DETACH DELETE n")
            print("  Database cleared successfully")
        except Exception as e:
            print(f"  Error clearing database: {e}")
    
    def create_entity(self, name: str, entity_type: str, properties: dict):
        """Create entity with proper error handling."""
        if properties is None:
            properties = {}
        
        # Add name to properties
        properties["name"] = name
        properties["entity_type"] = entity_type
        
        print(f"DEBUG: Creating entity {name} with type {entity_type}")
        
        query = f"""
        CREATE (e:{entity_type} $properties)
        RETURN e
        """
        
        try:
            result = self.session.run(query, properties=properties)
            record = result.single()
            if record:
                print(f"SUCCESS: Created {entity_type} entity '{name}'")
                return record
            else:
                print(f"WARNING: No entity created")
                return None
                
        except Exception as e:
            print(f"ERROR: Failed to create entity {name}: {e}")
            print(f"Query: {query}")
            print(f"Properties: {properties}")
            return None

    def create_relationship(self, source_name: str, target_name: str, rel_type: str, properties: dict = None):
        """Create relationship between two entities with proper error handling."""
        if properties is None:
            properties = {}
        
        # Sanitize relationship type (Neo4j doesn't like certain characters)
        clean_rel_type = re.sub(r'[^A-Za-z0-9_]', '_', rel_type)
        
        # print(f"DEBUG: Creating {source_name} -[{clean_rel_type}]-> {target_name}")
        
        # First verify both entities exist
        source_check = self.session.run("MATCH (n {name: $name}) RETURN count(n) as count", name=source_name)
        source_exists = source_check.single()["count"] > 0
        
        target_check = self.session.run("MATCH (n {name: $name}) RETURN count(n) as count", name=target_name)
        target_exists = target_check.single()["count"] > 0
        
        if not source_exists:
            print(f"ERROR: Source entity '{source_name}' does not exist!")
            return None
        
        if not target_exists:
            print(f"ERROR: Target entity '{target_name}' does not exist!")
            return None
        
        # Create relationship with parameterized query
        query = f"""
        MATCH (source {{name: $source_name}})
        MATCH (target {{name: $target_name}})
        CREATE (source)-[r:{clean_rel_type}]->(target)
        SET r += $properties
        RETURN r
        """
        
        try:
            result = self.session.run(query, 
                                     source_name=source_name, 
                                     target_name=target_name, 
                                     properties=properties)
            
            # Check if relationship was actually created
            record = result.single()
            if record:
                # print(f"SUCCESS: Created {clean_rel_type} relationship")
                return record
            else:
                print(f"WARNING: No relationship created")
                return None
                
        except Exception as e:
            print(f"ERROR: Failed to create relationship: {e}")
            print(f"Query: {query}")
            print(f"Params: source_name={source_name}, target_name={target_name}")
            return None

    def create_indexes(self, entity_types: List[str]):
        """Create indexes for entity types."""
        for entity_type in entity_types:
            try:
                query = f"CREATE INDEX IF NOT EXISTS FOR (n:{entity_type}) ON (n.name)"
                self.session.run(query)
                print(f"  ✓ Created index for {entity_type}")
            except Exception as e:
                print(f"  ✗ Failed to create index for {entity_type}: {e}")
    
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
    
    def create_procedure_scoped_entity(self, entity_type: str, name: str, procedure_name: str, properties: dict):
        """Create entities with proper procedure scoping."""
        
        # For entities that should be procedure-specific
        if entity_type in ["NetworkFunction", "Message", "Step"]:
            scoped_name = f"{procedure_name}_{name}"
            properties["original_name"] = name
            properties["procedure"] = procedure_name
            properties["scoped"] = True
        else:
            # Parameters and Keys can be global
            scoped_name = name
            properties["procedure"] = procedure_name
            properties["scoped"] = False
        
        properties["entity_type"] = entity_type
        
        query = f"""
        MERGE (e:{entity_type} {{name: $scoped_name}})
        SET e += $properties
        RETURN e
        """
        
        return self.session.run(query, scoped_name=scoped_name, properties=properties)

    def create_procedure_scoped_relationship(self, source_name: str, target_name: str, 
                                           rel_type: str, procedure_name: str, properties: dict = None):
        """Create relationships with proper entity scoping."""
        
        if properties is None:
            properties = {}
        
        properties["procedure"] = procedure_name
        
        # Helper to get correct entity name based on scoping rules
        def get_entity_name(name: str, context: str) -> str:
            # Check if this should be a scoped entity
            if any(keyword in context.lower() for keyword in ["step", "message"]) or \
               name in ["AMF", "SMF", "UPF", "AUSF", "UDM", "PCF", "NRF", "NEF", "NSSF", 
                       "UDR", "UDSF", "CHF", "BSF", "SEPP", "TNGF", "N3IWF", "W-AGF", 
                       "SMSF", "TSCTSF", "NSACF", "SCP", "GNB", "NG-ENB"]:
                return f"{procedure_name}_{name}"
            return name
        
        scoped_source = get_entity_name(source_name, "source")
        scoped_target = get_entity_name(target_name, "target") 
        
        query = f"""
        MATCH (source {{name: $source_name}})
        MATCH (target {{name: $target_name}})
        MERGE (source)-[r:{rel_type}]->(target)
        SET r += $properties
        RETURN r
        """
        
        return self.session.run(query, 
                             source_name=scoped_source, 
                             target_name=scoped_target, 
                             properties=properties)

    def clear_procedure_data(self, procedure_name: str):
        """Clear data for a specific procedure (useful for testing)."""
        query = """
        MATCH (n {procedure: $procedure_name})
        DETACH DELETE n
        """
        self.session.run(query, procedure_name=procedure_name)
    
    def verify_required_relationships(self):
        """Verify that required relationships were actually created."""
        required_types = ["FOLLOWED_BY", "PART_OF", "INVOKE", "INVOLVE", "CONTAINS", "SEND_BY", "SEND_TO"]
        
        print("\n=== VERIFYING RELATIONSHIP CREATION ===")
        
        for rel_type in required_types:
            query = f"MATCH ()-[r:{rel_type}]->() RETURN count(r) as count"
            try:
                result = self.session.run(query)
                count = result.single()["count"]
                
                if count > 0:
                    print(f"✓ {rel_type}: {count} relationships")
                    
                    # Show example relationships
                    example_query = f"""
                    MATCH (a)-[r:{rel_type}]->(b) 
                    RETURN a.name as source, b.name as target, r
                    LIMIT 2
                    """
                    examples = self.session.run(example_query)
                    for record in examples:
                        print(f"    Example: {record['source']} -[{rel_type}]-> {record['target']}")
                else:
                    print(f"⚠️  {rel_type}: {count} relationships - MISSING!")
                    
            except Exception as e:
                print(f"❌ {rel_type}: Error checking - {e}")