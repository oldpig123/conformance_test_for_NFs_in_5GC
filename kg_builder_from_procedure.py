import docx
import re
import spacy
from neo4j import GraphDatabase
import json
from typing import Dict, List, Tuple, Set
from dataclasses import dataclass
from pathlib import Path

# Configuration
NEO4J_URI = "bolt://localhost:7688"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "12345678"

@dataclass
class Entity:
    name: str
    entity_type: str
    properties: Dict = None

@dataclass
class Relationship:
    source: str
    target: str
    rel_type: str
    properties: Dict = None

class GPP_Procedure_KG_Builder:
    def __init__(self, neo4j_uri=NEO4J_URI, neo4j_user=NEO4J_USER, neo4j_password=NEO4J_PASSWORD):
        print("Initializing 3GPP Procedure Knowledge Graph Builder...")
        
        # Initialize NLP
        self.nlp = spacy.load("en_core_web_trf")
        
        # Initialize Neo4j driver
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        
        # Define our ontology
        self.setup_ontology()
        print("Initialization complete.")

    def setup_ontology(self):
        """Define the entities and relationships we want to extract"""
        
        # Network Functions in 5G
        self.network_functions = {
            'AMF', 'AUSF', 'UDM', 'SEAF', 'UPF', 'SMF', 'NSSF', 'NEF', 'NRF', 'PCF', 'UDSF',
            'AF', 'SMSF', 'NWDAF', 'CHF', 'BSF', 'LMF', 'GMLC', 'SLF'
        }
        
        # Other actors
        self.actors = {
            'UE', 'USIM', 'ME', 'gNB', 'NG-RAN', 'DN', 'HSS', 'HLR'
        }
        
        # Key cryptographic parameters and identifiers
        self.crypto_params = {
            'K_AMF', 'K_SEAF', 'K_AUSF', 'K_encr', 'K_int', 'SUCI', 'SUPI', 'RAND', 'AUTN', 
            'XRES', 'HXRES', 'CK', 'IK', 'RES', 'AUTS', 'SQN', 'MAC', 'AK', 'KAUSF', 'KSEAF',
            'Kasme', 'NH', 'NCC', 'GUTI', 'TMSI', 'IMSI', 'IMEI'
        }
        
        # Common procedures
        self.procedures = {
            '5G-AKA', 'EAP-AKA', 'Registration', 'Deregistration', 'Authentication', 
            'Key Derivation', 'Handover', 'Service Request', 'PDU Session Establishment',
            'Initial Context Setup', 'UE Configuration Update'
        }
        
        # Message patterns to look for
        self.message_patterns = [
            r'(\w+)\s+(?:Request|Response|Indication|Confirm)',
            r'N\w+_\w+_\w+',  # Interface messages like Nausf_UEAuthentication_Authenticate
            r'[A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*'  # Multi-word messages
        ]

    def close(self):
        if self.driver:
            self.driver.close()
            print("Neo4j connection closed.")

    def extract_text_from_docx(self, file_path: str) -> List[Dict]:
        """Extract structured text from DOCX with section information"""
        print(f"Extracting text from {file_path}...")
        
        try:
            doc = docx.Document(file_path)
            sections = []
            current_section = ""
            current_clause = ""
            current_text = ""
            
            for para in doc.paragraphs:
                text = para.text.strip()
                if not text:
                    continue
                    
                # Check if it's a heading (section/clause)
                if para.style.name.startswith('Heading') or re.match(r'^\d+\.[\d\.]*\s+', text):
                    # Save previous section
                    if current_text:
                        sections.append({
                            'section': current_section,
                            'clause': current_clause,
                            'text': current_text.strip(),
                            'document': Path(file_path).name
                        })
                    
                    # Extract clause number and section title
                    clause_match = re.match(r'^(\d+\.[\d\.]*)\s+(.*)', text)
                    if clause_match:
                        current_clause = clause_match.group(1)
                        current_section = clause_match.group(2)
                    else:
                        current_section = text
                        current_clause = ""
                    
                    current_text = ""
                else:
                    current_text += text + "\n"
            
            # Add the last section
            if current_text:
                sections.append({
                    'section': current_section,
                    'clause': current_clause,
                    'text': current_text.strip(),
                    'document': Path(file_path).name
                })
            
            print(f"Extracted {len(sections)} sections from {file_path}")
            return sections
            
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return []

    def extract_entities_and_relations(self, section: Dict) -> Tuple[List[Entity], List[Relationship]]:
        """Extract entities and relationships from a section - IMPROVED VERSION"""
        text = section['text']
        doc = self.nlp(text)
        
        entities = []
        relationships = []
        
        # Extract entities
        found_entities = set()
        
        # 1. Predefined entities (existing code)
        for nf in self.network_functions:
            if nf in text:
                entities.append(Entity(nf, "NetworkFunction"))
                found_entities.add(nf)
        
        for actor in self.actors:
            if actor in text:
                entities.append(Entity(actor, "Actor"))
                found_entities.add(actor)
        
        for param in self.crypto_params:
            if param in text or param.replace('_', '') in text:
                entities.append(Entity(param, "CryptoParameter"))
                found_entities.add(param)
        
        for proc in self.procedures:
            if proc.lower() in text.lower():
                entities.append(Entity(proc, "Procedure", {
                    'clause': section.get('clause', ''),
                    'section': section.get('section', '')
                }))
                found_entities.add(proc)
        
        # 2. USE SPACY NER to find additional entities
        for ent in doc.ents:
            if ent.label_ in ['ORG', 'PRODUCT', 'EVENT', 'FAC']:  # Relevant entity types
                if len(ent.text) > 2 and ent.text not in found_entities:
                    entities.append(Entity(ent.text, "NamedEntity"))
                    found_entities.add(ent.text)
        
        # 3. Extract Messages using improved patterns
        messages = set()
        improved_message_patterns = [
            r'(\w+)\s+(?:Request|Response|Indication|Confirm|Command|Reply|Acknowledgment)',
            r'N\w+_\w+_\w+',  # Interface messages
            r'[A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*'  # Multi-word messages
            r'[A-Z]{2,}\s+[A-Z]{2,}',  # Acronym combinations
            r'\b[A-Z]+\s+message\b',  # Messages explicitly called "message"
        ]
        
        for pattern in improved_message_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if isinstance(match, tuple):
                    match = ' '.join(match)
                if len(match) > 2 and match not in found_entities:
                    messages.add(match)
                    entities.append(Entity(match, "Message"))
                    found_entities.add(match)
        
        # 4. Extract relationships using improved pattern matching
        relationships.extend(self._extract_relationships(text, found_entities, messages))
        
        return entities, relationships

    def _extract_relationships(self, text: str, entities: Set[str], messages: Set[str]) -> List[Relationship]:
        """Extract relationships using improved pattern matching"""
        relationships = []
        
        # Convert entities to list for easier processing
        entity_list = list(entities)
        
        # 1. IMPROVED SENDS patterns (more flexible)
        send_patterns = [
            # Basic sends patterns
            r'(\w+)\s+sends?\s+([^\.]+?)\s+to\s+(\w+)',
            r'(\w+)\s+transmits?\s+([^\.]+?)\s+to\s+(\w+)',
            r'(\w+)\s+forwards?\s+([^\.]+?)\s+to\s+(\w+)',
            
            # More natural language patterns
            r'(\w+)\s+(?:shall\s+)?(?:send|transmit|forward)\s+([^\.]+?)\s+to\s+(?:the\s+)?(\w+)',
            r'([A-Z]+)\s+→\s+([A-Z]+)',  # Arrow notation
            r'From\s+(\w+)\s+to\s+(\w+)',
            r'(\w+)\s+requests?\s+([^\.]+?)\s+from\s+(\w+)',
            r'(\w+)\s+provides?\s+([^\.]+?)\s+to\s+(\w+)',
            
            # Message flow patterns
            r'(\w+)\s+issues?\s+([^\.]+?)\s+to\s+(\w+)',
            r'(\w+)\s+delivers?\s+([^\.]+?)\s+to\s+(\w+)',
            r'(\w+)\s+returns?\s+([^\.]+?)\s+to\s+(\w+)',
        ]
        
        for pattern in send_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if len(match) == 3:
                    sender, message, receiver = match
                    sender = sender.strip()
                    receiver = receiver.strip()
                    if sender in entities and receiver in entities:
                        relationships.append(Relationship(
                            sender, receiver, "SENDS",
                            {"message": message.strip()}
                        ))
                elif len(match) == 2:  # For arrow notation
                    sender, receiver = match
                    if sender in entities and receiver in entities:
                        relationships.append(Relationship(
                            sender, receiver, "COMMUNICATES_WITH"
                        ))

        # 2. IMPROVED DERIVES patterns
        derive_patterns = [
            # Key derivation patterns
            r'(\w+)\s+(?:is\s+)?derives?\s+([A-Z_]+)',
            r'([A-Z_]+)\s+(?:is\s+)?derived\s+(?:from\s+)?([A-Z_]+)',
            r'([A-Z_]+)\s+=\s+KDF\s*\([^)]*([A-Z_]+)[^)]*\)',  # KDF functions
            r'derives?\s+([A-Z_]+)\s+from\s+([A-Z_]+)',
            r'([A-Z_]+)\s+(?:shall\s+)?(?:be\s+)?calculated\s+(?:from\s+)?([A-Z_]+)',
            r'([A-Z_]+)\s+(?:shall\s+)?(?:be\s+)?computed\s+(?:from\s+)?([A-Z_]+)',
            r'generates?\s+([A-Z_]+)\s+(?:from\s+)?([A-Z_]+)',
        ]
        
        for pattern in derive_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if len(match) == 2:
                    source, target = match
                    source = source.strip()
                    target = target.strip()
                    # Check if both are crypto parameters or entities
                    if (source in self.crypto_params or source in entities) and \
                       (target in self.crypto_params or target in entities):
                        relationships.append(Relationship(
                            source, target, "DERIVES"
                        ))

        # 3. CONTAINS patterns (messages contain parameters)
        contains_patterns = [
            r'([^\.]+?)\s+contains?\s+([A-Z_]+)',
            r'([^\.]+?)\s+includes?\s+([A-Z_]+)',
            r'([A-Z_]+)\s+(?:is\s+)?(?:included\s+)?in\s+([^\.]+)',
            r'([^\.]+?)\s+carries?\s+([A-Z_]+)',
            r'([^\.]+?)\s+(?:shall\s+)?(?:include|contain)\s+([A-Z_]+)',
        ]
        
        for pattern in contains_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                container, contained = match
                container = container.strip()
                contained = contained.strip()
                if contained in self.crypto_params or contained in entities:
                    relationships.append(Relationship(
                        container, contained, "CONTAINS"
                    ))

        # 4. INITIATES patterns
        initiate_patterns = [
            r'(\w+)\s+initiates?\s+([^\.]+)',
            r'([^\.]+?)\s+(?:is\s+)?initiated\s+by\s+(\w+)',
            r'(\w+)\s+starts?\s+([^\.]+)',
            r'(\w+)\s+triggers?\s+([^\.]+)',
            r'(\w+)\s+(?:shall\s+)?(?:begin|start|initiate)\s+([^\.]+)',
            r'([^\.]+?)\s+(?:shall\s+)?(?:be\s+)?(?:started|initiated|triggered)\s+by\s+(\w+)',
        ]
        
        for pattern in initiate_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if len(match) == 2:
                    initiator, procedure = match
                    initiator = initiator.strip()
                    procedure = procedure.strip()
                    if initiator in entities:
                        relationships.append(Relationship(
                            initiator, procedure, "INITIATES"
                        ))

        # 5. AUTHENTICATES patterns
        auth_patterns = [
            r'(\w+)\s+authenticates?\s+(\w+)',
            r'(\w+)\s+(?:shall\s+)?(?:verify|validate)\s+(\w+)',
            r'(\w+)\s+(?:performs?\s+)?authentication\s+(?:of\s+)?(\w+)',
            r'authentication\s+(?:of\s+)?(\w+)\s+(?:by\s+)?(\w+)',
        ]
        
        for pattern in auth_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                authenticator, authenticated = match
                authenticator = authenticator.strip()
                authenticated = authenticated.strip()
                if authenticator in entities and authenticated in entities:
                    relationships.append(Relationship(
                        authenticator, authenticated, "AUTHENTICATES"
                    ))

        # 6. STORES patterns
        store_patterns = [
            r'(\w+)\s+stores?\s+([A-Z_]+)',
            r'([A-Z_]+)\s+(?:is\s+)?stored\s+(?:in|at)\s+(\w+)',
            r'(\w+)\s+(?:shall\s+)?(?:keep|maintain|store)\s+([A-Z_]+)',
        ]
        
        for pattern in store_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if len(match) == 2:
                    store_entity, stored_item = match
                    store_entity = store_entity.strip()
                    stored_item = stored_item.strip()
                    if store_entity in entities and (stored_item in self.crypto_params or stored_item in entities):
                        relationships.append(Relationship(
                            store_entity, stored_item, "STORES"
                        ))

        # 7. VALIDATES patterns
        validate_patterns = [
            r'(\w+)\s+validates?\s+([^\.]+)',
            r'(\w+)\s+verifies?\s+([^\.]+)',
            r'(\w+)\s+checks?\s+([^\.]+)',
            r'([^\.]+?)\s+(?:is\s+)?(?:validated|verified|checked)\s+by\s+(\w+)',
        ]
        
        for pattern in validate_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                validator, validated = match
                validator = validator.strip()
                validated = validated.strip()
                if validator in entities:
                    relationships.append(Relationship(
                        validator, validated, "VALIDATES"
                    ))

        # 8. USES patterns
        use_patterns = [
            r'(\w+)\s+uses?\s+([A-Z_]+)',
            r'(\w+)\s+employs?\s+([A-Z_]+)',
            r'(\w+)\s+utilizes?\s+([A-Z_]+)',
            r'([A-Z_]+)\s+(?:is\s+)?used\s+by\s+(\w+)',
        ]
        
        for pattern in use_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if len(match) == 2:
                    user, used = match
                    user = user.strip()
                    used = used.strip()
                    if user in entities and (used in self.crypto_params or used in entities):
                        relationships.append(Relationship(
                            user, used, "USES"
                        ))

        # 9. PART_OF patterns (for procedures and steps)
        part_of_patterns = [
            r'([^\.]+?)\s+(?:is\s+)?(?:part\s+of|belongs\s+to)\s+([^\.]+)',
            r'([^\.]+?)\s+(?:in|during)\s+([^\.]+?)\s+procedure',
            r'step\s+\d+\s+of\s+([^\.]+)',
        ]
        
        for pattern in part_of_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if len(match) == 2:
                    part, whole = match
                    part = part.strip()
                    whole = whole.strip()
                    relationships.append(Relationship(
                        part, whole, "PART_OF"
                    ))

        # 10. REQUIRES patterns
        require_patterns = [
            r'(\w+)\s+requires?\s+([^\.]+)',
            r'(\w+)\s+needs?\s+([^\.]+)',
            r'([^\.]+?)\s+(?:is\s+)?required\s+(?:by|for)\s+(\w+)',
            r'(\w+)\s+depends?\s+on\s+([^\.]+)',
        ]
        
        for pattern in require_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if len(match) == 2:
                    requirer, required = match
                    requirer = requirer.strip()
                    required = required.strip()
                    if requirer in entities:
                        relationships.append(Relationship(
                            requirer, required, "REQUIRES"
                        ))

        return relationships

    def create_knowledge_graph(self, file_paths: List[str]):
        """Main method to create the knowledge graph from multiple files"""
        print("Starting knowledge graph creation...")
        
        all_entities = []
        all_relationships = []
        
        for file_path in file_paths:
            print(f"\nProcessing {file_path}...")
            sections = self.extract_text_from_docx(file_path)
            
            for section in sections:
                entities, relationships = self.extract_entities_and_relations(section)
                
                # Add document reference to entities
                for entity in entities:
                    if entity.properties is None:
                        entity.properties = {}
                    entity.properties['source_document'] = section['document']
                    entity.properties['source_section'] = section.get('section', '')
                    entity.properties['source_clause'] = section.get('clause', '')
                
                all_entities.extend(entities)
                all_relationships.extend(relationships)
        
        print(f"\nExtracted {len(all_entities)} entities and {len(all_relationships)} relationships")
        
        # Load into Neo4j
        self._load_to_neo4j(all_entities, all_relationships)
        
        print("Knowledge graph creation completed!")

    def _load_to_neo4j(self, entities: List[Entity], relationships: List[Relationship]):
        """Load entities and relationships into Neo4j"""
        print("Loading data into Neo4j...")
        
        with self.driver.session() as session:
            # Clear existing data (optional)
            # session.run("MATCH (n) DETACH DELETE n")
            
            # Create entities
            for entity in entities:
                self._create_entity(session, entity)
            
            # Create relationships
            for rel in relationships:
                self._create_relationship(session, rel)
        
        print("Data loaded successfully!")

    def _create_entity(self, session, entity: Entity):
        """Create a single entity in Neo4j"""
        props = entity.properties or {}
        props['name'] = entity.name
        
        query = f"""
        MERGE (e:{entity.entity_type} {{name: $name}})
        SET e += $properties
        """
        
        session.run(query, name=entity.name, properties=props)

    def _create_relationship(self, session, rel: Relationship):
        """Create a single relationship in Neo4j"""
        props = rel.properties or {}
        
        query = f"""
        MATCH (a {{name: $source}})
        MATCH (b {{name: $target}})
        MERGE (a)-[r:{rel.rel_type}]->(b)
        SET r += $properties
        """
        
        session.run(query, source=rel.source, target=rel.target, properties=props)

    def create_indexes(self):
        """Create useful indexes for the knowledge graph"""
        print("Creating indexes...")
        
        with self.driver.session() as session:
            indexes = [
                "CREATE INDEX entity_name IF NOT EXISTS FOR (n) ON (n.name)",
                "CREATE INDEX nf_name IF NOT EXISTS FOR (n:NetworkFunction) ON (n.name)",
                "CREATE INDEX actor_name IF NOT EXISTS FOR (n:Actor) ON (n.name)",
                "CREATE INDEX procedure_name IF NOT EXISTS FOR (n:Procedure) ON (n.name)",
                "CREATE INDEX message_name IF NOT EXISTS FOR (n:Message) ON (n.name)",
                "CREATE INDEX crypto_name IF NOT EXISTS FOR (n:CryptoParameter) ON (n.name)"
            ]
            
            for index_query in indexes:
                try:
                    session.run(index_query)
                except Exception as e:
                    print(f"Index creation warning: {e}")
        
        print("Indexes created!")

    def run_sample_queries(self):
        """Run sample queries to demonstrate the knowledge graph"""
        print("\n=== Sample Queries ===")
        
        with self.driver.session() as session:
            # Count all relationship types
            print("\n1. Relationship Type Counts:")
            result = session.run("""
                MATCH ()-[r]->()
                RETURN type(r) as relationship_type, count(r) as count
                ORDER BY count DESC
            """)
            for record in result:
                print(f"   - {record['relationship_type']}: {record['count']}")
            
            # Show all network functions
            print("\n2. All Network Functions:")
            result = session.run("MATCH (nf:NetworkFunction) RETURN nf.name ORDER BY nf.name")
            for record in result:
                print(f"   - {record['nf.name']}")
            
            # Show authentication relationships
            print("\n3. Authentication relationships:")
            result = session.run("""
                MATCH (a)-[:AUTHENTICATES]->(b)
                RETURN a.name, b.name
                LIMIT 10
            """)
            for record in result:
                print(f"   - {record['a.name']} authenticates {record['b.name']}")
            
            # Show key derivations
            print("\n4. Key derivations:")
            result = session.run("""
                MATCH (source)-[:DERIVES]->(target)
                RETURN source.name, target.name
                LIMIT 10
            """)
            for record in result:
                print(f"   - {record['source.name']} derives {record['target.name']}")

if __name__ == "__main__":
    # Initialize builder
    builder = GPP_Procedure_KG_Builder()
    
    try:
        # Get all DOCX files from 3GPP directory
        gpp_dir = Path("3GPP")
        docx_files = list(gpp_dir.glob("*_new.docx"))
        
        if not docx_files:
            print("No DOCX files found in 3GPP directory!")
            exit(1)
        
        print(f"Found {len(docx_files)} DOCX files:")
        for file in docx_files:
            print(f"  - {file}")
        
        # Create knowledge graph
        builder.create_knowledge_graph([str(f) for f in docx_files])
        
        # Create indexes for better performance
        builder.create_indexes()
        
        # Run sample queries
        builder.run_sample_queries()
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        builder.close()