import os
import re
from pathlib import Path
from typing import List, Dict, Set, Tuple, NamedTuple
import docx
from neo4j import GraphDatabase

# --- Configuration ---
NEO4J_URI = "bolt://localhost:7688"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "12345678"  # Replace with your Neo4j password
DOCS_PATH = Path("3GPP")

# --- Data Structures ---
class Entity(NamedTuple):
    name: str
    entity_type: str
    properties: Dict = {}

class Relationship(NamedTuple):
    source_name: str
    target_name: str
    rel_type: str
    properties: Dict = {}

# --- Main Builder Class ---
class KnowledgeGraphBuilder:
    """
    Extracts entities and relationships from 3GPP documents and builds a Neo4j knowledge graph.
    """
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.ontology = {}
        self.all_known_entities = set()
        self._setup_ontology()
        print("KnowledgeGraphBuilder initialized.")

    def _setup_ontology(self):
        """Defines the data model (schema) for the knowledge graph."""
        self.ontology = {
            'NetworkFunction': {'AMF', 'AUSF', 'UDM', 'UDR', 'PCF', 'SMF', 'UPF', 'NSSF', 'NEF', 'NRF', 'SEAF'},
            'Actor': {'UE', 'USIM', 'ME', 'gNB', 'NG-RAN', 'DN'},
            'Procedure': {
                'Registration procedure', 'Authentication procedure', '5G AKA',
                'Security Mode Control procedure', 'Initial Context Setup', 'Handover', 'UE Requested PDU session establishment',
                'Initiation of authentication and selection of authentication method'
            },
            'Key': {'KAMF', 'KSEAF', 'KAUSF', 'KGNB', 'HXRES*', 'XRES*','AV','RES*'},
            'Parameter': {'SUCI', 'RAND', 'AUTN', 'HXRES*', 'XRES*', 'SUPI', '5G-GUTI', 'IMSI', 'IMEI'},
            'State': {'CM-IDLE', 'CM-CONNECTED', 'RRC_IDLE', 'RRC_CONNECTED', 'RRC_INACTIVE'},
        }
        # Create a flat set for quick lookups
        for entity_set in self.ontology.values():
            self.all_known_entities.update(entity_set)
        print("Ontology defined.")

    def close(self):
        """Closes the Neo4j database connection."""
        if self.driver:
            self.driver.close()
            print("Neo4j connection closed.")

    def extract_text_from_docx(self, file_path: Path) -> List[Dict]:
        """Extracts structured text from a DOCX file, associating text with clauses."""
        print(f"Extracting text from: {file_path.name}")
        try:
            doc = docx.Document(file_path)
            sections = []
            current_clause = "Unknown"
            current_text = []

            # Regex to find clause numbers like 6.1.3.2 or A.4
            clause_pattern = re.compile(r'^[A-Z]?\.?\d+(\.\d+)*\s+')

            for para in doc.paragraphs:
                if para.text.strip():
                    match = clause_pattern.match(para.text)
                    # Check if the paragraph is a heading or starts with a clause number
                    if para.style.name.startswith('Heading') or match:
                        if current_text: # Save the previous section
                            sections.append({
                                'document': file_path.stem.replace('_new', ''),
                                'clause': current_clause.split('\t')[0].strip(),
                                'text': "\n".join(current_text)
                            })
                        current_clause = para.text.strip()
                        current_text = []
                    else:
                        current_text.append(para.text.strip())
            
            if current_text: # Add the last section
                sections.append({
                    'document': file_path.stem.replace('_new', ''),
                    'clause': current_clause.split('\t')[0].strip(),
                    'text': "\n".join(current_text)
                })
            
            print(f"  -> Found {len(sections)} sections.")
            return sections
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return []

    def extract_from_section(self, section: Dict) -> Tuple[Set[Entity], Set[Relationship]]:
        """Extracts entities and relationships from a single text section."""
        text = section['text']
        entities = []
        relationships = []

        found_entity_keys = set()

        # 1. Find all known entities mentioned in the text
        found_entity_names = set()
        for entity_type, entity_set in self.ontology.items():
            for entity_name in entity_set:
                # Use case-insensitive search and ensure it's a whole word
                if re.search(r'\b' + re.escape(entity_name) + r'\b', text, re.IGNORECASE):
                    entity_key = (entity_name, entity_type)
                    
                    # Add the entity object only once per section to avoid duplicates
                    if entity_key not in found_entity_keys:
                        props = {'source_clause': section['clause'], 'source_document': section['document']}
                        entities.append(Entity(name=entity_name, entity_type=entity_type, properties=props))
                        found_entity_keys.add(entity_key)
                    
                    # Correctly add the name to the set of entities found *in this section*
                    found_entity_names.add(entity_name)

        # 2. Extract relationships using regex
        entity_list_str = '|'.join(map(re.escape, self.all_known_entities))
        
        # SENDS: (Entity) sends [Message] to (Entity)
        sends_pattern = re.compile(r'([A-Z0-9/ -]+)\s+sends an?\s+(.+?)\s+to the\s+([A-Z0-9/ -]+)')
        for match in sends_pattern.finditer(text):
            src, msg, tgt = [m.strip() for m in match.groups()]
            if src in found_entity_names and tgt in found_entity_names:
                # Add the message as a node
                msg_key = (msg, 'Message')
                if msg_key not in found_entity_keys:
                    msg_props = {'source_clause': section['clause'], 'source_document': section['document']}
                    entities.append(Entity(name=msg, entity_type='Message', properties=msg_props))
                    found_entity_keys.add(msg_key)
                # Add relationships
                rel_props = {'step': 'N/A', 'condition': ''} # Placeholder for step/condition
                relationships.append(Relationship(src, msg, 'SENDS_TO', rel_props))
                relationships.append(Relationship(msg, tgt, 'IS_SENT_TO', rel_props))
                relationships.append(Relationship(msg, src, 'SENT_BY', rel_props))


        # DERIVES: (Entity) derives (Key) from (Key/Param)
        # derives_pattern = re.compile(r'(' + entity_list_str + r')\s+derives?\s+the\s+(' + entity_list_str + r')\s+from\s+(' + entity_list_str + r')')
        # for src, key, param in derives_pattern.findall(text):
        #     if src in found_entity_names and key in found_entity_names and param in found_entity_names:
        #         relationships.append(Relationship(src, key, 'DERIVES'))
        #         relationships.append(Relationship(key, param, 'DERIVED_FROM'))
        
        # This logic is more robust. It finds the word "derive" and then looks
        # for the entities involved in the surrounding text.

        # Find all sentences containing "derive" or its variations.
        derive_sentences = [s.strip() for s in re.split(r'[.\n]', text) if 'deriv' in s]

        for sentence in derive_sentences:
            # Find all known entities mentioned in this specific sentence
            sentence_entities = []
            for entity_name in found_entity_names:
                if re.search(r'\b' + re.escape(entity_name) + r'\b', sentence):
                    sentence_entities.append(entity_name)

            if len(sentence_entities) < 2:
                continue # Need at least two entities to form a relationship

            # A simple heuristic: The first entity is the deriver, the second is the derived key.
            # This is an assumption but works for many common phrasings.
            # e.g., "The SEAF derives the K_AMF..."
            # e.g., "The K_AMF is derived by the SEAF..." (Here it might be reversed, but it's a start)

            # Let's assume a simple S-V-O (Subject-Verb-Object) or O-V-S (Object-Verb-Subject) structure
            # We will create a DERIVES relationship between the first two entities found.
            
            # Find the position of the verb "derive"
            derive_match = re.search(r'\bderiv(?:es|ed|ing)\b', sentence)
            if not derive_match:
                continue

            verb_pos = derive_match.start()

            # Find positions of entities
            entity_positions = {}
            for entity in sentence_entities:
                # Find the first occurrence of the entity in the sentence
                entity_match = re.search(r'\b' + re.escape(entity) + r'\b', sentence)
                if entity_match:
                    entity_positions[entity] = entity_match.start()

            # Sort entities by their position in the sentence
            sorted_entities = sorted(entity_positions.keys(), key=lambda e: entity_positions[e])

            if len(sorted_entities) >= 2:
                # A common pattern is that the entity before the verb "derive" is the source,
                # and the one after is the target.
                source_entity = None
                derived_key = None

                for entity in reversed(sorted_entities):
                    if entity_positions[entity] < verb_pos:
                        source_entity = entity
                        break
                
                for entity in sorted_entities:
                    if entity_positions[entity] > verb_pos:
                        derived_key = entity
                        break

                if source_entity and derived_key:
                    print(f"  -> Found DERIVES relationship: ({source_entity}) -[:DERIVES]-> ({derived_key})")
                    relationships.append(Relationship(source_entity, derived_key, 'DERIVES', {}))

                    # If there's a third entity, assume it's part of the derivation
                    if len(sorted_entities) > 2:
                        # Find a parameter that isn't the source or the key
                        param = next((e for e in sorted_entities if e not in [source_entity, derived_key]), None)
                        if param:
                             relationships.append(Relationship(derived_key, param, 'DERIVED_FROM', {}))

        # CONTAINS: (Message) contains (Parameter)
        # This is often implicit. We link parameters to the first entity found in the text.
        if found_entity_names:
            first_entity = next(iter(found_entity_names))
            for param_type in ['Key', 'Parameter']:
                for param_name in self.ontology[param_type]:
                    if param_name in text:
                        relationships.append(Relationship(first_entity, param_name, 'CONTAINS'))

        # PART_OF: (Action/Message) is part of a (Procedure)
        for proc_name in self.ontology['Procedure']:
            if proc_name in text:
                for entity in entities:
                    if entity.name != proc_name:
                        relationships.append(Relationship(entity.name, proc_name, 'PART_OF'))

        return entities, relationships

    def build_graph_from_files(self, file_paths: List[Path]):
        """Orchestrates the entire KG construction process."""
        print("\n--- Starting Knowledge Graph Construction ---")
        all_entities = {}
        all_relationships = []

        for file_path in file_paths:
            # Create a node for the specification document itself
            spec_name = file_path.stem.replace('_new', '')
            spec_version = spec_name.split('-')[1]
            spec_doc = f"TS {spec_name.split('-')[0]}"
            spec_entity = Entity(name=spec_name, entity_type='Specification', properties={'document': spec_doc, 'version': spec_version})
            all_entities[(spec_entity.name, spec_entity.entity_type)] = spec_entity

            sections = self.extract_text_from_docx(file_path)
            for section in sections:
                entities, relationships = self.extract_from_section(section)
                for entity in entities:
                    all_entities[(entity.name, entity.entity_type)] = entity
                all_relationships.extend(relationships)
        
        print(f"\n--- Extraction Complete ---")
        print(f"Total unique entities found: {len(all_entities)}")
        print(f"Total unique relationships found: {len(all_relationships)}")
        
        self._load_to_neo4j(all_entities.values(), all_relationships)
        self._create_indexes()

    def _load_to_neo4j(self, entities: List[Entity], relationships: Set[Relationship]):
        """Loads the extracted entities and relationships into Neo4j."""
        print("\n--- Loading Data into Neo4j ---")
        with self.driver.session(database="neo4j") as session:
            # Clear existing graph (optional, for clean runs)
            print("Clearing existing database...")
            session.run("MATCH (n) DETACH DELETE n")

            # Create Nodes
            print(f"Creating {len(entities)} nodes...")
            for entity in entities:
                query = f"""
                MERGE (n:{entity.entity_type} {{name: $name}})
                ON CREATE SET n += $props
                ON MATCH SET n += $props
                """
                session.run(query, name=entity.name, props=entity.properties)

            # Create Relationships
            print(f"Creating {len(relationships)} relationships...")
            for rel in relationships:
                query = f"""
                MATCH (a {{name: $source_name}}), (b {{name: $target_name}})
                MERGE (a)-[r:{rel.rel_type}]->(b)
                ON CREATE SET r += $props
                """
                session.run(query, source_name=rel.source_name, target_name=rel.target_name, props=rel.properties)
        print("--- Loading Complete ---")

    def _create_indexes(self):
        """Creates indexes in Neo4j for faster queries."""
        print("\n--- Creating Database Indexes ---")
        with self.driver.session(database="neo4j") as session:
            for entity_type in self.ontology.keys():
                query = f"CREATE INDEX IF NOT EXISTS FOR (n:{entity_type}) ON (n.name)"
                session.run(query)
            session.run("CREATE INDEX IF NOT EXISTS FOR (n:Specification) ON (n.name)")
            session.run("CREATE INDEX IF NOT EXISTS FOR (n:Message) ON (n.name)")
        print("Indexes created successfully.")

    def run_sample_queries(self):
        """Runs example queries to demonstrate the graph's utility."""
        print("\n--- Running Sample Queries ---")
        queries = {
            "What entities derive K_AMF?": """
                MATCH (e)-[:DERIVES]->(k:Key {name: 'KAMF'})
                RETURN e.name AS DerivingEntity, k.name AS DerivedKey
            """,
            "Which procedures are impacted by SUCI?": """
                MATCH (p:Procedure)<-[:PART_OF]-(e)-[:CONTAINS]->(param {name: 'SUCI'})
                RETURN DISTINCT p.name AS Procedure, p.source_clause AS Clause
            """,
            "Show entities involved in the 'Authentication procedure'": """
                MATCH (e)-[:PART_OF]->(p:Procedure {name: 'Authentication procedure'})
                RETURN e.name AS Entity, labels(e)[0] AS Type, p.name AS Procedure
                LIMIT 25
            """
        }
        with self.driver.session(database="neo4j") as session:
            for question, query in queries.items():
                print(f"\n> {question}")
                result = session.run(query)
                for record in result:
                    print(dict(record))

# --- Main Execution ---
def main():
    """Main function to run the KG builder."""
    # Find all relevant docx files
    doc_files = [p for p in DOCS_PATH.glob('*_new.docx')]
    if not doc_files:
        print(f"Error: No '*_new.docx' files found in {DOCS_PATH}")
        return

    builder = None
    try:
        builder = KnowledgeGraphBuilder(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
        builder.build_graph_from_files(doc_files)
        builder.run_sample_queries()
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        print("Please ensure Neo4j is running and credentials in the script are correct.")
    finally:
        if builder:
            builder.close()

if __name__ == "__main__":
    main()
