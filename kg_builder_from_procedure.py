import os
import re
from pathlib import Path
from typing import List, Dict, Set, Tuple, NamedTuple
import docx
from neo4j import GraphDatabase
import spacy
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModelForSequenceClassification
from transformers import pipeline
import torch
import json
from collections import defaultdict, Counter
from tqdm import tqdm

# --- Configuration ---
NEO4J_URI = "bolt://localhost:7688"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "12345678"
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

class AIKnowledgeGraphBuilder:
    """
    Uses AI/ML models to automatically discover entities and relationships without manual ontology.
    """
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        
        # Initialize AI models
        print("Loading AI models...")
        self._setup_ai_models()
        
        # Dynamic ontology learned from documents
        self.discovered_entities = defaultdict(int)
        self.discovered_relationships = defaultdict(int)
        
        print("AI KnowledgeGraphBuilder initialized.")

    def _setup_ai_models(self):
        """Initialize pre-trained AI models for NER and relation extraction."""
        # Check GPU availability
        self.device = 0 if torch.cuda.is_available() else -1
        print(f"Using device: {'GPU' if self.device >= 0 else 'CPU'}")
        
        try:
            # NER model for technical documents with GPU support
            print("Loading NER pipeline...")
            self.ner_pipeline = pipeline(
                "ner", 
                model="dbmdz/bert-large-cased-finetuned-conll03-english",
                aggregation_strategy="simple",
                device=self.device  # Enable GPU if available
            )
            
            # For relationship extraction with GPU support
            print("Loading relation classification pipeline...")
            self.relation_pipeline = pipeline(
                "text-classification",
                model="cross-encoder/nli-deberta-v3-base",
                device=self.device  # Enable GPU if available
            )
            
            # Load spaCy with GPU support
            print("Loading spaCy model...")
            if torch.cuda.is_available():
                # Use GPU-enabled spaCy model
                spacy.require_gpu()
                self.nlp = spacy.load("en_core_web_trf")
                print("spaCy loaded with GPU support")
            else:
                self.nlp = spacy.load("en_core_web_sm")
                print("spaCy loaded with CPU (GPU not available)")
            
            print("AI models loaded successfully.")
            
        except Exception as e:
            print(f"Error loading AI models: {e}")
            print("Falling back to CPU-only models...")
            self.device = -1
            self.nlp = spacy.load("en_core_web_sm")
            self.ner_pipeline = None
            self.relation_pipeline = None

    def close(self):
        """Closes the Neo4j database connection."""
        if self.driver:
            self.driver.close()
            print("Neo4j connection closed.")

    def extract_text_from_docx(self, file_path: Path) -> List[Dict]:
        """Extracts text from DOCX with better section detection."""
        print(f"Extracting text from: {file_path.name}")
        try:
            doc = docx.Document(file_path)
            sections = []
            current_section = {"title": "Introduction", "text": "", "clause": "0"}
            
            # Add progress bar for paragraph processing
            for para in tqdm(doc.paragraphs, desc=f"Processing {file_path.name}", leave=False):
                text = para.text.strip()
                if not text:
                    continue
                
                # More precise section header detection
                is_header = False
                
                # Method 1: Check paragraph style (most reliable)
                if para.style.name.startswith('Heading'):
                    is_header = True
                
                # Method 2: Check for numbered sections (e.g., "4.2.1 Title")
                elif re.match(r'^\d+(\.\d+)*\s+[A-Z]', text):
                    is_header = True
                
                # Method 3: Check for appendix sections (e.g., "A.1 Title")
                elif re.match(r'^[A-Z]\.\d+(\.\d+)*\s+[A-Z]', text):
                    is_header = True
                
                # Method 4: Short text that's all uppercase AND doesn't end with punctuation
                # (but be more restrictive)
                elif (len(text) < 80 and 
                      text.isupper() and 
                      not text.endswith(('.', ':', ';', '!', '?')) and
                      len(text.split()) > 1 and  # Must have at least 2 words
                      len(text.split()) < 8):    # But not too many words
                    is_header = True
                
                # Method 5: Starts with a number/letter and has title case pattern
                elif (re.match(r'^(\d+(\.\d+)*|[A-Z](\.\d+)*)\s+[A-Z][a-z]', text) and
                      len(text.split()) < 12 and  # Reasonable title length
                      not text.endswith('.')):    # Titles don't usually end with periods
                    is_header = True
                
                # If we found a header and current section has content, save it
                if is_header and current_section["text"].strip():
                    sections.append(current_section.copy())
                    # Extract clause number (first part before space)
                    clause_match = re.match(r'^(\d+(\.\d+)*|[A-Z](\.\d+)*)', text)
                    clause = clause_match.group(1) if clause_match else "unknown"
                    
                    current_section = {
                        "title": text,
                        "text": "",
                        "clause": clause
                    }
                else:
                    # Add text to current section
                    current_section["text"] += " " + text
            
            # Don't forget the last section
            if current_section["text"].strip():
                sections.append(current_section)
            
            print(f"  -> Found {len(sections)} sections.")
            return sections
            
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return []

    def extract_entities_ai(self, text: str, section: Dict) -> List[Entity]:
        """Extract entities using AI models with GPU acceleration."""
        entities = []
        found_entities = set()
        
        # Method 1: Use transformer-based NER with GPU if available
        if self.ner_pipeline:
            try:
                # Process text in larger chunks when using GPU
                chunk_size = 1024 if self.device >= 0 else 512
                chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
                
                # Add progress bar for chunk processing
                for chunk in tqdm(chunks, desc="NER processing", leave=False):
                    if len(chunk.strip()) < 10:  # Skip very short chunks
                        continue
                        
                    # GPU-accelerated NER
                    with torch.cuda.device(self.device) if self.device >= 0 else torch.no_grad():
                        ner_results = self.ner_pipeline(chunk)
                    
                    for entity in ner_results:
                        entity_name = entity['word'].strip()
                        confidence = entity['score']
                        
                        if confidence > 0.8 and entity_name not in found_entities and len(entity_name) > 2:
                            entity_type = self._classify_entity_ai(entity_name, chunk)
                            if entity_type:
                                props = {
                                    'confidence': confidence,
                                    'source_clause': section.get('clause', 'unknown'),
                                    'source_section': section.get('title', 'unknown'),
                                    'extraction_device': 'GPU' if self.device >= 0 else 'CPU'
                                }
                                entities.append(Entity(name=entity_name, entity_type=entity_type, properties=props))
                                found_entities.add(entity_name)
                                
            except Exception as e:
                print(f"GPU NER pipeline error: {e}")
                print("Falling back to pattern-based extraction...")
        
        # Method 2: Known 5G Network Functions (precise list)
        known_5g_network_functions = {
            'AMF', 'SMF', 'UPF', 'AUSF', 'UDM', 'UDR', 'PCF', 'NSSF', 
            'NEF', 'NRF', 'SMSF', 'LMF', 'GMLC', 'SCP', 'SEPP', 'NWDAF',
            'CHF', 'BSF', 'UDSF', 'DCCF', 'UCMF', 'EASDF', 'SEAF'
        }
        
        # Extract known network functions with high precision
        for nf_name in known_5g_network_functions:
            if re.search(r'\b' + re.escape(nf_name) + r'\b', text, re.IGNORECASE):
                if nf_name not in found_entities:
                    props = {
                        'discovery_method': 'known_nf',
                        'source_clause': section.get('clause', 'unknown'),
                        'source_section': section.get('title', 'unknown')
                    }
                    entities.append(Entity(name=nf_name, entity_type='NetworkFunction', properties=props))
                    found_entities.add(nf_name)
                    self.discovered_entities['NetworkFunction'] += 1
        
        # Method 3: Pattern-based entity extraction for other types (flexible)
        technical_patterns = {
            'Key': [
                r'\bK_[A-Z_]+\b',
                r'\b[A-Z]+\*\b',  # XRES*, HXRES*
            ],
            'Parameter': [
                r'\b[A-Z]{3,6}\b(?=\s+(?:parameter|identifier|value))',
                r'\b(?:SUCI|SUPI|GUTI|IMSI|IMEI)\b',
            ],
            'Procedure': [
                r'\b\w+\s+procedure\b',
                r'\b\w+\s+authentication\b',
                r'\b\w+\s+registration\b',
            ],
            'Message': [
                r'\b\w+\s+(?:request|response|indication|confirm)\b',
            ],
            'Actor': [
                r'\b(?:UE|gNB|ng-eNB|USIM|ME|NG-RAN|DN|AF)\b',
            ]
        }
        
        # Add progress bar for pattern matching
        pattern_progress = tqdm(technical_patterns.items(), desc="Pattern matching", leave=False)
        for entity_type, patterns in pattern_progress:
            pattern_progress.set_postfix({"entity_type": entity_type})
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    entity_name = match.group().strip()
                    if (entity_name not in found_entities and 
                        len(entity_name) > 2 and
                        not entity_name.lower() in ['the', 'and', 'for', 'with', 'from']):  # Filter common words
                        
                        props = {
                            'discovery_method': 'pattern',
                            'source_clause': section.get('clause', 'unknown'),
                            'source_section': section.get('title', 'unknown')
                        }
                        entities.append(Entity(name=entity_name, entity_type=entity_type, properties=props))
                        found_entities.add(entity_name)
                        self.discovered_entities[entity_type] += 1
        
        # Method 4: spaCy NER as fallback (but not for NetworkFunction)
        doc = self.nlp(text)
        for ent in tqdm(doc.ents, desc="spaCy NER", leave=False):
            entity_name = ent.text.strip()
            if (entity_name not in found_entities and 
                len(entity_name) > 2 and
                not entity_name.lower() in ['the', 'and', 'for', 'with', 'from']):
                
                entity_type = self._map_spacy_label(ent.label_)
                # Skip NetworkFunction from spaCy - we only use known list for those
                if entity_type and entity_type != 'NetworkFunction':
                    props = {
                        'discovery_method': 'spacy',
                        'spacy_label': ent.label_,
                        'source_clause': section.get('clause', 'unknown'),
                        'source_section': section.get('title', 'unknown')
                    }
                    entities.append(Entity(name=entity_name, entity_type=entity_type, properties=props))
                    found_entities.add(entity_name)
        
        return entities

    def _classify_entity_ai(self, entity_name: str, context: str) -> str:
        """Use AI to classify entity type based on context with known NF list."""
        entity_upper = entity_name.upper()
        context_lower = context.lower()
        
        # Define known 5G network functions explicitly (same list as above)
        known_nfs = {'AMF', 'SMF', 'UPF', 'AUSF', 'UDM', 'UDR', 'PCF', 'NSSF', 
                     'NEF', 'NRF', 'SMSF', 'LMF', 'GMLC', 'SCP', 'SEPP', 'NWDAF',
                     'CHF', 'BSF', 'UDSF', 'DCCF', 'UCMF', 'EASDF', 'SEAF'}
        
        # Only classify as NetworkFunction if it's in our known list
        if entity_upper in known_nfs:
            return 'NetworkFunction'
        elif entity_name.startswith('K_') or entity_name.endswith('*'):
            return 'Key'
        elif 'procedure' in context_lower:
            return 'Procedure'
        elif any(word in context_lower for word in ['parameter', 'identifier', 'value']):
            return 'Parameter'
        elif any(word in context_lower for word in ['request', 'response', 'message']):
            return 'Message'
        elif entity_name.upper() in ['UE', 'GNB', 'NG-ENB', 'USIM', 'ME', 'NG-RAN', 'DN', 'AF']:
            return 'Actor'
        else:
            # For other entities, use a generic type or let AI decide
            return 'Entity'

    def _map_spacy_label(self, spacy_label: str) -> str:
        """Map spaCy entity labels to our domain (excluding NetworkFunction)."""
        mapping = {
            'ORG': 'Actor',
            'PRODUCT': 'Entity',  # Don't map to NetworkFunction anymore
            'EVENT': 'Procedure',
            'PERSON': None,  # Ignore persons in technical docs
            'GPE': None,     # Ignore geographical entities
        }
        return mapping.get(spacy_label, 'Entity')

    def extract_relationships_ai(self, text: str, entities: List[Entity], section: Dict) -> List[Relationship]:
        """Extract relationships using AI and pattern discovery."""
        relationships = []
        entity_names = {e.name for e in entities}
        
        if len(entity_names) < 2:
            return relationships
        
        print(f"  -> Extracting relationships from {len(entity_names)} entities")
        
        # Method 1: Dependency parsing for verb-based relationships
        doc = self.nlp(text)
        relationships.extend(self._extract_verb_relationships(doc, entity_names))
        
        # Method 2: Co-occurrence based relationships
        relationships.extend(self._extract_cooccurrence_relationships(text, entities))
        
        # Method 3: Template-based extraction
        relationships.extend(self._extract_template_relationships(text, entity_names))
        
        return relationships

    def _extract_verb_relationships(self, doc, entity_names: Set[str]) -> List[Relationship]:
        """Extract relationships based on verbs and dependency structure."""
        relationships = []
        
        # Define verb categories and their corresponding relationship types
        verb_categories = {
            'communication': (['send', 'transmit', 'receive', 'forward'], 'COMMUNICATES_WITH'),
            'derivation': (['derive', 'generate', 'compute', 'calculate'], 'DERIVES'),
            'containment': (['contain', 'include', 'comprise'], 'CONTAINS'),
            'participation': (['participate', 'involve', 'engage'], 'PARTICIPATES_IN'),
            'authentication': (['authenticate', 'verify', 'validate'], 'AUTHENTICATES'),
        }
        
        # Add progress bar for token processing
        for token in tqdm(doc, desc="Verb analysis", leave=False):
            if token.pos_ == 'VERB':
                for category, (verbs, rel_type) in verb_categories.items():
                    if token.lemma_ in verbs:
                        # Find subject and object
                        subject = None
                        obj = None
                        
                        for child in token.children:
                            if child.dep_ in ['nsubj', 'nsubjpass'] and child.text in entity_names:
                                subject = child.text
                            elif child.dep_ in ['dobj', 'pobj', 'attr'] and child.text in entity_names:
                                obj = child.text
                        
                        if subject and obj and subject != obj:
                            relationships.append(Relationship(subject, obj, rel_type, {
                                'verb': token.lemma_,
                                'confidence': 0.8
                            }))
                            self.discovered_relationships[rel_type] += 1
        
        return relationships

    def _extract_cooccurrence_relationships(self, text: str, entities: List[Entity]) -> List[Relationship]:
        """Extract relationships based on entity co-occurrence in sentences."""
        relationships = []
        sentences = re.split(r'[.!?]', text)
        
        # Add progress bar for sentence processing
        for sentence in tqdm(sentences, desc="Co-occurrence analysis", leave=False):
            sentence_entities = [e for e in entities if e.name in sentence]
            
            if len(sentence_entities) >= 2:
                # Create relationships between co-occurring entities
                for i, entity1 in enumerate(sentence_entities):
                    for entity2 in sentence_entities[i+1:]:
                        rel_type = self._infer_relationship_type(entity1, entity2, sentence)
                        if rel_type:
                            relationships.append(Relationship(
                                entity1.name, 
                                entity2.name, 
                                rel_type, 
                                {'confidence': 0.6, 'method': 'cooccurrence'}
                            ))
        
        return relationships

    def _infer_relationship_type(self, entity1: Entity, entity2: Entity, context: str) -> str:
        """Infer relationship type based on entity types and context."""
        type1, type2 = entity1.entity_type, entity2.entity_type
        context_lower = context.lower()
        
        # Define rules based on entity type combinations
        if type1 == 'Procedure' or type2 == 'Procedure':
            return 'PART_OF'
        elif (type1 in ['Key', 'Parameter'] and type2 in ['NetworkFunction', 'Actor']) or \
             (type2 in ['Key', 'Parameter'] and type1 in ['NetworkFunction', 'Actor']):
            if any(word in context_lower for word in ['derive', 'generate', 'compute']):
                return 'DERIVES'
            else:
                return 'USES'
        elif type1 == 'Message' or type2 == 'Message':
            if any(word in context_lower for word in ['send', 'transmit', 'receive']):
                return 'SENDS'
            else:
                return 'CONTAINS'
        elif type1 in ['NetworkFunction', 'Actor'] and type2 in ['NetworkFunction', 'Actor']:
            return 'INTERACTS_WITH'
        else:
            return 'RELATED_TO'

    def _extract_template_relationships(self, text: str, entity_names: Set[str]) -> List[Relationship]:
        """Extract relationships using flexible templates."""
        relationships = []
        
        # Dynamic templates that adapt to the text
        templates = [
            (r'(\w+)\s+(?:sends?|transmits?)\s+.*?\s+to\s+(\w+)', 'SENDS_TO'),
            (r'(\w+)\s+(?:derives?|generates?)\s+(\w+)', 'DERIVES'),
            (r'(\w+)\s+(?:contains?|includes?)\s+(\w+)', 'CONTAINS'),
            (r'(\w+)\s+is\s+part\s+of\s+(\w+)', 'PART_OF'),
            (r'(\w+)\s+(?:authenticates?|verifies?)\s+(\w+)', 'AUTHENTICATES'),
        ]
        
        # Add progress bar for template matching
        for pattern, rel_type in tqdm(templates, desc="Template matching", leave=False):
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                entity1, entity2 = match.groups()
                if entity1 in entity_names and entity2 in entity_names:
                    relationships.append(Relationship(entity1, entity2, rel_type, {
                        'method': 'template',
                        'confidence': 0.7
                    }))
        
        return relationships

    def extract_from_section(self, section: Dict) -> Tuple[List[Entity], List[Relationship]]:
        """Main extraction function using AI techniques."""
        text = section['text']
        
        if len(text) < 50:  # Skip very short sections
            return [], []
        
        # Extract entities using AI
        entities = self.extract_entities_ai(text, section)
        
        # Extract relationships using AI
        relationships = self.extract_relationships_ai(text, entities, section)
        
        return entities, relationships

    def build_graph_from_files(self, file_paths: List[Path]):
        """Build knowledge graph using AI-driven discovery."""
        print("\n--- Starting AI-based Knowledge Graph Construction ---")
        all_entities = {}
        all_relationships = []

        # Add progress bar for file processing
        for file_path in tqdm(file_paths, desc="Processing documents"):
            print(f"\nProcessing {file_path.name}...")
            
            sections = self.extract_text_from_docx(file_path)
            
            # Add progress bar for section processing
            section_progress = tqdm(sections, desc=f"Sections in {file_path.name}", leave=False)
            for section in section_progress:
                section_progress.set_postfix({"clause": section.get('clause', 'unknown')})
                
                entities, relationships = self.extract_from_section(section)
                
                for entity in entities:
                    key = (entity.name, entity.entity_type)
                    if key not in all_entities:
                        all_entities[key] = entity
                
                all_relationships.extend(relationships)
        
        print(f"\n--- Extraction Complete ---")
        print(f"Total unique entities found: {len(all_entities)}")
        print(f"Total relationships found: {len(all_relationships)}")
        
        self._print_discovery_stats()
        self._load_to_neo4j(list(all_entities.values()), all_relationships)
        self._create_indexes()

    def _print_discovery_stats(self):
        """Print statistics about discovered entities and relationships."""
        print("\n--- Discovery Statistics ---")
        print("Entity types discovered:")
        for entity_type, count in sorted(self.discovered_entities.items()):
            print(f"  {entity_type}: {count}")
        
        print("\nRelationship types discovered:")
        for rel_type, count in sorted(self.discovered_relationships.items()):
            print(f"  {rel_type}: {count}")

    def _load_to_neo4j(self, entities: List[Entity], relationships: List[Relationship]):
        """Load data into Neo4j with enhanced properties."""
        print("\n--- Loading Data into Neo4j ---")
        with self.driver.session(database="neo4j") as session:
            print("Clearing existing database...")
            session.run("MATCH (n) DETACH DELETE n")

            print(f"Creating {len(entities)} nodes...")
            # Add progress bar for entity creation
            for entity in tqdm(entities, desc="Creating nodes"):
                query = f"""
                MERGE (n:{entity.entity_type} {{name: $name}})
                SET n += $props
                """
                session.run(query, name=entity.name, props=entity.properties)

            print(f"Creating {len(relationships)} relationships...")
            # Add progress bar for relationship creation
            for rel in tqdm(relationships, desc="Creating relationships"):
                query = f"""
                MATCH (a {{name: $source_name}}), (b {{name: $target_name}})
                MERGE (a)-[r:{rel.rel_type}]->(b)
                SET r += $props
                """
                session.run(query, 
                           source_name=rel.source_name, 
                           target_name=rel.target_name, 
                           props=rel.properties)
        
        print("--- Loading Complete ---")

    def _create_indexes(self):
        """Create dynamic indexes based on discovered entity types."""
        print("\n--- Creating Dynamic Indexes ---")
        with self.driver.session(database="neo4j") as session:
            # Add progress bar for index creation
            entity_types = list(self.discovered_entities.keys())
            for entity_type in tqdm(entity_types, desc="Creating indexes"):
                query = f"CREATE INDEX IF NOT EXISTS FOR (n:{entity_type}) ON (n.name)"
                session.run(query)
        print("Indexes created successfully.")

    def run_discovery_queries(self):
        """Run queries to explore the discovered knowledge graph."""
        print("\n--- Exploring Discovered Knowledge Graph ---")
        
        queries = [
            ("Most connected entities", """
                MATCH (n)
                RETURN n.name as Entity, labels(n)[0] as Type, 
                       size((n)--()) as Connections
                ORDER BY Connections DESC
                LIMIT 10
            """),
            ("Relationship type distribution", """
                MATCH ()-[r]->()
                RETURN type(r) as RelationType, count(r) as Count
                ORDER BY Count DESC
            """),
            ("Entity type distribution", """
                MATCH (n)
                RETURN labels(n)[0] as EntityType, count(n) as Count
                ORDER BY Count DESC
            """),
        ]
        
        with self.driver.session(database="neo4j") as session:
            for title, query in tqdm(queries, desc="Running discovery queries"):
                print(f"\n> {title}")
                result = session.run(query)
                for record in result:
                    print(dict(record))

# --- Main Execution ---
def main():
    """Main function to run the AI-based KG builder."""
    # Check GPU availability
    if torch.cuda.is_available():
        print(f"GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"VRAM available: {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB")
    else:
        print("No GPU detected, using CPU")
    
    doc_files = [p for p in DOCS_PATH.glob('*_new.docx')]
    if not doc_files:
        print(f"Error: No '*_new.docx' files found in {DOCS_PATH}")
        return

    builder = None
    try:
        builder = AIKnowledgeGraphBuilder(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
        builder.build_graph_from_files(doc_files)
        builder.run_discovery_queries()
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if builder:
            builder.close()

if __name__ == "__main__":
    main()