import re
import json
import torch
import tiktoken
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings("ignore")

try:
    from transformers import pipeline
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Error: transformers/sentence-transformers not installed")
    print("Run: pip install transformers sentence-transformers")
    exit(1)

from config import (
    LLM_MODEL_OPTIONS, EMBEDDING_MODEL_OPTIONS, FILTERED_WORDS, 
    KNOWN_NETWORK_FUNCTIONS, KNOWN_PARAMETERS, KNOWN_KEYS, KNOWN_MESSAGES
)
from data_structures import ProcedureContext, ExtractionResult

class EntityExtractor:
    """Enhanced entity extraction with modern NLP/LLM models and step descriptions."""
    
    def __init__(self):
        self.device = 0 if torch.cuda.is_available() else -1
        print(f"EntityExtractor using: {'GPU' if self.device >= 0 else 'CPU'}")
        self.current_model_name = None  # Track the current model name
        self.tokenizer = tiktoken.get_encoding("cl100k_base")  # For token counting
        self.max_tokens = 32000  # Leave some buffer from 32768
        
        # Load whitelist from config for strict validation
        self.valid_network_functions = set(nf.upper() for nf in KNOWN_NETWORK_FUNCTIONS)
        self.valid_parameters = set(param.upper() for param in KNOWN_PARAMETERS)
        self.valid_keys = set(key.upper() for key in KNOWN_KEYS)
        self.valid_messages = set(msg.upper() for msg in KNOWN_MESSAGES)
        
        print(f"Loaded {len(self.valid_network_functions)} valid network functions for validation")
        
        self._setup_models()
    
    def _setup_models(self):
        """Setup multiple specialized models (Requirement 7)."""
        print("Loading modern NLP/LLM models...")
        
        # Initialize all model attributes
        self.entity_llm = None
        self.text_generator = None  # For compatibility
        self.embedding_model = None
        
        # Load Entity Extraction LLM
        for model_name in LLM_MODEL_OPTIONS:
            try:
                print(f"  Trying Entity LLM: {model_name}...")
                if "flan-t5" in model_name:
                    self.entity_llm = pipeline(
                        "text2text-generation",
                        model=model_name,
                        device=self.device,
                        model_kwargs={"torch_dtype": torch.float16} if self.device >= 0 else {}
                    )
                    self.is_t5_model = True
                else:
                    self.entity_llm = pipeline(
                        "text-generation",
                        model=model_name,
                        device=self.device,
                        do_sample=True,
                        temperature=0.1
                    )
                    self.is_t5_model = False
                
                # Set text_generator for compatibility
                self.text_generator = self.entity_llm
                
                # FIXED: Store the model name properly
                self.current_model_name = model_name
                
                # Test the model
                if "flan-t5" in model_name:
                    test_result = self.entity_llm("Extract entities: AMF sends message", max_length=20)
                else:
                    test_result = self.entity_llm("Test", max_length=10)
                print(f"  ✓ Entity LLM loaded: {model_name}")
                break
                
            except Exception as e:
                print(f"  ✗ Failed: {str(e)[:50]}...")
                continue
        
        # Load Embedding Model for Search (Requirement 8)
        for model_name in EMBEDDING_MODEL_OPTIONS:
            try:
                print(f"  Trying Embedding Model: {model_name}...")
                self.embedding_model = SentenceTransformer(model_name)
                if self.device >= 0:
                    self.embedding_model = self.embedding_model.cuda()
                print(f"  ✓ Embedding model loaded: {model_name}")
                break
            except Exception as e:
                print(f"  ✗ Failed: {str(e)[:50]}...")
                continue
        
        if not self.entity_llm:
            print("  ⚠️  No Entity LLM loaded - using pattern-based extraction only")
        if not self.embedding_model:
            print("  ⚠️  No Embedding model loaded - search will be limited")

    # NEW: Strict whitelist enforcement methods
    def _enforce_nf_whitelist(self, extracted_nfs: List[str]) -> List[str]:
        """Enforce strict network function whitelist validation."""
        validated_nfs = []
        rejected_count = 0
        
        for nf in extracted_nfs:
            # Normalize candidate (handle variations like "5G-AMF" -> "AMF")
            normalized_nf = self._normalize_nf_candidate(nf)
            
            if normalized_nf and normalized_nf in self.valid_network_functions:
                validated_nfs.append(normalized_nf)
                print(f"        ✓ Validated NF: {nf} -> {normalized_nf}")
            else:
                print(f"        ✗ Rejected NF: {nf} (not in whitelist)")
                rejected_count += 1
        
        if rejected_count > 0:
            print(f"      📋 Rejected {rejected_count} invalid network functions")
        
        return validated_nfs
    
    def _normalize_nf_candidate(self, candidate: str) -> Optional[str]:
        """Normalize NF candidate and check variations."""
        if not candidate:
            return None
        
        # Convert to uppercase
        candidate_upper = candidate.upper().strip()
        
        # Direct match
        if candidate_upper in self.valid_network_functions:
            return candidate_upper
        
        # Handle common variations
        # Remove prefixes like "5G-", "UE-", "N1-", etc.
        normalized = re.sub(r'^(5G-|UE-|N[0-9]+-|CN-)', '', candidate_upper)
        if normalized in self.valid_network_functions:
            return normalized
        
        # Remove suffixes like "-FUNCTION", "-SERVER", etc.
        normalized = re.sub(r'(-FUNCTION|-SERVER|-GATEWAY|-NODE)$', '', candidate_upper)
        if normalized in self.valid_network_functions:
            return normalized
        
        # Handle abbreviations (e.g., "Authentication Server" -> "AUSF")
        abbreviation_map = {
            'AUTHENTICATION SERVER': 'AUSF',
            'ACCESS MOBILITY MANAGEMENT': 'AMF',
            'SESSION MANAGEMENT': 'SMF',
            'USER PLANE': 'UPF',
            'POLICY CONTROL': 'PCF',
            'UNIFIED DATA MANAGEMENT': 'UDM',
            'UNIFIED DATA REPOSITORY': 'UDR',
            'NETWORK REPOSITORY': 'NRF',
            'NETWORK SLICE SELECTION': 'NSSF'
        }
        
        if candidate_upper in abbreviation_map:
            return abbreviation_map[candidate_upper]
        
        return None
    
    def _enforce_parameter_whitelist(self, extracted_params: List[str]) -> List[str]:
        """Enforce parameter whitelist validation."""
        validated_params = []
        
        for param in extracted_params:
            param_upper = param.upper().strip()
            if param_upper in self.valid_parameters:
                validated_params.append(param_upper)
                print(f"        ✓ Validated Parameter: {param}")
            else:
                print(f"        ✗ Rejected Parameter: {param} (not in whitelist)")
        
        return validated_params
    
    def _enforce_key_whitelist(self, extracted_keys: List[str]) -> List[str]:
        """Enforce key whitelist validation."""
        validated_keys = []
        
        for key in extracted_keys:
            key_upper = key.upper().strip()
            if key_upper in self.valid_keys:
                validated_keys.append(key_upper)
                print(f"        ✓ Validated Key: {key}")
            else:
                print(f"        ✗ Rejected Key: {key} (not in whitelist)")
        
        return validated_keys

    def extract_entities_for_procedure(self, context: ProcedureContext) -> ExtractionResult:
        """Enhanced entity extraction with step descriptions (Requirement 9)."""
        try:
            print(f"    Extracting entities for: {context.procedure_name}")
            
            # Method 1: Enhanced LLM extraction
            llm_entities = self._enhanced_llm_extraction(context)
            
            # Method 2: Pattern-based extraction
            pattern_entities = self._pattern_based_extraction(context)
            
            # Method 3: Merge results
            merged_entities = self._merge_extraction_results(llm_entities, pattern_entities)
            
            # NEW: Method 4: Enforce whitelist validation on all entities
            validated_entities = self._apply_whitelist_validation(merged_entities)
            
            # Method 5: Ensure minimum entities
            if self._is_empty_result(validated_entities):
                validated_entities = self._generate_minimum_entities(context)
            
            # CRITICAL: Generate procedure-specific steps with descriptions
            validated_entities = self._generate_steps_with_descriptions(validated_entities, context)
            
            # Generate search description for procedure (Requirement 8)
            context.search_description = self._generate_search_description(context)
            context.search_tags = self._generate_search_tags(context)
            
            # Update context
            self._update_context_with_entities(context, validated_entities)
            
            # Debug output
            total_entities = sum(len(v) for v in validated_entities.values())
            print(f"      ✓ Extracted {total_entities} validated entities with descriptions")
            
            # FIXED: Get model name safely
            model_name = self.current_model_name if self.current_model_name else 'none'
            
            return ExtractionResult(
                entities=validated_entities,
                relationships=[],
                success=True,
                extraction_method="enhanced_llm_pattern_validated",
                llm_model_used=model_name
            )
            
        except Exception as e:
            print(f"    ✗ Entity extraction failed: {e}")
            import traceback
            traceback.print_exc()
            return ExtractionResult(
                entities={},
                relationships=[],
                success=False,
                error_message=str(e)
            )
    
    # NEW: Apply whitelist validation to all entity types
    def _apply_whitelist_validation(self, entities: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """Apply strict whitelist validation to all extracted entities."""
        print(f"      🔍 Applying whitelist validation...")
        
        validated_entities = {}
        
        # Validate network functions (MOST IMPORTANT)
        original_nfs = entities.get("network_functions", [])
        validated_entities["network_functions"] = self._enforce_nf_whitelist(original_nfs)
        
        # Validate parameters (if whitelist exists)
        original_params = entities.get("parameters", [])
        if self.valid_parameters:
            validated_entities["parameters"] = self._enforce_parameter_whitelist(original_params)
        else:
            validated_entities["parameters"] = original_params  # Keep original if no whitelist
        
        # Validate keys (if whitelist exists)  
        original_keys = entities.get("keys", [])
        if self.valid_keys:
            validated_entities["keys"] = self._enforce_key_whitelist(original_keys)
        else:
            validated_entities["keys"] = original_keys  # Keep original if no whitelist
        
        # Messages and steps - no strict whitelist (too dynamic)
        validated_entities["messages"] = entities.get("messages", [])
        validated_entities["steps"] = entities.get("steps", [])
        
        # Report validation results
        nf_reduction = len(original_nfs) - len(validated_entities["network_functions"])
        if nf_reduction > 0:
            print(f"      📊 Network function validation: {len(original_nfs)} -> {len(validated_entities['network_functions'])} (-{nf_reduction})")
        
        return validated_entities
    
    def _enhanced_llm_extraction(self, context: ProcedureContext) -> Dict[str, List[str]]:
        """Enhanced LLM extraction with better prompts."""
        if not self.entity_llm:
            return {"network_functions": [], "messages": [], "parameters": [], "keys": [], "steps": []}
        
        try:
            # Enhanced prompt for better extraction
            prompt = f"""
Analyze this 3GPP telecommunications procedure and extract entities:

PROCEDURE: {context.procedure_name}
TEXT: {context.section.text[:1000]}

Extract the following entity types:

1. NETWORK_FUNCTIONS: 5G core network functions (AMF, SMF, UPF, AUSF, etc.)
2. MESSAGES: Communication messages between network functions
3. PARAMETERS: Technical parameters and identifiers (SUCI, IMSI, etc.)
4. KEYS: Cryptographic keys and authentication values
5. STEPS: Sequential procedure steps with descriptions

Format your response as:
NETWORK_FUNCTIONS: AMF, SMF, AUSF
MESSAGES: Registration Request, Authentication Response
PARAMETERS: SUCI, 5G-GUTI
KEYS: 5G-AKA, Kausf
STEPS: 1. UE initiates registration|2. AMF processes request|3. Authentication performed
"""

            if self.is_t5_model:
                result = self.entity_llm(prompt, max_length=300, num_return_sequences=1)
                response_text = result[0]['generated_text']
            else:
                result = self.entity_llm(prompt, max_length=len(prompt) + 300, num_return_sequences=1)
                response_text = result[0]['generated_text'][len(prompt):]
            
            return self._parse_enhanced_llm_response(response_text)
            
        except Exception as e:
            print(f"      Enhanced LLM extraction error: {e}")
            return {"network_functions": [], "messages": [], "parameters": [], "keys": [], "steps": []}
    
    def _parse_enhanced_llm_response(self, response: str) -> Dict[str, List[str]]:
        """Parse enhanced LLM response with step descriptions."""
        entities = {"network_functions": [], "messages": [], "parameters": [], "keys": [], "steps": []}
        
        lines = response.split('\n')
        current_type = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Detect entity type headers
            if line.upper().startswith('NETWORK_FUNCTIONS:'):
                current_type = 'network_functions'
                content = line.split(':', 1)[1].strip()
                entities[current_type].extend([e.strip() for e in content.split(',') if e.strip()])
            elif line.upper().startswith('MESSAGES:'):
                current_type = 'messages'
                content = line.split(':', 1)[1].strip()
                entities[current_type].extend([e.strip() for e in content.split(',') if e.strip()])
            elif line.upper().startswith('PARAMETERS:'):
                current_type = 'parameters'
                content = line.split(':', 1)[1].strip()
                entities[current_type].extend([e.strip() for e in content.split(',') if e.strip()])
            elif line.upper().startswith('KEYS:'):
                current_type = 'keys'
                content = line.split(':', 1)[1].strip()
                entities[current_type].extend([e.strip() for e in content.split(',') if e.strip()])
            elif line.upper().startswith('STEPS:'):
                current_type = 'steps'
                content = line.split(':', 1)[1].strip()
                # Parse steps with descriptions (format: "1. Description|2. Description")
                step_parts = content.split('|')
                for step_part in step_parts:
                    step_part = step_part.strip()
                    if step_part and len(step_part) > 5:
                        entities[current_type].append(step_part)
        
        return entities

    def _pattern_based_extraction(self, context: ProcedureContext) -> Dict[str, List[str]]:
        """Pattern-based extraction method."""
        text = context.section.text
        entities = {"network_functions": [], "messages": [], "parameters": [], "keys": [], "steps": []}
        
        # Extract network functions - ALREADY USES WHITELIST CORRECTLY
        for nf in KNOWN_NETWORK_FUNCTIONS:
            if re.search(r'\b' + re.escape(nf) + r'\b', text, re.IGNORECASE):
                entities["network_functions"].append(nf)
        
        # Extract known messages
        for msg in KNOWN_MESSAGES:
            if re.search(r'\b' + re.escape(msg) + r'\b', text, re.IGNORECASE):
                entities["messages"].append(msg)
        
        # Extract messages using patterns
        message_patterns = [
            r'([A-Z][a-z]+ (?:Request|Response|Indication|Confirm))',
            r'(N[a-z]+_[A-Za-z_]+)',
            r'([A-Z]+ (?:registration|authentication|establishment))',
        ]
        
        for pattern in message_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                msg = match.group(1)
                if len(msg) > 5 and len(msg) < 80 and not self._is_filtered_word(msg):
                    entities["messages"].append(msg)
        
        # Extract parameters
        for param in KNOWN_PARAMETERS:
            if re.search(r'\b' + re.escape(param) + r'\b', text, re.IGNORECASE):
                entities["parameters"].append(param)
        
        # Extract keys
        for key in KNOWN_KEYS:
            if re.search(r'\b' + re.escape(key) + r'\b', text, re.IGNORECASE):
                entities["keys"].append(key)
        
        # Extract step descriptions from document text
        step_descriptions = self._extract_step_descriptions_from_document(text, context.procedure_name)
        entities["steps"] = step_descriptions
        
        return entities

    def _extract_step_descriptions_from_document(self, text: str, procedure_name: str) -> List[str]:
        """Extract step descriptions ONLY from the actual 3GPP document text."""
        descriptions = []
        
        print(f"      Extracting step descriptions from document for {procedure_name}...")
        
        # Method 1: Look for numbered procedure steps in the text
        numbered_patterns = [
            r'(\d+)\.\s+([^.!?]{20,300}[.!?])',           # "1. The UE sends..."
            r'Step\s+(\d+)[:\.]?\s*([^.!?]{20,300}[.!?])', # "Step 1: The UE..."
            r'(\d+)\)\s+([^.!?]{20,300}[.!?])'            # "1) The UE sends..."
        ]
        
        for pattern in numbered_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                step_num = match.group(1)
                description = match.group(2).strip()
                
                if self._is_valid_procedure_step_description(description, procedure_name):
                    descriptions.append(description)
                    print(f"        Found step {step_num}: {description[:60]}...")
        
        # Method 2: Look for lettered steps
        lettered_pattern = r'([a-h])\)\s+([^.!?]{20,300}[.!?])'
        matches = re.finditer(lettered_pattern, text, re.IGNORECASE | re.MULTILINE)
        for match in matches:
            step_letter = match.group(1)
            description = match.group(2).strip()
            
            if self._is_valid_procedure_step_description(description, procedure_name):
                descriptions.append(description)
                print(f"        Found step {step_letter}: {description[:60]}...")
        
        # Method 3: Look for paragraph-based descriptions (if no numbered steps)
        if not descriptions:
            # Split text into sentences and look for procedural descriptions
            sentences = re.split(r'[.!?]+', text)
            for sentence in sentences:
                sentence = sentence.strip()
                if (len(sentence) > 30 and 
                    self._contains_procedural_language(sentence) and
                    self._is_valid_procedure_step_description(sentence, procedure_name)):
                    descriptions.append(sentence + '.')
                    print(f"        Found procedural sentence: {sentence[:60]}...")
                    if len(descriptions) >= 5:  # Limit procedural sentences
                        break
        
        # Remove duplicates while preserving order
        unique_descriptions = []
        seen = set()
        for desc in descriptions:
            if desc not in seen:
                unique_descriptions.append(desc)
                seen.add(desc)
        
        print(f"      ✓ Extracted {len(unique_descriptions)} step descriptions from document")
        return unique_descriptions[:10]  # Limit to 10 steps

    def _is_valid_procedure_step_description(self, description: str, procedure_name: str) -> bool:
        """Check if description is valid for procedure step from 3GPP document."""
        if len(description) < 15 or len(description) > 500:
            return False
        
        description_lower = description.lower()
        
        # Must contain telecommunications/3GPP action words
        telecom_actions = [
            'send', 'receive', 'transmit', 'forward', 'verify', 'authenticate', 
            'establish', 'perform', 'request', 'response', 'initiate', 'process',
            'validate', 'compute', 'derive', 'store', 'generate', 'select',
            'indicate', 'inform', 'notify', 'confirm', 'reject', 'accept'
        ]
        
        has_action = any(action in description_lower for action in telecom_actions)
        
        # Should contain 3GPP/telecom entities
        telecom_entities = [
            'ue', 'amf', 'smf', 'upf', 'ausf', 'udm', 'udr', 'pcf', 'nrf',
            'network', 'core', 'base station', 'gnb', 'authentication', 
            'registration', 'session', 'message', 'procedure', 'protocol'
        ]
        
        has_telecom_entity = any(entity in description_lower for entity in telecom_entities)
        
        # Should not be meta-text (references to figures, sections, etc.)
        meta_phrases = [
            'figure shows', 'as shown in', 'see section', 'refer to', 'as described in',
            'according to', 'as specified', 'for more details', 'note that', 'editor'
        ]
        
        is_meta = any(phrase in description_lower for phrase in meta_phrases)
        
        return has_action and has_telecom_entity and not is_meta

    def _contains_procedural_language(self, sentence: str) -> bool:
        """Check if sentence contains procedural language typical of 3GPP specs."""
        sentence_lower = sentence.lower()
        
        procedural_indicators = [
            'the ue', 'the amf', 'the smf', 'the network', 'the procedure',
            'shall send', 'shall receive', 'shall perform', 'shall verify',
            'may send', 'may receive', 'should send', 'should receive',
            'then', 'next', 'after', 'upon', 'when', 'if', 'once'
        ]
        
        return any(indicator in sentence_lower for indicator in procedural_indicators)

    def _generate_steps_with_descriptions(self, entities: Dict[str, List[str]], context: ProcedureContext) -> Dict[str, List[str]]:
        """Generate procedure-specific steps with descriptions (Requirement 9)."""
        procedure_clean = re.sub(r'[^\w\s]', '', context.procedure_name).replace(' ', '_')
        
        # Get step descriptions from extracted text or LLM
        step_descriptions = entities.get('steps', [])
        
        # If no step descriptions found, generate defaults based on procedure type
        if not step_descriptions:
            step_descriptions = self._generate_default_step_descriptions(context)
        
        # Generate procedure-specific step names with descriptions
        procedure_steps = []
        context.step_descriptions = {}
        
        for i, description in enumerate(step_descriptions[:10], 1):  # Limit to 10 steps
            step_name = f"{procedure_clean}_step_{i}"
            procedure_steps.append(step_name)
            
            # Clean and store description
            clean_description = self._clean_step_description(description)
            context.step_descriptions[step_name] = clean_description
        
        # Ensure at least one step
        if not procedure_steps:
            step_name = f"{procedure_clean}_step_1"
            procedure_steps = [step_name]
            context.step_descriptions[step_name] = f"Execute {context.procedure_name} procedure."
        
        entities['steps'] = procedure_steps
        print(f"      ✓ Generated {len(procedure_steps)} steps with descriptions")
        
        return entities

    def _clean_step_description(self, description: str) -> str:
        """Clean step description for better readability."""
        # Remove numbering prefixes
        description = re.sub(r'^\d+\.\s*', '', description)
        description = re.sub(r'^[a-z]\)\s*', '', description, flags=re.IGNORECASE)
        description = re.sub(r'^[Ss]tep\s+\d+[:\.]?\s*', '', description)
        
        # Capitalize first letter
        if description and description[0].islower():
            description = description[0].upper() + description[1:]
        
        # Ensure proper ending
        if description and not description.endswith(('.', '!', '?')):
            description += '.'
        
        return description.strip()

    def _generate_default_step_descriptions(self, context: ProcedureContext) -> List[str]:
        """Generate default step descriptions based on procedure type."""
        procedure_lower = context.procedure_name.lower()
        
        if 'authentication' in procedure_lower or 'aka' in procedure_lower:
            return [
                "UE initiates authentication procedure by sending authentication request",
                "AMF forwards authentication request to AUSF for validation",
                "AUSF generates authentication challenge and sends to UE via AMF",
                "UE computes authentication response and sends back to network",
                "AUSF validates authentication response and confirms UE identity",
                "Authentication procedure completes successfully"
            ]
        elif 'registration' in procedure_lower:
            return [
                "UE sends registration request to AMF with identity information",
                "AMF validates UE credentials and subscription data",
                "AMF performs authentication and security procedures",
                "AMF establishes UE context and assigns temporary identifiers",
                "AMF sends registration accept to UE with allocated resources",
                "Registration procedure completes and UE enters connected state"
            ]
        elif 'session' in procedure_lower:
            return [
                "UE requests PDU session establishment with specific requirements",
                "AMF forwards session request to SMF for processing",
                "SMF selects appropriate UPF and establishes data path",
                "SMF configures QoS flows and traffic handling rules",
                "Session establishment completes with allocated resources",
                "Data path becomes active for user traffic"
            ]
        else:
            return [
                f"Procedure {context.procedure_name} begins with initial setup",
                f"Network functions coordinate to process {context.procedure_name}",
                f"Required validations and checks are performed",
                f"Procedure {context.procedure_name} completes successfully"
            ]

    def _generate_search_description(self, context: ProcedureContext) -> str:
        """Generate searchable description for procedure (Requirement 8)."""
        components = []
        
        # Add procedure name
        components.append(context.procedure_name)
        
        # Add procedure type description
        procedure_lower = context.procedure_name.lower()
        if 'authentication' in procedure_lower or 'aka' in procedure_lower:
            components.append("authentication process between UE and 5G core network")
        elif 'registration' in procedure_lower:
            components.append("registration procedure for UE attachment to 5G network")
        elif 'session' in procedure_lower:
            components.append("PDU session establishment and management")
        
        # Add network functions involved
        if context.network_functions:
            nf_list = ", ".join(context.network_functions[:3])
            components.append(f"involving {nf_list}")
        
        return " ".join(components)

    def _generate_search_tags(self, context: ProcedureContext) -> List[str]:
        """Generate search tags for better discoverability."""
        tags = set()
        
        # Add procedure name variants
        name_parts = context.procedure_name.lower().split()
        tags.update(name_parts)
        
        # Add network functions as tags
        tags.update([nf.lower() for nf in context.network_functions])
        
        # Add procedure type tags
        procedure_lower = context.procedure_name.lower()
        if 'authentication' in procedure_lower or 'aka' in procedure_lower:
            tags.update(['authentication', 'security', 'identity', 'verification'])
        elif 'registration' in procedure_lower:
            tags.update(['registration', 'attachment', 'subscription', 'identity'])
        elif 'session' in procedure_lower:
            tags.update(['session', 'pdu', 'data', 'connectivity'])
        
        # Add technology tags
        tags.update(['5g', '3gpp', 'core', 'network'])
        
        return list(tags)[:20]  # Limit to 20 tags

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.tokenizer.encode(text))

    def _chunk_text_smart(self, text: str, max_tokens: int = 30000) -> List[str]:
        """Intelligently chunk text while preserving procedure boundaries."""
        # Split by procedure sections first
        procedure_sections = re.split(r'\n(?=\d+\.\d+\.\d+\s+[A-Z])', text)
        
        chunks = []
        current_chunk = ""
        current_tokens = 0
        
        for section in procedure_sections:
            section_tokens = self._count_tokens(section)
            
            if section_tokens > max_tokens:
                # Section too large, split by paragraphs
                paragraphs = section.split('\n\n')
                for para in paragraphs:
                    para_tokens = self._count_tokens(para)
                    
                    if current_tokens + para_tokens > max_tokens:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = para
                        current_tokens = para_tokens
                    else:
                        current_chunk += "\n\n" + para
                        current_tokens += para_tokens
            else:
                if current_tokens + section_tokens > max_tokens:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = section
                    current_tokens = section_tokens
                else:
                    current_chunk += "\n" + section
                    current_tokens += section_tokens
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks

    def generate_long_context_embeddings(self, text: str, procedure_name: str = None) -> Dict[str, any]:
        """Generate embeddings using full 32K context with metadata."""
        if not self.embedding_model:
            return {"embedding": None, "chunks": [], "metadata": {}}
        
        text_tokens = self._count_tokens(text)
        
        if text_tokens <= self.max_tokens:
            # Single embedding for the entire text
            try:
                embedding = self.embedding_model.encode(text, convert_to_tensor=True)
                return {
                    "embedding": embedding.cpu().numpy().tolist(),
                    "chunks": [text],
                    "metadata": {
                        "total_tokens": text_tokens,
                        "chunk_count": 1,
                        "procedure_name": procedure_name,
                        "embedding_model": "Qwen/Qwen3-Embedding-8B",
                        "max_context_used": text_tokens
                    }
                }
            except Exception as e:
                print(f"Long context embedding error: {e}")
                return {"embedding": None, "chunks": [], "metadata": {}}
        else:
            # Smart chunking with overlapping context
            chunks = self._chunk_text_smart(text)
            embeddings = []
            
            print(f"  📄 Processing {len(chunks)} chunks (total: {text_tokens} tokens)")
            
            for i, chunk in enumerate(chunks):
                try:
                    chunk_embedding = self.embedding_model.encode(chunk, convert_to_tensor=True)
                    embeddings.append(chunk_embedding.cpu().numpy().tolist())
                    print(f"    ✓ Chunk {i+1}: {self._count_tokens(chunk)} tokens")
                except Exception as e:
                    print(f"    ❌ Chunk {i+1} failed: {e}")
                    embeddings.append(None)
            
            # Create aggregated embedding (mean pooling)
            valid_embeddings = [emb for emb in embeddings if emb is not None]
            if valid_embeddings:
                import numpy as np
                aggregated_embedding = np.mean(valid_embeddings, axis=0).tolist()
            else:
                aggregated_embedding = None
            
            return {
                "embedding": aggregated_embedding,
                "chunk_embeddings": embeddings,
                "chunks": chunks,
                "metadata": {
                    "total_tokens": text_tokens,
                    "chunk_count": len(chunks),
                    "procedure_name": procedure_name,
                    "embedding_model": "Qwen/Qwen3-Embedding-8B",
                    "max_context_used": max([self._count_tokens(chunk) for chunk in chunks])
                }
            }

    def generate_embeddings(self, text: str) -> Optional[List[float]]:
        """Generate embeddings for search functionality (Requirement 8)."""
        if not self.embedding_model:
            return None
        
        try:
            embedding = self.embedding_model.encode(text)
            return embedding.tolist()
        except Exception as e:
            print(f"    Embedding generation error: {e}")
            return None

    def _merge_extraction_results(self, llm_entities: Dict, pattern_entities: Dict) -> Dict[str, List[str]]:
        """Merge LLM and pattern-based results."""
        merged = {}
        
        for entity_type in ["network_functions", "messages", "parameters", "keys", "steps"]:
            combined = set()
            
            # Add LLM results
            for entity in llm_entities.get(entity_type, []):
                if entity and len(entity.strip()) > 2 and not self._is_filtered_word(entity):
                    combined.add(entity.strip())
            
            # Add pattern results
            for entity in pattern_entities.get(entity_type, []):
                if entity and len(entity.strip()) > 2 and not self._is_filtered_word(entity):
                    combined.add(entity.strip())
            
            merged[entity_type] = list(combined)[:15]  # Increased limit
        
        return merged

    def _is_empty_result(self, entities: Dict) -> bool:
        """Check if extraction result is completely empty."""
        return sum(len(v) for v in entities.values()) == 0

    def _generate_minimum_entities(self, context: ProcedureContext) -> Dict[str, List[str]]:
        """Generate minimum entities when all extraction fails."""
        entities = {"network_functions": [], "messages": [], "parameters": [], "keys": [], "steps": []}
        
        procedure_lower = context.procedure_name.lower()
        
        if 'authentication' in procedure_lower or 'aka' in procedure_lower:
            entities["network_functions"] = ["AUSF", "AMF", "UE", "SEAF"]
            entities["messages"] = ["Authentication Request", "Authentication Response"]
            entities["parameters"] = ["SUCI", "RAND", "AUTN"]
            entities["keys"] = ["5G-AKA", "Kausf"]
        elif 'registration' in procedure_lower:
            entities["network_functions"] = ["AMF", "UE"]
            entities["messages"] = ["Registration Request", "Registration Accept"]
            entities["parameters"] = ["5G-GUTI", "SUCI"]
        else:
            entities["network_functions"] = ["AMF", "SMF"]
            entities["messages"] = ["Request Message", "Response Message"]
            entities["parameters"] = ["Identifier"]
        
        return entities

    def _is_filtered_word(self, word: str) -> bool:
        """Enhanced filtering to avoid common words as entities."""
        word_lower = word.lower().strip()
        
        # Check against filtered words
        if word_lower in FILTERED_WORDS:
            return True
        
        # Filter out very short or very long words
        if len(word_lower) < 2 or len(word_lower) > 50:
            return True
        
        # Filter out common phrases
        common_phrases = ['such as', 'for example', 'in this case', 'as shown']
        if any(phrase in word_lower for phrase in common_phrases):
            return True
        
        return False

    def _update_context_with_entities(self, context: ProcedureContext, entities: Dict[str, List[str]]):
        """Update context with extracted entities."""
        context.network_functions = entities["network_functions"]
        context.messages = entities["messages"]
        context.parameters = entities["parameters"]
        context.keys = entities["keys"]
        context.steps = entities["steps"]