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
        
        # Store current context for step entity creation
        self._current_context = context
        
        # Extract step descriptions from document text
        step_descriptions = self._extract_step_descriptions_from_document(text, context.procedure_name)
        entities["steps"] = step_descriptions
        
        return entities

    def _extract_step_descriptions_from_document(self, text: str, procedure_name: str) -> List[str]:
        """Extract step descriptions from 3GPP document, finding where steps actually begin."""
        
        print(f"        Enhanced step extraction for: {procedure_name}")
        
        # CRITICAL: Preprocess text to handle single-line input with embedded steps
        preprocessed_text = self._preprocess_step_text(text)
        
        # Split text into lines
        lines = preprocessed_text.split('\n')
        print(f"        📊 Total lines after preprocessing: {len(lines)}")
        
        # Show first 10 lines to understand structure
        print(f"        📋 First 10 lines after preprocessing:")
        for i, line in enumerate(lines[:10]):
            print(f"          Line {i+1}: '{line.strip()}'")
        
        # STEP 1: Find where the actual steps begin
        step_start_index = self._find_step_start_location(lines)
        
        if step_start_index == -1:
            print(f"        ❌ No step start location found - using full text")
            step_lines = lines
            start_line_used = 1
        else:
            print(f"        ✅ Found step start at line {step_start_index + 1}")
            step_lines = lines[step_start_index:]
            # start_line_used = step_start_index + 1
            start_line_used = step_start_index # FIXED: Correct line number for reporting
        
        print(f"        📊 Processing {len(step_lines)} lines starting from line {start_line_used}")
        
        # CORRECTED: Simplified step patterns (remove duplicates)
        step_patterns = [
            r'^(\d+)\.\s+(.+)',          # "1. ", "2. ", "12. " (handles multiple spaces/tabs)
            r'^(\d+[a-z])\.\s+(.+)',     # "7a. ", "1c. ", "2b. " (handles multiple spaces/tabs)
            r'^(\d+)\.[\s\t]+(.+)',      # Handle tabs and multiple spaces
            r'^(\d+[a-z])\.[\s\t]+(.+)', # Handle tabs and multiple spaces with letters
            # "7-12b. " pattern removed to avoid confusion
            
        ]
        
        steps = []
        current_step_num = None
        current_step_paragraphs = []
        
        i = 0
        while i < len(step_lines):
            line = step_lines[i].strip()
            step_found = False
            step_match = None;  # FIXED: Use separate variable to avoid overwriting
            
            # Check if this line starts a new step
            for pattern in step_patterns:
                pattern_match = re.match(pattern, line, re.IGNORECASE);  # FIXED: Use different variable name
                if pattern_match:
                    step_match = pattern_match;  # Save the successful match
                    step_found = True
                    break
            
            if step_found and step_match:  # FIXED: Check both conditions
                # Save previous step if exists
                if current_step_num is not None and current_step_paragraphs:
                    step_text = self._combine_step_paragraphs(current_step_paragraphs)
                    if step_text:
                        steps.append({
                            'number': current_step_num,
                            'text': step_text
                        })
                        print(f"          ✅ Saved step {current_step_num}: {len(step_text)} chars, {len([p for p in current_step_paragraphs if p.strip()])} paragraphs")
        
                # Start new step - FIXED: Use step_match instead of match
                current_step_num = step_match.group(1)  # "1", "2", "7a", "1c"
                initial_text = step_match.group(2)      # First paragraph of the step
                current_step_paragraphs = [initial_text]
                
                print(f"          🆕 Found step {current_step_num}: {initial_text[:50]}...")
        
            elif not step_found:
                # This line is part of current step (if we have one active)
                if current_step_num is not None:
                    if line.strip():  # Non-empty line
                        current_step_paragraphs.append(line.strip())
                        print(f"          ➕ Added to step {current_step_num}: {line.strip()[:40]}...")
                    else:
                        # Empty line - preserve paragraph separation
                        if current_step_paragraphs and current_step_paragraphs[-1]:
                            current_step_paragraphs.append("")  # Mark paragraph break
            
            i += 1
        
        # Don't forget the last step
        if current_step_num is not None and current_step_paragraphs:
            step_text = self._combine_step_paragraphs(current_step_paragraphs)
            if step_text:
                steps.append({
                    'number': current_step_num,
                    'text': step_text
                })
                print(f"          ✅ Final step {current_step_num}: {len(step_text)} chars, {len([p for p in current_step_paragraphs if p.strip()])} paragraphs")
    
        # Convert to expected format and sort by step number
        steps.sort(key=lambda x: self._step_sort_key(x['number']))
        
        # NEW: Create step entities directly from extracted steps
        if hasattr(self, '_current_context') and self._current_context:
            step_entities = self._create_step_entities_from_extracted_steps(steps, self._current_context)
            # Store the step entities for later use
            self._current_context.extracted_step_entities = step_entities
    
        # Still return formatted steps for backward compatibility
        formatted_steps = []
        for step in steps:
            formatted_steps.append(f"Step {step['number']}: {step['text']}")
        
        print(f"        ✅ Successfully extracted {len(formatted_steps)} multi-paragraph steps")
        
        return formatted_steps
    
    def _preprocess_step_text(self, text: str) -> str:
        """Preprocess text to handle single-line input with embedded step patterns."""
        
        print(f"        🔧 Preprocessing text for step extraction...")
        print(f"        📊 Original text length: {len(text)} chars, lines: {text.count(chr(10)) + 1}")
        
        # CRITICAL: Insert line breaks before step patterns
        # This handles cases where steps are concatenated in a single line
        
        # Pattern 1: Insert newline before "1. ", "2. ", etc. (but not at start of text)
        text = re.sub(r'(?<!^)(\d+)\.\t+([A-Z\[\(a-z])', r'\n\1. \2', text)
        
        # Pattern 2: Insert newline before "7a. ", "1c. ", etc. (but not at start of text)  
        text = re.sub(r'(?<!^)(\d+[a-z])\.\t+([A-Z\[\(a-z])', r'\n\1. \2', text)
        
        # Pattern 2-1: Insert newline before "7(A). "", etc. (but not at start of text)
        text = re.sub(r'(?<!^)(\d+\([A-Za-z]\))\.\t+([A-Z\[\(a-z])', r'\n\1. \2', text)

        # pattern 3: Insert newline before "7-12b. ", etc (but not at start of text)
        text = re.sub(r'(?<!^)(\d+-\d+[a-z])\.\t+([A-Z\[\(a-z])', r'\n\1. \2', text)
        
        # pattern 3-1: Insert newline before "7-12. ", etc (but not at start of text)
        text = re.sub(r'(?<!^)(\d+-\d+)\.\t+([A-Z\[\(a-z])', r'\n\1. \2', text)

        # pattern 3-2: Insert newline before "7b-12b. ", etc (but not at start of text) with spaces
        text = re.sub(r'(?<!^)(\d+[a-z]-\d+[a-z])\.\t+([A-Z\[\(a-z])', r'\n\1. \2', text)
        
        # pattern 3-3: Insert newline before "7b-12b. ", etc (but not at start of text) with spaces
        text = re.sub(r'(?<!^)(\d+[a-z]-\d+[a-z])\.\t+([A-Z\[\(a-z])', r'\n\1. \2', text)

        # Clean up multiple consecutive newlines
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        
        # Remove excessive spaces but preserve intentional formatting
        text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces/tabs to single space
        
        print(f"        📊 After preprocessing: {len(text)} chars, lines: {text.count(chr(10)) + 1}")
        print(f"        📋 Preprocessing preview: '{text[:300]}...'")
        
        return text.strip()

    def _find_step_start_location(self, lines: List[str]) -> int:
        """Find the line index where actual procedure steps begin."""
        
        # Look for common indicators that steps are about to start
        step_start_indicators = [
            # r'^The following steps?\s+(are\s+)?performed:?',          # "The following steps are performed:"
            # r'^The procedure\s+(consists\s+of\s+)?the following:?',   # "The procedure consists of the following:"
            # r'^Steps?\s+\d+',                                         # "Step 1", "Steps 1-5"
            # r'^Below\s+(are\s+)?the\s+detailed\s+steps:?',           # "Below are the detailed steps:"
            # r'^The\s+detailed\s+procedure\s+is\s+as\s+follows:?',    # "The detailed procedure is as follows:"
            # r'^Procedure\s+steps:?',                                  # "Procedure steps:"
            r'^\d+\.\s+',     # "1. "
            r'^\d+[a-z]\.\s+' # "1a. "
            r'^(\d+)\.[\s\t]+(.+)',      # Handle tabs and multiple spaces
            r'^(\d+[a-z])\.[\s\t]+(.+)', # Handle tabs and multiple spaces with letters
            r'^(\d+-\d+[a-z])\.[\s\t]+(.+)', # "7-12b. "
            r'^(\d+-\d+)\.[\s\t]+(.+)',    # "7-12. "
            r'^(\d+[a-z]-\d+[a-z])\.[\s\t]+(.+)', # "7b-12b. "
            r'^(\d+[a-z]-\d+)\.[\s\t]+(.+)',    # "7b-12. "
            r'^(\d+\([A-Za-z]\))\.[\s\t]+(.+)'
        ]
        
        # Also look for the first occurrence of step numbering
        step_number_patterns = [
            r'^\d+\.\s+',     # "1. "
            r'^\d+[a-z]\.\s+' # "1a. "
            r'^(\d+)\.[\s\t]+(.+)',      # Handle tabs and multiple spaces
            r'^(\d+[a-z])\.[\s\t]+(.+)', # Handle tabs and multiple spaces with letters
            r'^(\d+-\d+[a-z])\.[\s\t]+(.+)', # "7-12b. "
            r'^(\d+-\d+)\.[\s\t]+(.+)',    # "7-12. "
            r'^(\d+[a-z]-\d+[a-z])\.[\s\t]+(.+)', # "7b-12b. "
            r'^(\d+[a-z]-\d+)\.[\s\t]+(.+)',    # "7b-12. "
            r'^(\d+\([A-Za-z]\))\.[\s\t]+(.+)'
        ]
        
        print(f"        🔍 Searching for step start indicators in {len(lines)} lines...")
        
        # First, look for explicit step start indicators
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            
            # Check for step start indicators
            for pattern in step_start_indicators:
                if re.match(pattern, line_stripped, re.IGNORECASE):
                    print(f"          📍 Found step start indicator at line {i+1}: '{line_stripped}'")
                    # Return the next line after the indicator, or current line if it contains steps
                    return i if i < len(lines) else i - 1

        # If no explicit indicators, look for the first step number
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            
            # Check for step numbering patterns
            for pattern in step_number_patterns:
                if re.match(pattern, line_stripped, re.IGNORECASE):
                    print(f"          📍 Found first step at line {i+1}: '{line_stripped}'")
                    return i
        
        # If still no steps found, look for common step content patterns
        # This catches cases where steps start without explicit numbering
        step_content_patterns = [
            r'^(The\s+)?(UE|AMF|SMF|UPF|AUSF|PCF|UDM|NRF)\s+(sends?|receives?|performs?|initiates?)',
            r'^(Upon|After|When|If)\s+.+(the\s+)?(UE|AMF|SMF)',
            r'^First,?\s+(the\s+)?(UE|AMF|SMF)',
            r'^Initially,?\s+(the\s+)?(UE|AMF|SMF)'
        ]
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            
            # Skip very short lines and section headers
            if len(line_stripped) < 20 or re.match(r'^\d+(\.\d+)*\s+[A-Z]', line_stripped):
                continue
            
            # Check for step content patterns
            for pattern in step_content_patterns:
                if re.match(pattern, line_stripped, re.IGNORECASE):
                    print(f"          📍 Found step content pattern at line {i+1}: '{line_stripped[:50]}...'")
                    return i
        
        print(f"        ❌ No step start location found - will process entire text")
        return -1  # No step start found, use entire text

    def _combine_step_paragraphs(self, paragraphs: List[str]) -> str:
        """Combine multiple paragraphs into a coherent step description."""
        
        if not paragraphs:
            return ""
        
        # Process paragraphs while preserving structure
        processed_paragraphs = []
        current_paragraph = ""
        
        for para in paragraphs:
            if para == "":  # Empty string marks paragraph break
                if current_paragraph.strip():
                    processed_paragraphs.append(current_paragraph.strip())
                    current_paragraph = ""
            else:
                if current_paragraph:
                    current_paragraph += " " + para
                else:
                    current_paragraph = para
        
        # Add the last paragraph
        if current_paragraph.strip():
            processed_paragraphs.append(current_paragraph.strip())
        
        if not processed_paragraphs:
            return ""
        
        # Join paragraphs with double space to indicate paragraph breaks
        result = "  ".join(processed_paragraphs)
        
        # Clean up excessive whitespace within paragraphs
        result = re.sub(r'[ \t]+', ' ', result)  # Multiple spaces/tabs to single space
        
        return result.strip()

    def _step_sort_key(self, step_num: str):
        """Create sort key for step numbers including letters (1, 2, 7a, 7b, 8)."""
        
        # Handle patterns like "1", "2", "7a", "1c"
        match = re.match(r'^(\d+)([a-z]?)$', step_num.lower())
        if match:
            num = int(match.group(1))
            letter = match.group(2)
            letter_ord = ord(letter) if letter else 0
            return (num, letter_ord)
        
        # Fallback for unexpected patterns
        return (999, 999)

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
        """Use pre-created step entities if available, otherwise fall back to old logic."""
    
        # Check if we already have step entities from extraction
        if hasattr(context, 'extracted_step_entities') and context.extracted_step_entities:
            print(f"      ✅ Using {len(context.extracted_step_entities)} pre-created step entities")
            entities['steps'] = context.extracted_step_entities
            return entities
        
        # Fallback to existing logic if no pre-created entities
        print(f"      📝 No pre-created entities, using fallback logic")
        
        # Extract section number and clean procedure name separately  
        procedure_title = context.procedure_name
        
        # Extract section number (like "4.3.2.2.1") from the beginning
        section_match = re.match(r'^(\d+(?:\.\d+)*)\s+(.+)', procedure_title)
        
        if section_match:
            section_number = section_match.group(1).replace('.', '')  # "43221"
            clean_title = section_match.group(2)  # "Non-roaming and Roaming with Local Breakout"
        else:
            # Fallback if no section number found
            section_number = ""
            clean_title = procedure_title
        
        # Create procedure identifier for step naming
        procedure_clean = re.sub(r'[^\w\s]', '', clean_title).replace(' ', '_')
        
        # Get step descriptions from NEW step extraction system
        step_descriptions = entities.get('steps', [])
        
        print(f"      Processing {len(step_descriptions)} steps from new extraction system")
        
        # ENHANCED: Process steps with their original 3GPP numbering
        if step_descriptions:
            procedure_steps = []
            context.step_descriptions = {}
            
            for step_desc in step_descriptions:
                # Extract step number from description (e.g., "Step 1: UE sends..." → "1")
                step_match = re.match(r'^Step\s+([0-9a-z-]+):\s*(.+)', step_desc, re.IGNORECASE)
                
                if step_match:
                    original_step_num = step_match.group(1)  # "1", "7a", "7-12b" etc.
                    step_content = step_match.group(2)
                    
                    # Convert 3GPP step number to entity naming format
                    entity_step_num = self._convert_step_number_for_entity(original_step_num)
                    
                    # FIXED: Create step entity name using underscores instead of tabs
                    if section_number:
                        step_name = f"{section_number}_{procedure_clean}_step_{entity_step_num}"
                    else:
                        step_name = f"{procedure_clean}_step_{entity_step_num}"
                    
                    procedure_steps.append(step_name)
                    
                    # Store full multi-paragraph description (as requested)
                    context.step_descriptions[step_name] = step_content
                    
                    print(f"        ✓ Step {original_step_num} → {step_name}")
                    print(f"          Description: {step_content[:80]}...")
                
                else:
                    print(f"        ⚠️ Could not parse step: {step_desc[:50]}...")
            
            # If no valid steps were parsed, fall back to sequential numbering
            if not procedure_steps:
                procedure_steps, context.step_descriptions = self._create_fallback_steps(
                    section_number, procedure_clean, step_descriptions
                )
        
        else:
            # Generate default steps if no descriptions found
            procedure_steps, context.step_descriptions = self._create_default_steps(
                section_number, procedure_clean, context
            )
            print(f"      Generated {len(procedure_steps)} default steps")
        
        entities['steps'] = procedure_steps
        print(f"      ✅ Created {len(procedure_steps)} step entities with descriptions")
        
        return entities

    def _convert_step_number_for_entity(self, step_num: str) -> str:
        """Convert 3GPP step number to entity naming format."""
        # Handle various 3GPP step numbering formats
        
        # Simple numbers: "1" → "1"
        if re.match(r'^\d+$', step_num):
            return step_num
        
        # Letter suffixed: "7a" → "7a"  
        if re.match(r'^\d+[a-z]$', step_num):
            return step_num
        
        # Range patterns: "7-12b" → "7_12b"
        if re.match(r'^\d+-\d+[a-z]*$', step_num):
            return step_num.replace('-', '_')
        
        # Complex ranges: "7a-12b" → "7a_12b"
        if re.match(r'^\d+[a-z]-\d+[a-z]*$', step_num):
            return step_num.replace('-', '_')
        
        # Parenthetical: "7(A)" → "7A"
        if re.match(r'^\d+\([A-Za-z]\)$', step_num):
            return re.sub(r'\(([A-Za-z])\)', r'\1', step_num)
        
        # Default: replace special characters with underscore
        return re.sub(r'[^0-9a-zA-Z]', '_', step_num)

    def _create_fallback_steps(self, section_number: str, procedure_clean: str, 
                      step_descriptions: List[str]) -> Tuple[List[str], Dict[str, str]]:
        """Create fallback steps with sequential numbering when parsing fails."""
        procedure_steps = []
        descriptions_dict = {}
        
        for i, desc in enumerate(step_descriptions, 1):
            # FIXED: Use underscores instead of tabs
            if section_number:
                step_name = f"{section_number}_{procedure_clean}_step_{i}"
            else:
                step_name = f"{procedure_clean}_step_{i}"
        
            procedure_steps.append(step_name)
            
            # Clean description of any step prefixes
            clean_desc = re.sub(r'^Step\s+[0-9a-z-]+:\s*', '', desc, flags=re.IGNORECASE)
            descriptions_dict[step_name] = clean_desc
            
            print(f"        ✓ Fallback Step {i} → {step_name}")
        
        return procedure_steps, descriptions_dict

    def _create_default_steps(self, section_number: str, procedure_clean: str, 
                     context: ProcedureContext) -> Tuple[List[str], Dict[str, str]]:
        """Create default steps when no extraction is available."""
        default_descriptions = self._generate_default_step_descriptions(context)
        procedure_steps = []
        descriptions_dict = {}
        
        for i, desc in enumerate(default_descriptions, 1):
            # FIXED: Use underscores instead of tabs
            if section_number:
                step_name = f"{section_number}_{procedure_clean}_step_{i}"
            else:
                step_name = f"{procedure_clean}_step_{i}"
        
            procedure_steps.append(step_name)
            descriptions_dict[step_name] = desc
            
            print(f"        ✓ Default Step {i} → {step_name}")
    
        return procedure_steps, descriptions_dict

    def extract_entities_for_procedure_old(self, context: ProcedureContext) -> ExtractionResult:
        """Original entity extraction method (for comparison)."""
        try:
            print(f"    Extracting entities for (old method): {context.procedure_name}")
            
            # Simple keyword-based extraction
            text = context.section.text
            entities = {"network_functions": [], "messages": [], "parameters": [], "keys": [], "steps": []}
            
            # Extract network functions
            for nf in KNOWN_NETWORK_FUNCTIONS:
                if re.search(r'\b' + re.escape(nf) + r'\b', text, re.IGNORECASE):
                    entities["network_functions"].append(nf)
            
            # Extract messages
            for msg in KNOWN_MESSAGES:
                if re.search(r'\b' + re.escape(msg) + r'\b', text, re.IGNORECASE):
                    entities["messages"].append(msg)
            
            # Extract parameters
            for param in KNOWN_PARAMETERS:
                if re.search(r'\b' + re.escape(param) + r'\b', text, re.IGNORECASE):
                    entities["parameters"].append(param)
            
            # Extract keys
            for key in KNOWN_KEYS:
                if re.search(r'\b' + re.escape(key) + r'\b', text, re.IGNORECASE):
                    entities["keys"].append(key)
            
            # Simple step extraction (naive)
            step_pattern = r'(\d+)\.\s+([^\d]+)'
            steps = re.findall(step_pattern, text)
            for step in steps:
                step_number, step_text = step
                entities["steps"].append(f"{step_number}. {step_text.strip()}")
            
            # Clean up duplicates
            for key in entities:
                entities[key] = list(set(entities[key]))
            
            # Generate default step descriptions if no steps found
            if not entities["steps"]:
                entities["steps"] = self._generate_default_step_descriptions(context)
            
            # Update context with extracted entities
            self._update_context_with_entities(context, entities)
            
            total_entities = sum(len(v) for v in entities.values())
            print(f"      ✓ Extracted {total_entities} entities (old method)")
            
            return ExtractionResult(
                entities=entities,
                relationships=[],
                success=True,
                extraction_method="keyword_based_old",
                llm_model_used="none"
            )
        
        except Exception as e:
            print(f"    ✗ Entity extraction failed (old method): {e}")
            return ExtractionResult(
                entities={},
                relationships=[],
                success=False,
                error_message=str(e)
            )

    # NEW: Strict whitelist enforcement methods
    def _enforce_nf_whitelist_old(self, extracted_nfs: List[str]) -> List[str]:
        """Enforce strict network function whitelist validation (old method)."""
        validated_nfs = []
        rejected_count = 0
        
        for nf in extracted_nfs:
            nf_upper = nf.upper().strip()
            if nf_upper in self.valid_network_functions:
                validated_nfs.append(nf_upper)
                print(f"        ✓ Validated NF (old): {nf} -> {nf_upper}")
            else:
                print(f"        ✗ Rejected NF (old): {nf} (not in whitelist)")
                rejected_count += 1
        
        if rejected_count > 0:
            print(f"      📋 Rejected {rejected_count} invalid network functions (old method)")
        
        return validated_nfs

    def extract_entities_for_procedure_fallback(self, context: ProcedureContext) -> ExtractionResult:
        """Fallback entity extraction method (for comparison)."""
        try:
            print(f"    Extracting entities for (fallback method): {context.procedure_name}")
            
            # Simple keyword-based extraction
            text = context.section.text
            entities = {"network_functions": [], "messages": [], "parameters": [], "keys": [], "steps": []}
            
            # Extract network functions
            for nf in KNOWN_NETWORK_FUNCTIONS:
                if re.search(r'\b' + re.escape(nf) + r'\b', text, re.IGNORECASE):
                    entities["network_functions"].append(nf)
            
            # Extract messages
            for msg in KNOWN_MESSAGES:
                if re.search(r'\b' + re.escape(msg) + r'\b', text, re.IGNORECASE):
                    entities["messages"].append(msg)
            
            # Extract parameters
            for param in KNOWN_PARAMETERS:
                if re.search(r'\b' + re.escape(param) + r'\b', text, re.IGNORECASE):
                    entities["parameters"].append(param)
            
            # Extract keys
            for key in KNOWN_KEYS:
                if re.search(r'\b' + re.escape(key) + r'\b', text, re.IGNORECASE):
                    entities["keys"].append(key)
            
            # Simple step extraction (naive)
            step_pattern = r'(\d+)\.\s+([^\d]+)'
            steps = re.findall(step_pattern, text)
            for step in steps:
                step_number, step_text = step
                entities["steps"].append(f"{step_number}. {step_text.strip()}")
            
            # Clean up duplicates
            for key in entities:
                entities[key] = list(set(entities[key]))
            
            # Generate default step descriptions if no steps found
            if not entities["steps"]:
                entities["steps"] = self._generate_default_step_descriptions(context)
            
            # Update context with extracted entities
            self._update_context_with_entities(context, entities)
            
            total_entities = sum(len(v) for v in entities.values())
            print(f"      ✓ Extracted {total_entities} entities (fallback method)")
            
            return ExtractionResult(
                entities=entities,
                relationships=[],
                success=True,
                extraction_method="keyword_based_fallback",
                llm_model_used="none"
            )
        
        except Exception as e:
            print(f"    ✗ Entity extraction failed (fallback method): {e}")
            return ExtractionResult(
                entities={},
                relationships=[],
                success=False,
                error_message=str(e)
            )

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

    def _is_step_continuation(self, line: str, context: Dict) -> bool:
        """Determine if line is continuation using contextual analysis instead of explicit patterns."""
        
        if not line or len(line.strip()) < 5:
            return True
        
        line_stripped = line.strip()
        
        # 1. DEFINITE step starters (these override everything else)
        step_starter_patterns = [
            r'^\d+[a-z]*\.\s+',           # "1. ", "7a. "
            r'^Step\s+\d+',               # "Step 1"
            # r'^Figure\s+\d+',             # "Figure 1"
            # r'^NOTE[:\s]',                # "NOTE:"
        ]
        
        for pattern in step_starter_patterns:
            if re.match(pattern, line_stripped, re.IGNORECASE):
                return False  # Definitely NOT continuation
        
        # 2. CONTEXTUAL analysis - is this similar to previous step content?
        current_step_text = context.get('current_step_text', '')
        if current_step_text:
            similarity = self._calculate_semantic_similarity(line_stripped, current_step_text)
            if similarity > 0.7:  # High semantic similarity suggests continuation
                return True
        
        # 3. SENTENCE structure analysis
        sentence_features = self._analyze_sentence_structure(line_stripped)
        
        # Features that suggest continuation:
        continuation_score = 0
        
        # Starts with connection words (but learned, not hard-coded)
        if sentence_features['starts_with_connector']:
            continuation_score += 0.3
        
        # References previous concepts
        if sentence_features['has_back_reference']:
            continuation_score += 0.4
        
        # Similar technical vocabulary to current step
        if sentence_features['vocab_similarity'] > 0.6:
            continuation_score += 0.3
        
        # 4. LENGTH and COMPLEXITY heuristics
        if len(line_stripped) < 100:  # Short sentences more likely to be continuation
            continuation_score += 0.2
        
        # 5. FINAL decision based on cumulative score
        return continuation_score > 0.5  # Threshold can be tuned

    def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between texts using embeddings."""
        if not self.embedding_model or not text1 or not text2:
            return 0.0
        
        try:
            # Use the embedding model for semantic comparison
            emb1 = self.embedding_model.encode(text1[:200])  # Limit length
            emb2 = self.embedding_model.encode(text2[:200])
            
            # Cosine similarity
            import numpy as np
            similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            return max(0, similarity)  # Ensure non-negative
            
        except:
            return 0.0

    def _analyze_sentence_structure(self, sentence: str) -> Dict[str, any]:
        """Analyze sentence structure for continuation indicators."""
        features = {
            'starts_with_connector': False,
            'has_back_reference': False,
            'vocab_similarity': 0.0,
            'sentence_length': len(sentence),
            'starts_with_lowercase': sentence and sentence[0].islower()
        }
        
        sentence_lower = sentence.lower()
        
        # LEARNED connectors (more flexible than hard-coded list)
        connector_patterns = [
            # Causal/temporal
            r'^(after|before|when|once|since|until|while)',
            # Conditional  
            r'^(if|unless|provided|given)',
            # Additive
            r'^(also|additionally|furthermore|moreover)',
            # Contrastive
            r'^(however|nevertheless|nonetheless|although)',
            # Reference
            r'^(this|that|these|those|such|the\s+\w+)',
        ]
        
        features['starts_with_connector'] = any(
            re.match(pattern, sentence_lower) for pattern in connector_patterns
        )
        
        # Back-references to previous content
        back_ref_indicators = [
            r'\b(this|that|these|those|such|the\s+above|the\s+following)\b',
            r'\b(it|they|them|its|their)\b',
            r'\b(the\s+\w+\s+mentioned|as\s+described|as\s+stated)\b'
        ]
        
        features['has_back_reference'] = any(
            re.search(pattern, sentence_lower) for pattern in back_ref_indicators
        )
        
        return features

    def _create_step_entities_from_extracted_steps(self, steps: List[Dict[str, str]], context: ProcedureContext) -> List[str]:
        """Create step entities directly from extracted steps data."""

        # Extract section number and clean procedure name
        procedure_title = context.procedure_name
        section_match = re.match(r'^(\d+(?:\.\d+)*)\s+(.+)', procedure_title)

        if section_match:
            section_number = section_match.group(1).replace('.', '')  # "4.3.2.2.1" → "43221"
            clean_title = section_match.group(2)  # "Non-roaming and Roaming with Local Breakout"
        else:
            section_number = ""
            clean_title = procedure_title

        # Create procedure identifier for step naming
        procedure_clean = re.sub(r'[^\w\s]', '', clean_title).replace(' ', '_')

        procedure_steps = []
        context.step_descriptions = {}

        print(f"        🏗️  Creating {len(steps)} step entities from extracted steps")

        for step_dict in steps:
            original_step_num = step_dict['number']  # "1", "7a", "20b"
            step_content = step_dict['text']         # Full content

            # Convert 3GPP step number to entity naming format
            entity_step_num = self._convert_step_number_for_entity(original_step_num)

            # FIXED: Create step entity name using underscores instead of tabs
            if section_number:
                step_name = f"{section_number}_{procedure_clean}_step_{entity_step_num}"
            else:
                step_name = f"{procedure_clean}_step_{entity_step_num}"

            procedure_steps.append(step_name)

            # Store full multi-paragraph description
            context.step_descriptions[step_name] = step_content

            print(f"          ✓ Step {original_step_num} → {step_name}")

        print(f"        ✅ Created {len(procedure_steps)} step entities with descriptions")

        return procedure_steps