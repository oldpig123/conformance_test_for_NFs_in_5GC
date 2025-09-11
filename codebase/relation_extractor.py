import re
import json
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings("ignore")

from config import REQUIRED_RELATIONS
from data_structures import ProcedureContext, Relationship

class RelationExtractor:
    """Handles relationship extraction using NLP/LLM (Step 3)."""
    
    def __init__(self, text_generator=None):
        self.text_generator = text_generator
        self.is_t5_model = getattr(text_generator, 'model', None) and 't5' in str(text_generator.model.__class__).lower()
    
    def extract_relationships_for_procedure(self, context: ProcedureContext) -> List[Relationship]:
        """Extract relationships with step-based NF filtering."""
        print(f"      Extracting relationships for: {context.procedure_name}")
        
        # NEW: Filter NFs by step involvement
        original_nfs = context.network_functions.copy()
        validated_nfs = self._filter_nfs_by_step_involvement(context)
        context.network_functions = validated_nfs
        
        print(f"        Step filtering: {len(original_nfs)} -> {len(validated_nfs)} NFs")
        
        relationships = []
        
        # Extract LLM-based relationships
        llm_relationships = self._query_llm_for_relationships(context)
        relationships.extend(self._process_llm_relationships(llm_relationships, context))
        
        # Extract required relationships (now with filtered NFs)
        required_relationships = self._extract_required_relationships(context)
        relationships.extend(required_relationships)
        
        # Restore original NFs for context
        context.network_functions = original_nfs
        
        print(f"      ✓ Extracted {len(relationships)} relationships")
        return relationships
    
    def _query_llm_for_relationships(self, context: ProcedureContext) -> List[Dict]:
        """Query LLM for relationship extraction."""
        if not self.text_generator:
            return []
        
        # Prepare entity list
        entity_list = []
        entity_list.append(f"- {context.procedure_name} (Procedure)")
        
        for nf in context.network_functions:
            entity_list.append(f"- {nf} (NetworkFunction)")
        for msg in context.messages:
            entity_list.append(f"- {msg} (Message)")
        for param in context.parameters:
            entity_list.append(f"- {param} (Parameter)")
        for key in context.keys:
            entity_list.append(f"- {key} (Key)")
        for step in context.steps:  # These are now procedure-specific names
            entity_list.append(f"- {step} (Step)")
        
        if len(entity_list) < 3:
            return []
        
        prompt = f"""
Extract relationships between entities in this 3GPP telecommunications procedure.

Procedure: "{context.procedure_name}"
Entities:
{chr(10).join(entity_list[:20])}

Use EXACTLY these relationship types:
- INVOLVE (NetworkFunction involves Step)
- FOLLOWED_BY (Step_n followed by Step_n+1)
- CONTAINS (Step contains Parameter/Key, Message contains Parameter/Key)
- INVOKE (Procedure invokes NetworkFunction)
- SEND_BY (Message sent by NetworkFunction)
- SEND_TO (Message sent to NetworkFunction)
- PART_OF (Step is part of Procedure)

JSON format:
{{
    "relationships": [
        {{"source": "AMF", "target": "Step 1", "relation": "INVOLVE"}},
        {{"source": "Step 1", "target": "Step 2", "relation": "FOLLOWED_BY"}}
    ]
}}
"""

        try:
            if self.is_t5_model:
                result = self.text_generator(prompt, max_length=300, num_return_sequences=1)
                response_text = result[0]['generated_text']
            else:
                result = self.text_generator(prompt, max_length=len(prompt) + 300, num_return_sequences=1)
                response_text = result[0]['generated_text'][len(prompt):]
            
            # Extract JSON
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                try:
                    relations_dict = json.loads(json_match.group())
                    if 'relationships' in relations_dict:
                        return relations_dict['relationships'][:15]
                except json.JSONDecodeError:
                    pass
            
        except Exception as e:
            print(f"        LLM relationship error: {e}")
        
        return []
    
    def _process_llm_relationships(self, llm_relationships: List[Dict], context: ProcedureContext) -> List[Relationship]:
        """Process LLM-extracted relationships."""
        relationships = []
        
        for rel in llm_relationships:
            if (isinstance(rel, dict) and 
                'source' in rel and 'target' in rel and 'relation' in rel):
                
                source = rel['source']
                target = rel['target']
                relation = rel['relation']
                
                if relation in REQUIRED_RELATIONS:
                    relationships.append(Relationship(
                        source_name=source,
                        target_name=target,
                        rel_type=relation,
                        properties={
                            "extraction_method": "llm_extraction",
                            "procedure": context.procedure_name
                        }
                    ))
        
        return relationships
    
    def _extract_required_relationships(self, context: ProcedureContext) -> List[Relationship]:
        """Extract strictly required relationships with UPDATED relation types."""
        relationships = []
        
        # 1. {Step, PART_OF, Procedure} - Use actual step names from context
        for step_name in context.steps:
            relationships.append(Relationship(
                source_name=step_name,
                target_name=context.procedure_name,
                rel_type="PART_OF",
                properties={
                    "relationship_type": "containment",
                    "extraction_method": "required_relationship"
                }
            ))
        
        # 2. {Step_n, FOLLOWED_BY, Step_n+1} - Use actual step names
        for i in range(len(context.steps) - 1):
            source_step = context.steps[i]
            target_step = context.steps[i + 1]
            relationships.append(Relationship(
                source_name=source_step,
                target_name=target_step,
                rel_type="FOLLOWED_BY",
                properties={
                    "sequence_order": i + 1,
                    "extraction_method": "required_relationship"
                }
            ))
        
        # 3. {Procedure, INVOKE, NetworkFunction}
        for nf in context.network_functions:
            relationships.append(Relationship(
                source_name=context.procedure_name,
                target_name=nf,
                rel_type="INVOKE",
                properties={
                    "relationship_type": "invocation",
                    "extraction_method": "required_relationship"
                }
            ))
        
        # 4. {NetworkFunction, INVOLVE, Step} - Use actual step names
        for nf in context.network_functions:
            for step_name in context.steps:
                if self._nf_involved_in_step(nf, step_name, context):
                    relationships.append(Relationship(
                        source_name=nf,
                        target_name=step_name,
                        rel_type="INVOLVE",
                        properties={
                            "relationship_type": "participation",
                            "extraction_method": "required_relationship"
                        }
                    ))
        
        # 5. {Message, SEND_BY/SEND_TO, NetworkFunction} - Enhanced message direction
        for msg in context.messages:
            sender_nf, receiver_nf = self._enhanced_message_direction(msg, context)
            
            if sender_nf and sender_nf in context.network_functions:
                relationships.append(Relationship(
                    source_name=msg,
                    target_name=sender_nf,
                    rel_type="SEND_BY",
                    properties={
                        "relationship_type": "communication",
                        "extraction_method": "enhanced_message_analysis",
                        "message_direction": "sender"
                    }
                ))
                print(f"        ✓ SEND_BY: {msg} -> {sender_nf}")
            
            if receiver_nf and receiver_nf in context.network_functions:
                relationships.append(Relationship(
                    source_name=msg,
                    target_name=receiver_nf,
                    rel_type="SEND_TO",
                    properties={
                        "relationship_type": "communication",
                        "extraction_method": "enhanced_message_analysis",
                        "message_direction": "receiver"
                    }
                ))
                print(f"        ✓ SEND_TO: {msg} -> {receiver_nf}")
        
        # 6. NEW: {Step, SEND, Message} - ADDED based on your requirements
        for step_name in context.steps:
            for msg in context.messages:
                # Heuristic: if message is mentioned in step description or step context
                step_desc = context.step_descriptions.get(step_name, "").lower()
                if (msg.lower() in step_desc or 
                    any(word in msg.lower() for word in ['request', 'response', 'message']) and
                    any(word in step_desc for word in ['send', 'transmit', 'forward'])):
                    
                    relationships.append(Relationship(
                        source_name=step_name,
                        target_name=msg,
                        rel_type="SEND",
                        properties={
                            "relationship_type": "message_transmission",
                            "extraction_method": "step_message_analysis"
                        }
                    ))
                    print(f"        ✓ SEND: {step_name} -> {msg}")
        
        # 7. {Step, CONTAINS, Parameter/Key} - Use actual step names with descriptions
        for step_name in context.steps:
            step_desc = context.step_descriptions.get(step_name, "").lower()
            
            for param in context.parameters:
                # Check if parameter is mentioned in step description
                if param.lower() in step_desc or param.lower() in step_name.lower():
                    relationships.append(Relationship(
                        source_name=step_name,
                        target_name=param,
                        rel_type="CONTAINS",
                        properties={
                            "relationship_type": "containment",
                            "extraction_method": "step_description_analysis"
                        }
                    ))
            
            for key in context.keys:
                # Check if key is mentioned in step description
                if key.lower() in step_desc or key.lower() in step_name.lower():
                    relationships.append(Relationship(
                        source_name=step_name,
                        target_name=key,
                        rel_type="CONTAINS",
                        properties={
                            "relationship_type": "containment",
                            "extraction_method": "step_description_analysis"
                        }
                    ))
        
        # 8. {Message, CONTAINS, Parameter/Key} - Enhanced with document analysis
        for msg in context.messages:
            for param in context.parameters:
                # More sophisticated containment analysis
                if self._message_contains_parameter(msg, param, context):
                    relationships.append(Relationship(
                        source_name=msg,
                        target_name=param,
                        rel_type="CONTAINS",
                        properties={
                            "relationship_type": "message_content",
                            "extraction_method": "document_analysis"
                        }
                    ))
            
            for key in context.keys:
                if self._message_contains_parameter(msg, key, context):
                    relationships.append(Relationship(
                        source_name=msg,
                        target_name=key,
                        rel_type="CONTAINS",
                        properties={
                            "relationship_type": "message_content",
                            "extraction_method": "document_analysis"
                        }
                    ))
        
        return relationships
    
    def _message_contains_parameter(self, message: str, parameter: str, context: ProcedureContext) -> bool:
        """Enhanced check if message contains parameter based on document context."""
        message_lower = message.lower()
        param_lower = parameter.lower()
        
        # Direct name match
        if param_lower in message_lower:
            return True
        
        # Check in document text context
        section_text_lower = context.section.text.lower()
        
        # Look for patterns like "MessageName contains ParameterName"
        patterns = [
            f"{message_lower}.*{param_lower}",
            f"{param_lower}.*{message_lower}",
            f"{message_lower}.*include.*{param_lower}",
            f"{message_lower}.*contain.*{param_lower}"
        ]
        
        for pattern in patterns:
            if re.search(pattern, section_text_lower, re.DOTALL):
                return True
        
        return False
    
    def _enhanced_message_direction(self, message: str, context: ProcedureContext) -> Tuple[Optional[str], Optional[str]]:
        """CRITICAL FIX: Enhanced message direction determination for SEND_BY relationships."""
        message_lower = message.lower()
        
        # Method 1: 5G Service-Based Interface pattern analysis
        # Pattern: Nxx_ServiceName_Operation (e.g., Nausf_UEAuthentication_Authenticate)
        sbi_pattern = r'^n([a-z]+)_'
        match = re.match(sbi_pattern, message_lower)
        
        if match:
            # Extract NF code from message pattern
            nf_code = match.group(1).upper()
            
            # Enhanced NF mapping for 5G core
            nf_mapping = {
                'AUSF': 'AUSF', 'AMF': 'AMF', 'SMF': 'SMF', 'UDM': 'UDM',
                'UDR': 'UDR', 'PCF': 'PCF', 'NSSF': 'NSSF', 'NEF': 'NEF',
                'NRF': 'NRF', 'SMSF': 'SMSF', 'UPF': 'UPF', 'CHF': 'CHF',
                'BSF': 'BSF', 'UDSF': 'UDSF'
            }
            
            service_provider = nf_mapping.get(nf_code)
            if service_provider and service_provider in context.network_functions:
                # For SBI messages, the NF in the interface name is typically the service provider
                # Find a different NF as consumer
                service_consumer = None
                for nf in context.network_functions:
                    if nf != service_provider:
                        service_consumer = nf
                        break
                
                # Determine direction based on message type
                if 'request' in message_lower:
                    return service_consumer, service_provider  # Consumer sends request TO provider
                elif 'response' in message_lower:
                    return service_provider, service_consumer  # Provider sends response TO consumer
                else:
                    return service_consumer, service_provider  # Default: consumer to provider
        
        # Method 2: Request/Response pattern analysis
        if 'request' in message_lower:
            # Find sender and receiver from context
            potential_senders = []
            potential_receivers = []
            
            for nf in context.network_functions:
                # Heuristics based on common 5G patterns
                if nf in ['UE', 'AMF', 'SMF']:  # Common initiators
                    potential_senders.append(nf)
                if nf in ['AUSF', 'UDM', 'UDR', 'PCF']:  # Common service providers
                    potential_receivers.append(nf)
            
            if potential_senders and potential_receivers:
                return potential_senders[0], potential_receivers[0]
            elif len(context.network_functions) >= 2:
                return context.network_functions[0], context.network_functions[1]
        
        elif 'response' in message_lower:
            # Response - reverse the direction
            potential_senders = []
            potential_receivers = []
            
            for nf in context.network_functions:
                if nf in ['AUSF', 'UDM', 'UDR', 'PCF']:  # Service providers send responses
                    potential_senders.append(nf)
                if nf in ['UE', 'AMF', 'SMF']:  # Service consumers receive responses
                    potential_receivers.append(nf)
            
            if potential_senders and potential_receivers:
                return potential_senders[0], potential_receivers[0]
            elif len(context.network_functions) >= 2:
                return context.network_functions[1], context.network_functions[0]
        
        # Method 3: Procedure-specific heuristics
        procedure_lower = context.procedure_name.lower()
        
        if 'authentication' in procedure_lower:
            # Authentication procedures typically involve UE/AMF -> AUSF
            if 'AUSF' in context.network_functions and 'AMF' in context.network_functions:
                return 'AMF', 'AUSF'
        
        elif 'registration' in procedure_lower:
            # Registration typically involves UE -> AMF
            if 'AMF' in context.network_functions:
                return 'UE' if 'UE' in context.network_functions else None, 'AMF'
        
        # Method 4: Fallback - ensure all messages have SEND_BY relationships
        if len(context.network_functions) >= 2:
            # Default: first NF sends to second NF
            return context.network_functions[0], context.network_functions[1]
        elif len(context.network_functions) == 1:
            # Single NF - could be sender or receiver
            return context.network_functions[0], None
        
        return None, None
    
    def _nf_involved_in_step(self, nf: str, step_name: str, context: ProcedureContext) -> bool:
        """Determine if network function is involved in a specific step."""
        # Simple heuristic - can be enhanced with LLM analysis
        step_lower = step_name.lower()
        nf_lower = nf.lower()
        
        # Check if NF acronym appears in step name
        if nf_lower in step_lower:
            return True
        
        # Check procedure-specific patterns
        procedure_lower = context.procedure_name.lower()
        
        if 'authentication' in procedure_lower:
            if nf in ['AUSF', 'AMF'] and 'auth' in step_lower:
                return True
        
        if 'registration' in procedure_lower:
            if nf == 'AMF' and 'regist' in step_lower:
                return True
        
        # Default: assume all NFs are involved in all steps (can be refined)
        return True
    
    def _filter_nfs_by_step_involvement(self, context: ProcedureContext) -> List[str]:
        """Filter NFs that have no step involvement evidence."""
        validated_nfs = []
        rejected_nfs = []
        
        for nf in context.network_functions:
            has_step_involvement = False
            
            # Check if NF appears in any step description
            for step_name in context.steps:
                step_desc = context.step_descriptions.get(step_name, "").lower()
                if nf.lower() in step_desc:
                    has_step_involvement = True
                    print(f"          Found {nf} in {step_name}: {step_desc[:50]}...")
                    break
            
            # Check if NF appears in procedural context (not cross-references)
            if not has_step_involvement:
                has_step_involvement = self._check_procedural_context(nf, context)
            
            if has_step_involvement:
                validated_nfs.append(nf)
                print(f"        ✓ {nf}: has step involvement")
            else:
                rejected_nfs.append(nf)
                print(f"        ✗ {nf}: no step involvement found")
        
        if rejected_nfs:
            print(f"        📊 Filtered out {len(rejected_nfs)} NFs without step involvement: {', '.join(rejected_nfs)}")
        
        return validated_nfs
    
    def _check_procedural_context(self, nf: str, context: ProcedureContext) -> bool:
        """Check if NF appears in active procedural context (not cross-references)."""
        section_text = context.section.text.lower()
        nf_lower = nf.lower()
        
        # Split text into sentences
        sentences = re.split(r'[.!?;]', section_text)
        
        for sentence in sentences:
            if nf_lower in sentence:
                # Skip cross-reference sentences
                if any(ref_phrase in sentence for ref_phrase in [
                    'see section', 'refer to', 'as described in', 'clause', 'figure',
                    'for more details', 'as specified', 'according to', 'see clause'
                ]):
                    continue
                
                # Check for procedural action words
                action_words = ['perform', 'execute', 'send', 'receive', 'process', 'handle',
                              'initiate', 'trigger', 'validate', 'authenticate', 'generate',
                              'establish', 'select', 'forward', 'respond', 'request', 'shall']
                
                if any(action in sentence for action in action_words):
                    print(f"          Found {nf} in procedural context: {sentence[:60]}...")
                    return True
        
        return False