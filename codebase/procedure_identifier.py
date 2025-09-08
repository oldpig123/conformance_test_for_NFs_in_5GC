import re
import json
from typing import List, Optional
from tqdm import tqdm

from data_structures import DocumentSection

class ProcedureIdentifier:
    """LLM-based procedure identification without keyword dependency."""
    
    def __init__(self, llm_pipeline=None):
        self.llm_pipeline = llm_pipeline
        self.is_t5_model = getattr(llm_pipeline, 'is_t5_model', False) if llm_pipeline else False
    
    def identify_procedure_sections(self, sections_with_figures: List[DocumentSection]) -> List[DocumentSection]:
        """Step 2a: Extract procedure names using LLM analysis."""
        print("\n=== STEP 2a: LLM-Based Procedure Identification ===")
        procedure_sections = []
        
        for section in tqdm(sections_with_figures, desc="LLM procedure identification"):
            procedure_name = self._query_llm_for_procedure_identification(section)
            
            if procedure_name:
                # Mark as procedure section
                section.procedure_name = procedure_name
                procedure_sections.append(section)
                print(f"  ✓ Identified: '{procedure_name}' in section '{section.title[:50]}...'")
        
        print(f"LLM identified {len(procedure_sections)} procedure sections")
        return procedure_sections
    
    def _query_llm_for_procedure_identification(self, section: DocumentSection) -> Optional[str]:
        """Use LLM to determine if section contains a 3GPP procedure."""
        if not self.llm_pipeline:
            return self._fallback_procedure_identification(section)
        
        # Clean title for analysis
        cleaned_title = re.sub(r'^\d+(\.\d+)*\s*', '', section.title)
        cleaned_title = re.sub(r'^[A-Z]\.\d+(\.\d+)*\s*', '', cleaned_title)
        cleaned_title = cleaned_title.strip()
        
        prompt = f"""
You are an expert in 3GPP 5G telecommunications specifications.

Task: Analyze if this section describes a specific 3GPP telecommunications procedure.

Section Title: "{cleaned_title}"
Section Content (first 800 chars): "{section.text[:800]}..."
Has Figure: {section.has_figure}
Figures Mentioned: {section.figures[:2]}

ANALYSIS CRITERIA:
1. Section must contain a figure showing step flow/process diagram
2. Title should be a specific procedure name (like "5G AKA", "Registration procedure")
3. Content describes sequential steps of a telecommunications process
4. NOT general descriptions, overviews, or requirements

EXAMPLES:
Valid procedures: "5G AKA", "Registration procedure", "Authentication procedure", "PDU Session Establishment procedure"
Invalid: "General procedures", "Overview", "Architecture", "Requirements", "Parameters"

If this IS a specific telecommunications procedure, return ONLY the procedure name.
If this is NOT a specific procedure, return "NOT_PROCEDURE".

Answer:"""

        try:
            if self.is_t5_model:
                result = self.llm_pipeline(prompt, max_length=50, num_return_sequences=1)
                response = result[0]['generated_text'].strip()
            else:
                result = self.llm_pipeline(prompt, max_length=len(prompt) + 50, num_return_sequences=1)
                response = result[0]['generated_text'][len(prompt):].strip()
            
            # Parse LLM response
            if "NOT_PROCEDURE" in response.upper():
                return None
            else:
                # Extract procedure name from response
                lines = response.split('\n')
                for line in lines:
                    line = line.strip()
                    if line and not line.startswith(('Answer:', 'Task:', 'ANALYSIS')):
                        # Clean up the response
                        cleaned_response = re.sub(r'^["\']|["\']$', '', line)
                        if 3 < len(cleaned_response) < 100:
                            return cleaned_response
                
                # Fallback to cleaned title if valid
                if 3 < len(cleaned_title) < 100:
                    return cleaned_title
                    
        except Exception as e:
            print(f"    LLM procedure identification error: {e}")
        
        return None
    
    def _fallback_procedure_identification(self, section: DocumentSection) -> Optional[str]:
        """Fallback when LLM is not available - minimal heuristics."""
        cleaned_title = re.sub(r'^\d+(\.\d+)*\s*', '', section.title)
        cleaned_title = re.sub(r'^[A-Z]\.\d+(\.\d+)*\s*', '', cleaned_title)
        cleaned_title = cleaned_title.strip()
        
        # Very basic fallback - only if contains specific procedure words
        if (3 < len(cleaned_title) < 100 and
            any(word in cleaned_title.lower() for word in [
                'aka', 'authentication', 'registration', 'establishment', 'procedure'
            ])):
            return cleaned_title
        
        return None