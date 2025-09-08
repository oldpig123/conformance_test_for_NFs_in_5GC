import re
import json
from pathlib import Path
from typing import List, Optional
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

try:
    import docx
except ImportError:
    print("Error: python-docx not installed. Run: pip install python-docx")
    exit(1)

from data_structures import DocumentSection

class DocumentLoader:
    """Handles loading and parsing of 3GPP documents."""
    
    def __init__(self, llm_pipeline=None):
        self.llm_pipeline = llm_pipeline
        self.is_t5_model = getattr(llm_pipeline, 'model', None) and 't5' in str(llm_pipeline.model.__class__).lower()
        
        self.figure_patterns = [
            r'[Ff]igure\s+\d+',
            r'[Ff]ig\.\s*\d+',
            r'see\s+[Ff]igure',
            r'shown\s+in\s+[Ff]igure',
            r'[Ff]low\s+chart',
            r'procedure\s+flow',
            r'step\s+flow'
        ]
        
        # Known 3GPP procedures for validation
        self.known_procedures = [
            '5g aka', 'registration procedure', 'authentication procedure',
            'pdu session establishment', 'service request', 'handover',
            'initial registration', 'mobility registration', 'periodic registration',
            'deregistration', 'ue configuration update', 'network slice selection',
            'primary authentication', 'secondary authentication', 'eap-aka',
            'n1 mode', 'n3iwf', 'untrusted non-3gpp', 'trusted non-3gpp'
        ]
    
    def load_documents(self, file_paths: List[Path]) -> List[DocumentSection]:
        """Step 1: Load and parse multiple 3GPP documents."""
        print("\n=== STEP 1: Loading 3GPP Documents ===")
        all_sections = []
        
        for file_path in tqdm(file_paths, desc="Loading documents"):
            try:
                sections = self._extract_sections_with_figures(file_path)
                all_sections.extend(sections)
                print(f"  ✓ Loaded {len(sections)} sections from {file_path.name}")
            except Exception as e:
                print(f"  ✗ Failed to load {file_path.name}: {e}")
        
        print(f"Total: {len(all_sections)} sections from {len(file_paths)} documents")
        return all_sections
    
    def _extract_sections_with_figures(self, file_path: Path) -> List[DocumentSection]:
        """Extract sections and detect figures."""
        doc = docx.Document(file_path)
        sections = []
        current_section = DocumentSection(
            title="Introduction",
            text="",
            clause="0",
            document=file_path.stem,
            has_figure=False,
            figures=[]
        )
        
        for para in doc.paragraphs:
            text = para.text.strip()
            if not text:
                continue
            
            # Check for figures
            if self._contains_figure_reference(text):
                current_section.has_figure = True
                current_section.figures.append(text)
            
            # Detect section headers
            if self._is_section_header(para, text):
                if current_section.text.strip():
                    sections.append(current_section)
                
                current_section = DocumentSection(
                    title=text,
                    text="",
                    clause=self._extract_clause_number(text),
                    document=file_path.stem,
                    has_figure=False,
                    figures=[]
                )
            else:
                current_section.text += " " + text
        
        if current_section.text.strip():
            sections.append(current_section)
        
        return sections
    
    def _contains_figure_reference(self, text: str) -> bool:
        """Detect if text contains figure references."""
        return any(re.search(pattern, text) for pattern in self.figure_patterns)
    
    def _is_section_header(self, para, text: str) -> bool:
        """Determine if paragraph is a section header."""
        return (
            para.style.name.startswith('Heading') or
            re.match(r'^\d+(\.\d+)*\s+[A-Z]', text) or
            re.match(r'^[A-Z]\.\d+(\.\d+)*\s+[A-Z]', text)
        )
    
    def _extract_clause_number(self, text: str) -> str:
        """Extract clause number from section header."""
        match = re.match(r'^(\d+(\.\d+)*|[A-Z](\.\d+)*)', text)
        return match.group(1) if match else "unknown"
    
    def identify_procedure_sections_with_llm(self, sections: List[DocumentSection]) -> List[DocumentSection]:
        """Step 4b: Use enhanced LLM + fallback to identify procedure sections."""
        print(f"\n=== STEP 4B: Enhanced Procedure Identification ===")
        procedure_sections = []
        
        # Filter: only sections with figures (Requirement 2)
        sections_with_figures = [s for s in sections if s.has_figure]
        print(f"Found {len(sections_with_figures)} sections with figures")
        
        # Debug: Show some section titles
        print(f"Sample section titles with figures:")
        for i, section in enumerate(sections_with_figures[:10]):
            cleaned_title = self._clean_section_title(section.title)
            print(f"  {i+1}. '{cleaned_title}' (text length: {len(section.text)})")
        
        # Try different identification methods
        llm_identified = 0
        fallback_identified = 0
        
        for section in tqdm(sections_with_figures, desc="Procedure identification"):
            # Method 1: Try LLM identification
            procedure_name = self._query_llm_for_procedure_identification(section)
            
            # Method 2: If LLM fails, use enhanced fallback
            if not procedure_name:
                procedure_name = self._enhanced_fallback_identification(section)
                if procedure_name:
                    fallback_identified += 1
            else:
                llm_identified += 1
            
            if procedure_name:
                section.is_procedure = True
                section.procedure_name = procedure_name
                procedure_sections.append(section)
                print(f"  ✓ Procedure: '{procedure_name}' (from {section.title[:50]}...)")
        
        print(f"Identification results:")
        print(f"  LLM identified: {llm_identified}")
        print(f"  Fallback identified: {fallback_identified}")
        print(f"  Total procedures: {len(procedure_sections)}")
        
        return procedure_sections
    
    def _query_llm_for_procedure_identification(self, section: DocumentSection) -> Optional[str]:
        """Use simplified LLM prompt for procedure identification."""
        if not self.llm_pipeline:
            return None
        
        cleaned_title = self._clean_section_title(section.title)
        
        # Simplified prompt for better success rate
        prompt = f"""
Is this a 3GPP telecommunications procedure?

Title: "{cleaned_title}"
Has Figure: Yes
Text: "{section.text[:300]}..."

A procedure describes step-by-step telecommunications process.

Examples of procedures:
- "5G AKA"
- "Registration procedure"  
- "Authentication procedure"
- "PDU Session Establishment"

Examples of NOT procedures:
- "Overview"
- "Architecture"
- "General"

Answer: YES or NO

If YES, what is the procedure name?
"""

        try:
            if self.is_t5_model:
                result = self.llm_pipeline(prompt, max_length=30, num_return_sequences=1)
                response = result[0]['generated_text'].strip()
            else:
                result = self.llm_pipeline(prompt, max_length=len(prompt) + 30, num_return_sequences=1)
                response = result[0]['generated_text'][len(prompt):].strip()
            
            # Debug: Print LLM response for first few sections
            if hasattr(self, '_debug_count') and self._debug_count < 3:
                print(f"    DEBUG - LLM response for '{cleaned_title}': {response[:100]}...")
                self._debug_count += 1
            elif not hasattr(self, '_debug_count'):
                self._debug_count = 1
            
            # Parse response
            response_lower = response.lower()
            if "yes" in response_lower and "no" not in response_lower:
                # Extract procedure name from response or use title
                lines = response.split('\n')
                for line in lines:
                    line = line.strip().strip('"\'')
                    if line and "yes" not in line.lower() and len(line) > 3 and len(line) < 80:
                        return line
                return cleaned_title if len(cleaned_title) > 3 else None
            
        except Exception as e:
            print(f"    LLM error for '{cleaned_title}': {e}")
        
        return None
    
    def _enhanced_fallback_identification(self, section: DocumentSection) -> Optional[str]:
        """Enhanced fallback procedure identification."""
        cleaned_title = self._clean_section_title(section.title)
        text_lower = section.text.lower()
        title_lower = cleaned_title.lower()
        
        # Method 1: Check against known procedures
        for known_proc in self.known_procedures:
            if known_proc in title_lower:
                return cleaned_title
        
        # Method 2: Check for procedure indicators in title
        procedure_indicators = [
            'procedure', 'authentication', 'registration', 'establishment',
            'aka', 'handover', 'attach', 'detach', 'selection', 'update'
        ]
        
        title_has_indicator = any(indicator in title_lower for indicator in procedure_indicators)
        
        # Method 3: Check for step-like content in text
        step_patterns = [
            r'step\s+\d+',
            r'\d+\.\s+[A-Z]',
            r'[a-z]\)\s+[A-Z]',
            r'first.*second.*third',
            r'then.*next.*finally'
        ]
        
        has_steps = any(re.search(pattern, text_lower) for pattern in step_patterns)
        
        # Method 4: Check for telecommunications entities
        telecom_entities = [
            'amf', 'smf', 'upf', 'ausf', 'udm', 'udr', 'pcf', 'nrf',
            'ue', 'gnb', 'ng-ran', 'n1', 'n2', 'n3', 'n4', 'n6'
        ]
        
        has_telecom_entities = sum(1 for entity in telecom_entities if entity in text_lower) >= 2
        
        # Decision logic
        if (title_has_indicator and has_steps and 
            len(cleaned_title.split()) <= 8 and 
            len(cleaned_title) < 100):
            return cleaned_title
        
        if (has_telecom_entities and has_steps and 
            len(cleaned_title.split()) <= 6 and
            'overview' not in title_lower and 'general' not in title_lower):
            return cleaned_title
        
        # Method 5: Pattern-based identification for specific cases
        if re.search(r'\b(aka|authentication|registration)\b', title_lower) and len(cleaned_title) < 50:
            return cleaned_title
        
        return None
    
    def _clean_section_title(self, title: str) -> str:
        """Clean section title for procedure name extraction."""
        # Remove clause numbers
        cleaned = re.sub(r'^\d+(\.\d+)*\s*', '', title)
        cleaned = re.sub(r'^[A-Z]\.\d+(\.\d+)*\s*', '', cleaned)
        return cleaned.strip()