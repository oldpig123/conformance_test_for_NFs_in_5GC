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
        """Extract sections and detect figures based on clause number matching."""
        doc = docx.Document(file_path)
        sections = []
        current_section = None

        with open(f"document_loader_debug_{file_path.stem}.log", "w", encoding="utf-8") as debug_f:
            for para in doc.paragraphs:
                text = para.text.strip()

                # Log paragraph details
                is_header = self._is_section_header(para, text)
                debug_f.write(f"[Is Header: {is_header}] [Style: {para.style.name}] [Text: {text}]\n")

                if not text:
                    continue

                if self._is_section_header(para, text):
                    if current_section:
                        sections.append(current_section)

                    current_section = DocumentSection(
                        title=text,
                        text="",
                        clause=self._extract_clause_number(text),
                        document=file_path.stem,
                        has_figure=False,
                        figures=[]
                    )
                elif current_section:
                    if self._is_figure_caption_for_section(text, current_section.clause):
                        current_section.has_figure = True
                        current_section.figures.append(text)

                    current_section.text += "\n" + text

        if current_section:
            sections.append(current_section)

        return sections

    def _is_figure_caption_for_section(self, text: str, section_clause: str) -> bool:
        """
        Checks if the text is a figure caption and if the figure number
        corresponds to the section number it is in. This version is more flexible
        to handle cases like section '6.1.3.2.0' and figure '6.1.3.2-1'.
        """
        match = re.search(r'Figure\s+([\d\.-]+a?):?', text, re.IGNORECASE)
        if not match:
            return False

        figure_number = match.group(1).strip()

        # Normalize the section clause by removing trailing .0, .0a, etc.
        normalized_section_clause = re.sub(r'\.0[a-z]?$', '', section_clause)

        # Normalize the figure number by taking the part before a hyphen
        figure_base = figure_number.split('-')[0]

        # Further normalize to handle cases like '4.2.2.2.2' vs '4.2.2.2-2'
        normalized_figure_base = re.sub(r'(\d(a-z)?)-(\d(a-z)?)$', r'\1.\3', figure_base)

        return normalized_figure_base == normalized_section_clause

    def _is_section_header(self, para, text: str) -> bool:
        """Determine if paragraph is a section header."""
        return (
            para.style.name.startswith('Heading') or
            re.match(r'^H\d+$', para.style.name) or
            re.match(r'^\d+(\.\d+)*\s+', text) or
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
                    line = line.strip().strip('\"\'')
                    if line and "yes" not in line.lower() and len(line) > 3 and len(line) < 80:
                        return line
                return cleaned_title if len(cleaned_title) > 3 else None

        except Exception as e:
            print(f"    LLM error for '{cleaned_title}': {e}")

        return None

    def _enhanced_fallback_identification(self, section: DocumentSection) -> Optional[str]:
        """Figure-prioritized procedure identification to reduce false negatives."""
        cleaned_title = self._clean_section_title(section.title)
        text_lower = section.text.lower()
        title_lower = cleaned_title.lower()

        # === FIGURE-FIRST APPROACH ===

        # If section has figure, it gets MAJOR advantage
        has_figure_boost = 50 if section.has_figure else 0  # INCREASED from 10 to 50!

        # 1. Basic procedure indicators
        procedure_indicators = [
            'procedure', 'authentication', 'registration', 'establishment',
            'aka', 'handover', 'attach', 'detach', 'selection', 'update',
            'roaming', 'breakout', 'mobility', 'session', 'bearer', 'context',
            'transfer', 'allocation', 'notification', 'exposure', 'policy',
            'control', 'management', 'slice', 'discovery', 'trigger',
            'emergency', 'suspend', 'resume', 'connection', 'modification',
            'release', 'preparation', 'execution', 'phase', 'cancel'
        ]

        # 2. Telecom entities
        telecom_entities = [
            'amf', 'smf', 'upf', 'ausf', 'udm', 'udr', 'pcf', 'nrf',
            'ue', 'gnb', 'ng-ran', 'enb', 'mme', 'sgw', 'pgw', 'hss',
            'nef', 'nssf', 'smsf', 'seaf', 'n3iwf', 'tngf', 'w-agf',
            'plmn', 'snpn', 'tai', 'guti', 'supi', 'imsi', 'imei',
            'pdu', 'qos', 'qfi', 'ambr', 'dnn', 's-nssai', 'slice',
            'bearer', 'flow', 'tunnel', 'session', 'context', 'anchor'
        ]

        # 3. Step patterns
        step_patterns = [
            r'step\s+\d+', r'\d+\.\s+[A-Z]', r'[a-z]\)\s+[A-Z]',
            r'first.*?second', r'then.*?next', r'phase\s+\d+',
            r'procedure.*?follows', r'flow.*?described'
        ]

        # === SCORING (Figure-Heavy) ===

        score = 0
        evidence = []

        # Score 1: FIGURE PRESENCE (MASSIVE BOOST!)
        score += has_figure_boost
        if has_figure_boost > 0:
            evidence.append("has_figure")

        # Score 2: Title indicators (0-20 points) - reduced from 30
        title_indicator_count = sum(1 for indicator in procedure_indicators if indicator in title_lower)
        title_score = min(title_indicator_count * 7, 20)  # Reduced multiplier
        score += title_score
        if title_score > 0:
            evidence.append(f"title_indicators({title_indicator_count})")

        # Score 3: Telecom entities (0-20 points) - reduced from 25
        entity_count = sum(1 for entity in telecom_entities if entity in text_lower)
        entity_score = min(entity_count * 2, 20)  # Simplified calculation
        score += entity_score
        if entity_score > 0:
            evidence.append(f"entities({entity_count})")

        # Score 4: Steps (0-15 points) - reduced from 20
        step_matches = sum(1 for pattern in step_patterns if re.search(pattern, text_lower))
        step_score = min(step_matches * 3, 15)  # Reduced multiplier
        score += step_score
        if step_score > 0:
            evidence.append(f"steps({step_matches})")

        # Score 5: Basic content quality (0-10 points)
        content_score = 0
        if len(section.text) > 1000:
            content_score = 10
        elif len(section.text) > 500:
            content_score = 5
        score += content_score
        if content_score > 0:
            evidence.append(f"content({len(section.text)})")

        # === FIGURE-FRIENDLY THRESHOLDS ===

        text_length = len(section.text)

        if section.has_figure:
            # MUCH LOWER thresholds for sections with figures
            if text_length > 5000:
                threshold = 60  # Even long sections with figures get lower threshold
            elif text_length > 2000:
                threshold = 55
            elif text_length > 1000:
                threshold = 50
            else:
                threshold = 45
        else:
            # Higher thresholds for sections without figures
            threshold = 80  # Much harder to qualify without figure

        # === RELAXED QUALITY FILTERS ===

        quality_filters_passed = True

        # Filter 1: VERY RELAXED exclusion for figures
        if section.has_figure:
            # Almost no exclusions for sections with figures
            hard_exclusions = ['overview', 'introduction', 'background']  # Removed 'general', 'architecture', 'summary'
            if any(pattern in title_lower for pattern in hard_exclusions) and score < threshold + 30:  # Much more lenient
                quality_filters_passed = False
        else:
            # Stricter for non-figure sections
            exclusion_patterns = ['overview', 'general', 'architecture', 'introduction', 'background', 'summary']
            if any(pattern in title_lower for pattern in exclusion_patterns):
                quality_filters_passed = False

        # Filter 2: Minimal content requirement (RELAXED)
        if text_length < 300:  # Reduced from 500
            quality_filters_passed = False

        # Filter 3: Very lenient title length (RELAXED)
        title_length = len(cleaned_title.split())
        if title_length > 25 or title_length < 1:  # Increased from 20
            quality_filters_passed = False

        # === FINAL DECISION (Figure-Friendly) ===

        # Special case: If has figure and minimal criteria, FORCE ACCEPT
        if section.has_figure and entity_count >= 2 and text_length > 500:
            print(f"    ✓ Figure-Priority: '{cleaned_title}' (score: {score}, entities: {entity_count}, has_figure)")
            return section.title

        # Normal decision
        if score >= threshold and quality_filters_passed:
            print(f"    ✓ Identified: '{cleaned_title}' (score: {score}, threshold: {threshold}, evidence: {', '.join(evidence)})")
            return section.title
        else:
            # Show rejections for debugging
            if section.has_figure and score >= 40:  # Show figure rejections
                print(f"    ✗ Figure-Rejected: '{cleaned_title[:50]}...' (score: {score}, threshold: {threshold}, quality: {quality_filters_passed})")
            elif score >= threshold - 15:  # Show close calls
                print(f"    ✗ Rejected: '{cleaned_title[:50]}...' (score: {score}, threshold: {threshold}, quality: {quality_filters_passed})")
            return None

    def _clean_section_title(self, title: str) -> str:
        """Clean section title while preserving section numbers for uniqueness."""
        # Keep section numbers but clean up formatting
        cleaned = re.sub(r'\s+', ' ', title)  # Normalize whitespace
        cleaned = re.sub(r'\t', ' ', cleaned)  # Replace tabs
        return cleaned.strip()