#!/usr/bin/env python3
"""Test OLE object extraction with diagram classification."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "codebase_figure"))

from document_loader import DocumentLoader
from diagram_parser import DiagramParser

def test_ole_with_classification():
    """Test OLE extraction with classification."""
    print("="*70)
    print("  Testing OLE Extraction + Diagram Classification")
    print("="*70)
    
    loader = DocumentLoader()
    parser = DiagramParser()
    doc_path = Path('3GPP/23502-j40_new.docx')
    
    print(f"\nLoading: {doc_path}")
    sections = loader._extract_sections_with_figures(doc_path)
    
    print(f"Total sections: {len(sections)}")
    
    # Test classification on first 10 OLE objects
    print(f"\n{'='*70}")
    print(f"  Testing Classification on First 10 OLE Objects")
    print(f"{'='*70}")
    
    count = 0
    results = {"visio_seq": 0, "visio_non_seq": 0, "word_doc": 0, "other": 0}
    
    for section in sections:
        if count >= 10:
            break
        if section.figures:
            for fig in section.figures:
                if fig.is_ole_object and count < 10:
                    count += 1
                    print(f"\n{count}. Testing: {fig.file_path.name}")
                    print(f"   ProgID: {fig.ole_prog_id}")
                    print(f"   Type: {fig.file_type}")
                    
                    # Try to classify
                    result = parser.parse_diagram(fig)
                    
                    if fig.file_type in ['vsdx', 'vsd']:
                        if result:
                            results["visio_seq"] += 1
                        else:
                            results["visio_non_seq"] += 1
                    elif fig.file_type in ['doc', 'docx', 'pptx']:
                        results["word_doc"] += 1
                    else:
                        results["other"] += 1
                    
                    print()
    
    print(f"{'='*70}")
    print(f"  Test Results Summary")
    print(f"{'='*70}")
    print(f"Visio - Sequence Diagrams: {results['visio_seq']}")
    print(f"Visio - Non-Sequence: {results['visio_non_seq']}")
    print(f"Word/PowerPoint (TODO): {results['word_doc']}")
    print(f"Other: {results['other']}")

if __name__ == "__main__":
    test_ole_with_classification()
