#!/usr/bin/env python3
"""Test OLE object extraction from 3GPP documents."""

import sys
from pathlib import Path

# Add codebase_figure to path
sys.path.insert(0, str(Path(__file__).parent / "codebase_figure"))

from document_loader import DocumentLoader

def test_ole_extraction():
    """Test OLE object extraction."""
    print("="*70)
    print("  Testing OLE Object Extraction")
    print("="*70)
    
    loader = DocumentLoader()
    doc_path = Path('3GPP/23502-j40_new.docx')
    
    print(f"\nLoading: {doc_path}")
    sections = loader._extract_sections_with_figures(doc_path)
    
    print(f"\nTotal sections: {len(sections)}")
    
    # Analyze figures
    ole_count = 0
    non_ole_count = 0
    prog_id_types = {}
    
    for section in sections:
        if section.figures:
            for fig in section.figures:
                if fig.is_ole_object:
                    ole_count += 1
                    prog_id = fig.ole_prog_id or "Unknown"
                    prog_id_types[prog_id] = prog_id_types.get(prog_id, 0) + 1
                else:
                    non_ole_count += 1
    
    print(f"\n{'='*70}")
    print(f"  Extraction Results")
    print(f"{'='*70}")
    print(f"OLE Objects extracted: {ole_count}")
    print(f"Non-OLE figures: {non_ole_count}")
    
    if prog_id_types:
        print(f"\nOLE Object Types (by ProgID):")
        for prog_id, count in sorted(prog_id_types.items(), key=lambda x: -x[1]):
            print(f"  {prog_id}: {count}")
    
    # Show first 5 OLE objects
    print(f"\n{'='*70}")
    print(f"  First 5 OLE Objects")
    print(f"{'='*70}")
    count = 0
    for section in sections:
        if count >= 5:
            break
        if section.figures:
            for fig in section.figures:
                if fig.is_ole_object and count < 5:
                    count += 1
                    print(f"\n{count}. Figure {fig.original_index}:")
                    print(f"   ProgID: {fig.ole_prog_id}")
                    print(f"   File: {fig.file_path.name}")
                    print(f"   Type: {fig.file_type}")
                    print(f"   Section: {section.title[:60]}...")

if __name__ == "__main__":
    test_ole_extraction()
