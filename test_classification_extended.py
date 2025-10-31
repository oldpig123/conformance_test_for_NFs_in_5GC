#!/usr/bin/env python3
"""Extended test for diagram classification - tests more samples."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "codebase_figure"))

from diagram_parser import DiagramParser
from data_structures import FigureMetadata

def test_extended():
    parser = DiagramParser()
    figure_dir = Path("output/fsm/figures/23502-j40_new")
    
    if not figure_dir.exists():
        print("Figure directory not found")
        return
    
    # Test first 20 figures
    figure_files = sorted(list(figure_dir.glob("figure_*.wmf")))[:20]
    
    print(f"Testing {len(figure_files)} figures...\n")
    
    results = []
    for i, fig_path in enumerate(figure_files, 1):
        file_type = fig_path.suffix[1:]
        metadata = FigureMetadata(
            caption=f"Test {i}",
            file_path=fig_path,
            file_type=file_type,
            original_index=i,
            r_id=f"rId{i}",
            target_ref=f"media/{fig_path.name}"
        )
        
        result = parser.parse_diagram(metadata)
        is_seq = result is not None
        results.append((fig_path.name, is_seq))
        
        status = "✓ SEQ" if is_seq else "✗ NOT"
        print(f"{i:2d}. {fig_path.name:20s} {status}")
    
    seq_count = sum(1 for _, is_seq in results if is_seq)
    print(f"\n{'='*50}")
    print(f"Sequence diagrams: {seq_count}/{len(results)}")
    print(f"Success rate: {seq_count/len(results)*100:.1f}%")

if __name__ == "__main__":
    test_extended()
