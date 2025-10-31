#!/usr/bin/env python3
"""
Test script for diagram classification functionality.
Tests the DiagramParser's ability to classify sequence diagrams.
"""

import sys
from pathlib import Path

# Add codebase_figure to path
sys.path.insert(0, str(Path(__file__).parent / "codebase_figure"))

from diagram_parser import DiagramParser
from data_structures import FigureMetadata

def test_diagram_classification():
    """Test diagram classification on extracted figures."""
    
    # Initialize parser
    parser = DiagramParser()
    print("=" * 70)
    print("DIAGRAM CLASSIFICATION TEST")
    print("=" * 70)
    
    # Look for extracted figures in the output directory
    figure_dir = Path("output/fsm/figures/23502-j40_new")
    
    if not figure_dir.exists():
        print(f"\n⚠️  Figure directory not found: {figure_dir}")
        print("Please run the main pipeline first to extract figures.")
        return
    
    # Get all figure files
    figure_files = list(figure_dir.glob("figure_*.wmf")) + \
                   list(figure_dir.glob("figure_*.emf")) + \
                   list(figure_dir.glob("figure_*.png")) + \
                   list(figure_dir.glob("figure_*.jpg"))
    
    if not figure_files:
        print(f"\n⚠️  No figures found in: {figure_dir}")
        return
    
    print(f"\nFound {len(figure_files)} figures to test.\n")
    
    # Test first 5 figures as a sample
    sequence_count = 0
    non_sequence_count = 0
    
    for i, fig_path in enumerate(figure_files[:5], 1):
        print(f"\n--- Test {i}/{min(5, len(figure_files))} ---")
        
        # Create FigureMetadata
        file_type = fig_path.suffix[1:]  # Remove the dot
        metadata = FigureMetadata(
            caption=f"Test figure {i}",
            file_path=fig_path,
            file_type=file_type,
            original_index=i,
            r_id=f"rId{i}",
            target_ref=f"media/{fig_path.name}"
        )
        
        # Parse and classify
        result = parser.parse_diagram(metadata)
        
        if result is not None:
            sequence_count += 1
            print(f"    ✓ Result: SEQUENCE DIAGRAM detected")
        else:
            non_sequence_count += 1
            print(f"    ✗ Result: Not a sequence diagram")
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total tested: {min(5, len(figure_files))}")
    print(f"Sequence diagrams: {sequence_count}")
    print(f"Non-sequence diagrams: {non_sequence_count}")
    print(f"\nRemaining figures: {max(0, len(figure_files) - 5)}")
    print("=" * 70)

if __name__ == "__main__":
    test_diagram_classification()
