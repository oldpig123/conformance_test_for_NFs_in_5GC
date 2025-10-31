from pathlib import Path
from typing import Optional, Dict, List
from lxml import etree

from data_structures import FigureMetadata

class DiagramParser:
    """Parses diagram files (images) to extract structured information."""

    def __init__(self):
        pass

    def parse_diagram(self, figure_metadata: FigureMetadata) -> Optional[Dict]:
        """
        Parses a diagram and returns a dictionary with extracted information
        if it is a sequence diagram, otherwise returns None.
        """
        print(f"Parsing diagram: {figure_metadata.file_path}")

        if figure_metadata.file_type in ['emf', 'wmf']:
            return self._parse_vector_diagram(figure_metadata)
        elif figure_metadata.file_type in ['png', 'jpg', 'jpeg']:
            return self._parse_raster_diagram(figure_metadata)
        else:
            print(f"  Unsupported diagram type: {figure_metadata.file_type}")
            return None

    def _parse_vector_diagram(self, figure_metadata: FigureMetadata) -> Optional[Dict]:
        """Parses a vector diagram (EMF/WMF) by analyzing its XML structure."""
        print(f"  Attempting to parse vector diagram: {figure_metadata.file_path}")
        
        is_sequence, root = self._is_sequence_diagram_vector(figure_metadata)
        
        if is_sequence:
            print(f"    ✓ Classified as sequence diagram.")
            # Placeholder for entity/relation extraction from vector diagram
            return {
                "network_functions": [],
                "messages": [],
                "sequence": []
            }
        else:
            print(f"    ✗ Not a sequence diagram.")
            return None

    def _parse_raster_diagram(self, figure_metadata: FigureMetadata) -> Optional[Dict]:
        """Parses a raster diagram (PNG/JPG) using CV and OCR."""
        print(f"  Attempting to parse raster diagram: {figure_metadata.file_path}")
        # Placeholder for raster parsing logic
        # In a real implementation, we would use OpenCV and Tesseract here.
        is_sequence = self._is_sequence_diagram_raster(figure_metadata)

        if is_sequence:
            print(f"    ✓ Classified as sequence diagram.")
            # Placeholder for entity/relation extraction from raster diagram
            return {
                "network_functions": [],
                "messages": [],
                "sequence": []
            }
        else:
            print(f"    ✗ Not a sequence diagram.")
            return None

    def _is_sequence_diagram_vector(self, figure_metadata: FigureMetadata) -> (bool, Optional[etree._Element]):
        """Checks if a vector diagram is a sequence diagram based on its XML structure."""
        try:
            tree = etree.parse(str(figure_metadata.file_path))
            root = tree.getroot()
            # A very basic heuristic: if it has a root element, it's a valid XML.
            # Real logic would inspect for shapes, lines, etc.
            if root is not None:
                return True, root
        except (etree.XMLSyntaxError, IOError) as e:
            print(f"    Could not parse XML for {figure_metadata.file_path}: {e}")
            return False, None
        return False, None

    def _is_sequence_diagram_raster(self, figure_metadata: FigureMetadata) -> bool:
        """Checks if a raster diagram is a sequence diagram using CV techniques."""
        # Placeholder logic: In a real implementation, we would use OpenCV to
        # detect vertical lines (lifelines) and horizontal lines (messages).
        return "sequence" in figure_metadata.caption.lower() or \
               "flow" in figure_metadata.caption.lower()
