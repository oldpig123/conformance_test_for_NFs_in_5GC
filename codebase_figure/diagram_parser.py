from pathlib import Path
from typing import Optional, Dict, List, Tuple
from lxml import etree
import cv2
import numpy as np
import tempfile
import subprocess

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

    def _is_sequence_diagram_vector(self, figure_metadata: FigureMetadata) -> Tuple[bool, Optional[etree._Element]]:
        """
        Checks if a vector diagram (EMF/WMF) is a sequence diagram by analyzing its structure.
        
        For vector diagrams, we look for:
        1. Multiple vertical rectangles (lifelines/actors)
        2. Horizontal lines/arrows between them (messages)
        3. Text labels on the arrows
        
        Note: WMF/EMF files are not always XML. They may be binary formats.
        We'll try XML parsing first, then fall back to raster analysis if it fails.
        """
        try:
            tree = etree.parse(str(figure_metadata.file_path))
            root = tree.getroot()
            
            # Look for typical sequence diagram elements in XML
            # This is a heuristic based on common DrawingML patterns
            vertical_shapes = []
            horizontal_lines = []
            
            # Search for shape elements (namespace-agnostic)
            for elem in root.iter():
                tag_name = elem.tag.split('}')[-1].lower()
                
                # Look for rectangles/shapes (lifelines)
                if tag_name in ['rect', 'shape', 'sp']:
                    vertical_shapes.append(elem)
                
                # Look for lines/connectors (messages)
                if tag_name in ['line', 'connector', 'cxn', 'cxnsp']:
                    horizontal_lines.append(elem)
            
            # Heuristic: A sequence diagram should have at least 2 lifelines and 2 messages
            is_seq_diagram = len(vertical_shapes) >= 2 and len(horizontal_lines) >= 2
            
            if is_seq_diagram:
                print(f"    Vector XML analysis: {len(vertical_shapes)} shapes, {len(horizontal_lines)} lines")
            
            return is_seq_diagram, root
            
        except (etree.XMLSyntaxError, IOError) as e:
            # If XML parsing fails, the file is likely a binary WMF/EMF
            # Convert to raster and analyze
            print(f"    Binary vector format detected, converting to raster...")
            temp_png = self._convert_wmf_to_png(figure_metadata.file_path)
            if temp_png:
                # Create temporary metadata for the converted image
                temp_metadata = FigureMetadata(
                    caption=figure_metadata.caption,
                    file_path=temp_png,
                    file_type='png',
                    original_index=figure_metadata.original_index,
                    r_id=figure_metadata.r_id,
                    target_ref=figure_metadata.target_ref
                )
                is_seq = self._is_sequence_diagram_raster(temp_metadata)
                # Clean up temp file and directory
                try:
                    temp_png.unlink()
                    temp_png.parent.rmdir()
                except:
                    pass
                return is_seq, None
            else:
                print(f"    Could not convert vector file to raster")
                return False, None
        
        return False, None
    
    def _convert_wmf_to_png(self, wmf_path: Path) -> Optional[Path]:
        """
        Convert WMF/EMF file to PNG for raster analysis.
        Uses LibreOffice in headless mode which supports EMF/WMF formats well.
        """
        try:
            # Create temporary directory for output
            temp_dir = tempfile.mkdtemp()
            temp_dir_path = Path(temp_dir)
            
            # Use LibreOffice to convert EMF/WMF to PNG
            result = subprocess.run(
                ['libreoffice', '--headless', '--convert-to', 'png', 
                 '--outdir', str(temp_dir_path), str(wmf_path)],
                capture_output=True,
                timeout=15
            )
            
            # LibreOffice creates output with same basename but .png extension
            output_filename = wmf_path.stem + '.png'
            temp_path = temp_dir_path / output_filename
            
            if result.returncode == 0 and temp_path.exists():
                return temp_path
            else:
                error_msg = result.stderr.decode('utf-8') if result.stderr else 'Unknown error'
                print(f"    LibreOffice conversion failed: {error_msg[:200]}")
                return None
            
        except subprocess.TimeoutExpired:
            print(f"    WMF/EMF conversion timed out")
            return None
        except Exception as e:
            print(f"    WMF/EMF conversion failed: {e}")
            return None

    def _is_sequence_diagram_raster(self, figure_metadata: FigureMetadata) -> bool:
        """
        Checks if a raster diagram (PNG/JPG) is a sequence diagram using Computer Vision.
        
        Sequence diagram characteristics:
        1. Multiple vertical lines (lifelines) of similar length
        2. Multiple horizontal lines/arrows (messages) between lifelines
        3. Vertical spacing suggests temporal ordering
        """
        try:
            # Read the image
            img = cv2.imread(str(figure_metadata.file_path))
            if img is None:
                print(f"    Could not read image: {figure_metadata.file_path}")
                return False
            
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            height, width = gray.shape
            
            # Apply edge detection
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            
            # Detect lines using Hough Line Transform
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, 
                                   minLineLength=30, maxLineGap=10)
            
            if lines is None:
                print(f"    No lines detected in image")
                return False
            
            vertical_lines = []
            horizontal_lines = []
            
            for line in lines:
                x1, y1, x2, y2 = line[0]
                
                # Calculate line angle
                angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                
                # Classify as vertical (near 90 degrees) or horizontal (near 0/180 degrees)
                if 80 <= angle <= 100:  # Vertical line (±10 degrees tolerance)
                    line_length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                    # Only count substantial vertical lines
                    if line_length > height * 0.2:  # At least 20% of image height
                        vertical_lines.append((x1, y1, x2, y2, line_length))
                
                elif angle < 10 or angle > 170:  # Horizontal line (±10 degrees tolerance)
                    line_length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                    # Only count substantial horizontal lines
                    if line_length > width * 0.1:  # At least 10% of image width
                        horizontal_lines.append((x1, y1, x2, y2, line_length))
            
            # Heuristic: A sequence diagram should have:
            # - At least 2 vertical lines (lifelines)
            # - At least 3 horizontal lines (messages)
            # - More horizontal than vertical lines (messages between actors)
            has_enough_lifelines = len(vertical_lines) >= 2
            has_enough_messages = len(horizontal_lines) >= 3
            has_message_dominance = len(horizontal_lines) > len(vertical_lines)
            
            is_seq_diagram = has_enough_lifelines and has_enough_messages and has_message_dominance
            
            if is_seq_diagram or (len(vertical_lines) > 0 or len(horizontal_lines) > 0):
                print(f"    Raster CV analysis: {len(vertical_lines)} vertical lines, "
                      f"{len(horizontal_lines)} horizontal lines")
            
            return is_seq_diagram
            
        except Exception as e:
            print(f"    Error in raster analysis: {e}")
            return False
