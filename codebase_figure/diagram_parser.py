from pathlib import Path
from typing import Optional, Dict, List, Tuple
from lxml import etree
import cv2
import numpy as np
import tempfile
import subprocess
from docx import Document
from pptx import Presentation
import zipfile

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

        # Handle Visio files (from OLE objects)
        if figure_metadata.file_type in ['vsdx', 'vsd']:
            return self._parse_visio_diagram(figure_metadata)
        # Handle embedded Word/PowerPoint - extract nested diagrams
        elif figure_metadata.file_type in ['doc', 'docx', 'pptx']:
            print(f"  OLE object type '{figure_metadata.file_type}' detected (ProgID: {figure_metadata.ole_prog_id})")
            return self._parse_nested_document(figure_metadata)
        # Handle vector formats (legacy WMF/EMF - now less common with OLE extraction)
        elif figure_metadata.file_type in ['emf', 'wmf']:
            return self._parse_vector_diagram(figure_metadata)
        # Handle raster formats
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

    def _parse_visio_diagram(self, figure_metadata: FigureMetadata) -> Optional[Dict]:
        """Parses a Visio diagram by converting to PNG and analyzing."""
        print(f"  Attempting to parse Visio diagram: {figure_metadata.file_path}")
        
        # Convert Visio to PNG using LibreOffice
        png_path = self._convert_visio_to_png(figure_metadata.file_path)
        
        if not png_path:
            print(f"    ✗ Failed to convert Visio to PNG")
            return None
        
        # Analyze the converted PNG
        temp_fig_meta = FigureMetadata(
            caption=figure_metadata.caption,
            file_path=png_path,
            file_type='png',
            original_index=figure_metadata.original_index,
            r_id=figure_metadata.r_id,
            target_ref=figure_metadata.target_ref,
            ole_prog_id=figure_metadata.ole_prog_id,
            ole_object_path=figure_metadata.ole_object_path,
            is_ole_object=True
        )
        
        is_sequence = self._is_sequence_diagram_raster(temp_fig_meta)
        
        if is_sequence:
            print(f"    ✓ Classified as sequence diagram.")
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
    
    def _convert_visio_to_png(self, visio_path: Path) -> Optional[Path]:
        """
        Convert Visio file to PNG for raster analysis.
        Uses LibreOffice which supports Visio formats.
        """
        return self._convert_to_png(visio_path)
    
    def _convert_wmf_to_png(self, wmf_path: Path) -> Optional[Path]:
        """
        Convert WMF/EMF file to PNG for raster analysis.
        Uses LibreOffice in headless mode which supports EMF/WMF formats well.
        """
        return self._convert_to_png(wmf_path)
    
    def _convert_to_png(self, file_path: Path) -> Optional[Path]:
        """
        Generic converter: converts various formats (WMF/EMF/VSDX) to PNG using LibreOffice.
        """
        try:
            # Create temporary directory for output
            temp_dir = tempfile.mkdtemp()
            temp_dir_path = Path(temp_dir)
            
            # Use LibreOffice to convert to PNG
            result = subprocess.run(
                ['libreoffice', '--headless', '--convert-to', 'png', 
                 '--outdir', str(temp_dir_path), str(file_path)],
                capture_output=True,
                timeout=15
            )
            
            # LibreOffice creates output with same basename but .png extension
            output_filename = file_path.stem + '.png'
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

    def _parse_nested_document(self, figure_metadata: FigureMetadata, depth: int = 0) -> Optional[Dict]:
        """
        Extract and parse diagrams from nested Word/PowerPoint documents.
        
        Args:
            figure_metadata: Metadata of the OLE object
            depth: Current recursion depth (max 3)
        
        Returns:
            Dictionary with extracted data if sequence diagram found, None otherwise
        """
        MAX_DEPTH = 3
        
        if depth >= MAX_DEPTH:
            print(f"    ⚠️  Max nesting depth ({MAX_DEPTH}) reached, skipping")
            return None
        
        if figure_metadata.nesting_level >= MAX_DEPTH:
            print(f"    ⚠️  Max nesting level ({MAX_DEPTH}) reached in metadata")
            return None
        
        try:
            if figure_metadata.file_type == 'pptx':
                return self._extract_from_pptx(figure_metadata, depth)
            elif figure_metadata.file_type == 'docx':
                return self._extract_from_docx(figure_metadata, depth)
            elif figure_metadata.file_type == 'doc':
                return self._extract_from_doc(figure_metadata, depth)
            else:
                print(f"    ⚠️  Unsupported nested document type: {figure_metadata.file_type}")
                return None
        except Exception as e:
            print(f"    ✗ Error extracting from nested document: {e}")
            return None

    def _extract_from_pptx(self, figure_metadata: FigureMetadata, depth: int) -> Optional[Dict]:
        """
        Extract and classify diagrams from PowerPoint slides.
        Uses LibreOffice to export slides as PNG images for classification.
        """
        print(f"    Extracting diagrams from PowerPoint (depth={depth})...")
        
        try:
            prs = Presentation(figure_metadata.file_path)
            temp_dir = Path(tempfile.mkdtemp(prefix='pptx_extract_'))
            
            print(f"      Total slides: {len(prs.slides)}")
            print(f"      Converting slides to PNG using LibreOffice...")
            
            # Convert entire PPTX to PNG images using LibreOffice
            cmd = [
                'libreoffice',
                '--headless',
                '--convert-to', 'png',
                '--outdir', str(temp_dir),
                str(figure_metadata.file_path)
            ]
            
            result_cmd = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result_cmd.returncode != 0:
                print(f"      ✗ LibreOffice conversion failed: {result_cmd.stderr}")
                return None
            
            # Find converted PNG files (LibreOffice creates one PNG per slide)
            png_files = sorted(temp_dir.glob('*.png'))
            
            if not png_files:
                print(f"      ✗ No PNG files generated")
                return None
            
            print(f"      ✓ Generated {len(png_files)} slide images")
            
            found_sequence = False
            result = None
            
            # Classify each slide
            for slide_idx, png_path in enumerate(png_files):
                print(f"      Slide {slide_idx + 1}/{len(png_files)}: {png_path.name}")
                
                # Create FigureMetadata for the slide image
                slide_meta = FigureMetadata(
                    caption=f"{figure_metadata.caption} (Slide {slide_idx+1})",
                    file_path=png_path,
                    file_type='png',
                    original_index=figure_metadata.original_index,
                    r_id=f"{figure_metadata.r_id}_slide{slide_idx}",
                    target_ref=str(png_path),
                    ole_prog_id=figure_metadata.ole_prog_id,
                    ole_object_path=figure_metadata.ole_object_path,
                    is_ole_object=True,
                    nesting_level=figure_metadata.nesting_level + 1
                )
                
                # Classify the slide
                parse_result = self.parse_diagram(slide_meta)
                if parse_result:
                    print(f"        ✓ Found sequence diagram in slide {slide_idx + 1}")
                    found_sequence = True
                    result = parse_result
                    break
            
            if not found_sequence:
                print(f"      ✗ No sequence diagrams found in {len(png_files)} slides")
            
            return result
            
        except subprocess.TimeoutExpired:
            print(f"      ✗ LibreOffice conversion timed out")
            return None
        except Exception as e:
            print(f"      ✗ Error processing PowerPoint: {e}")
            return None

    def _extract_from_docx(self, figure_metadata: FigureMetadata, depth: int) -> Optional[Dict]:
        """
        Extract and classify diagrams from Word document.
        Uses hybrid approach:
        1. First try to extract embedded images from document relationships
        2. If no sequence diagrams found, export pages as PNG using LibreOffice
        """
        print(f"    Extracting diagrams from Word .docx (depth={depth})...")
        
        try:
            doc = Document(figure_metadata.file_path)
            temp_dir = Path(tempfile.mkdtemp(prefix='docx_extract_'))
            
            found_sequence = False
            result = None
            image_count = 0
            
            # Phase 1: Extract embedded images from document relationships
            print(f"      Phase 1: Extracting embedded images...")
            for rel in doc.part.rels.values():
                if "image" in rel.target_ref:
                    try:
                        image_bytes = rel.target_part.blob
                        # Determine extension from content type
                        content_type = rel.target_part.content_type
                        ext_map = {
                            'image/png': 'png',
                            'image/jpeg': 'jpg',
                            'image/emf': 'emf',
                            'image/wmf': 'wmf',
                            'image/x-emf': 'emf',
                            'image/x-wmf': 'wmf'
                        }
                        ext = ext_map.get(content_type, 'png')
                        
                        image_path = temp_dir / f"doc_img{image_count}.{ext}"
                        with open(image_path, 'wb') as f:
                            f.write(image_bytes)
                        
                        print(f"        Extracted image: {image_path.name}")
                        image_count += 1
                        
                        # Create FigureMetadata for extracted image
                        nested_meta = FigureMetadata(
                            caption=f"{figure_metadata.caption} (Nested img {image_count})",
                            file_path=image_path,
                            file_type=ext,
                            original_index=figure_metadata.original_index,
                            r_id=f"{figure_metadata.r_id}_i{image_count}",
                            target_ref=str(image_path),
                            ole_prog_id=figure_metadata.ole_prog_id,
                            ole_object_path=figure_metadata.ole_object_path,
                            is_ole_object=True,
                            nesting_level=figure_metadata.nesting_level + 1
                        )
                        
                        # Classify the extracted image
                        parse_result = self.parse_diagram(nested_meta)
                        if parse_result:
                            print(f"        ✓ Found sequence diagram in nested image {image_count}")
                            found_sequence = True
                            result = parse_result
                            break
                            
                    except Exception as e:
                        print(f"        ⚠️  Error extracting image: {e}")
                        continue
            
            # Phase 2: If no embedded images or no sequence diagrams found, export as PNG
            if not found_sequence:
                print(f"      Phase 1 result: {image_count} images extracted, no sequence diagrams")
                print(f"      Phase 2: Converting document pages to PNG using LibreOffice...")
                
                # Convert Word document to PNG (creates one PNG per page)
                cmd = [
                    'libreoffice',
                    '--headless',
                    '--convert-to', 'png',
                    '--outdir', str(temp_dir),
                    str(figure_metadata.file_path)
                ]
                
                result_cmd = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                
                if result_cmd.returncode != 0:
                    print(f"        ✗ LibreOffice conversion failed: {result_cmd.stderr}")
                    return None
                
                # Find converted PNG files
                png_files = sorted(temp_dir.glob('*.png'))
                
                if not png_files:
                    print(f"        ✗ No PNG files generated")
                    return None
                
                print(f"        ✓ Generated {len(png_files)} page images")
                
                # Classify each page
                for page_idx, png_path in enumerate(png_files):
                    print(f"        Page {page_idx + 1}/{len(png_files)}: {png_path.name}")
                    
                    # Create FigureMetadata for the page image
                    page_meta = FigureMetadata(
                        caption=f"{figure_metadata.caption} (Page {page_idx+1})",
                        file_path=png_path,
                        file_type='png',
                        original_index=figure_metadata.original_index,
                        r_id=f"{figure_metadata.r_id}_page{page_idx}",
                        target_ref=str(png_path),
                        ole_prog_id=figure_metadata.ole_prog_id,
                        ole_object_path=figure_metadata.ole_object_path,
                        is_ole_object=True,
                        nesting_level=figure_metadata.nesting_level + 1
                    )
                    
                    # Classify the page
                    parse_result = self.parse_diagram(page_meta)
                    if parse_result:
                        print(f"          ✓ Found sequence diagram on page {page_idx + 1}")
                        found_sequence = True
                        result = parse_result
                        break
            
            if not found_sequence:
                print(f"      ✗ No sequence diagrams found")
            
            return result
            
        except subprocess.TimeoutExpired:
            print(f"      ✗ LibreOffice conversion timed out")
            return None
        except Exception as e:
            print(f"      ✗ Error processing Word .docx: {e}")
            return None

    def _extract_from_doc(self, figure_metadata: FigureMetadata, depth: int) -> Optional[Dict]:
        """
        Extract images from old Word .doc format.
        Uses LibreOffice to convert to .docx first.
        """
        print(f"    Extracting diagrams from Word .doc (depth={depth})...")
        print(f"      Converting .doc to .docx using LibreOffice...")
        
        try:
            temp_dir = Path(tempfile.mkdtemp(prefix='doc_convert_'))
            
            # Convert .doc to .docx using LibreOffice
            cmd = [
                'libreoffice',
                '--headless',
                '--convert-to', 'docx',
                '--outdir', str(temp_dir),
                str(figure_metadata.file_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                print(f"      ✗ LibreOffice conversion failed: {result.stderr}")
                return None
            
            # Find the converted .docx file
            docx_files = list(temp_dir.glob('*.docx'))
            if not docx_files:
                print(f"      ✗ No .docx file generated")
                return None
            
            docx_path = docx_files[0]
            print(f"      ✓ Converted to: {docx_path.name}")
            
            # Create new metadata for the converted file
            converted_meta = FigureMetadata(
                caption=figure_metadata.caption,
                file_path=docx_path,
                file_type='docx',
                original_index=figure_metadata.original_index,
                r_id=figure_metadata.r_id,
                target_ref=str(docx_path),
                ole_prog_id=figure_metadata.ole_prog_id,
                ole_object_path=figure_metadata.ole_object_path,
                is_ole_object=True,
                nesting_level=figure_metadata.nesting_level
            )
            
            # Now extract from the converted .docx
            return self._extract_from_docx(converted_meta, depth)
            
        except subprocess.TimeoutExpired:
            print(f"      ✗ LibreOffice conversion timed out")
            return None
        except Exception as e:
            print(f"      ✗ Error converting .doc: {e}")
            return None
