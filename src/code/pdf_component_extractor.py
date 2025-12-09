from __future__ import annotations
import io
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List

import fitz  # PyMuPDF
import numpy as np
from PIL import Image
from ultralytics import YOLO

from config import AppConfig
import pytesseract

@dataclass
class BoundingBox:
    """
    Axis-aligned bounding box in *image* coordinates (pixels).
    """
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    def as_tuple(self) -> tuple[float, float, float, float]:
        return self.x_min, self.y_min, self.x_max, self.y_max


@dataclass
class DetectedComponent:
    """
    One detected component (figure, caption, text region, etc.) in the PDF.
    """
    page_index: int
    label: str
    bbox: BoundingBox
    text: str | None = None
    image_name: str | None = None
    caption: str | None = None  

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["bbox"] = {
            "x_min": self.bbox.x_min,
            "y_min": self.bbox.y_min,
            "x_max": self.bbox.x_max,
            "y_max": self.bbox.y_max,
        }
        return data


class PdfComponentExtractor:
    """
    End-to-end pipeline:

    - Render each page to an image (PyMuPDF).
    - Run YOLO DocLayNet to detect layout components.
    - For each component:
        - If it's a Picture, crop and save as PNG.
        - For any component, extract text in that bounding box using pytesseract.
        - If the component is a Picture, find the nearest Text component below it as caption.
    - Save a JSON summary of all detected components.
    """

    def __init__(self, config: AppConfig) -> None:
        """
        Create a new extractor using the provided configuration.

        Parameters
        ----------
        config:
            Application configuration, including PDF path, YOLO model path, and output directory.
        """
        self.config = config
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        (self.config.output_dir / "images").mkdir(parents=True, exist_ok=True)

        self._model = YOLO(str(self.config.model_path))

    def _open_documents(self) -> fitz.Document:
        """
        Open the same PDF via PyMuPDF (for rendering).
        """
        fitz_doc = fitz.open(self.config.pdf_path)
        return fitz_doc

    def _render_page_to_image(self, doc: fitz.Document, page_index: int) -> fitz.Page:
        """
        Render a PDF page to a PIL image using PyMuPDF.

        Parameters
        ----------
        doc:
            Open PyMuPDF document.
        page_index:
            Zero-based page index.

        Returns
        -------
        fitz.Page
            Rendered page object.
        """
        page = doc.load_page(page_index)

        return page

    def _run_yolo_on_image(
        self,
        page: fitz.Page,
        page_index: int,
    ) -> List[DetectedComponent]:
        """
        Run YOLO DocLayNet model on a page image and return detected components.
        """

        pix = page.get_pixmap(dpi=300)
        img_bytes = pix.tobytes("png")
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        np_img = np.array(pil_img)

        results = self._model(
            source=np_img,
        )
        detections = results[0].boxes.xyxy.cpu().numpy()
        if not results:
            return []

        components: List[DetectedComponent] = []

        for label, box in zip(results[0].boxes.cls.tolist(), detections):
            label = results[0].names[label]
            x_min, y_min, x_max, y_max = map(float, box[:4])
            bbox = BoundingBox(
                x_min=x_min,    
                y_min=y_min,    
                x_max=x_max,
                y_max=y_max,
            )
            
            if label == "Picture":
                print(f"Detected Picture at {bbox.as_tuple()} on page {page_index + 1}")
                # save the cropped image
                cropped = pil_img.crop((x_min, y_min, x_max, y_max))
                images_dir = self.config.output_dir / "images"  
                out_name = f"page{page_index + 1:02d}_picture{len(components):02d}.png"
                out_path = images_dir / out_name
                cropped.save(out_path)
                print(f"Saved cropped picture to {out_path}")
                components.append(
                    DetectedComponent(
                        page_index=page_index,
                        label=label,
                        bbox=bbox,
                        image_name = out_name,
                    )
                )
            else:
                print(f"Detected {label} at {bbox.as_tuple()} on page {page_index + 1}")
                cropped = pil_img.crop((x_min, y_min, x_max, y_max))
                text = pytesseract.image_to_string(cropped)
                components.append(
                    DetectedComponent(
                        page_index=page_index,
                        label=label,
                        bbox=bbox,
                        text=text.strip() if text.strip() else None,
                    )
                )

        return components

    def _save_picture(
        self,
        image: Image.Image,
        component: DetectedComponent,
        index_on_page: int,
    ) -> Path:
        """
        Crop a Picture component from the page image and save to disk.

        Returns
        -------
        Path to the saved PNG file.
        """
        x_min, y_min, x_max, y_max = component.bbox.as_tuple()
        crop = image.crop((x_min, y_min, x_max, y_max))

        images_dir = self.config.output_dir / "images"
        out_name = f"page{component.page_index + 1:02d}_picture{index_on_page + 1:02d}.png"
        out_path = images_dir / out_name
        crop.save(out_path)
        return out_path

    def run(self) -> List[DetectedComponent]:
        """
        Execute the full pipeline on the configured PDF.

        Returns
        -------
        list of DetectedComponent
            All detected components with extracted text (where available).
        """
        fitz_doc = self._open_documents()

        all_components: List[DetectedComponent] = []

        try:
            num_pages = len(fitz_doc)
            for page_index in range(num_pages):
                print(f"Processing page {page_index + 1}/{num_pages}...")
                page_image = self._render_page_to_image(fitz_doc, page_index)

                comps_on_page = self._run_yolo_on_image(page_image, page_index)
                for comp in comps_on_page:
                    # tag the smallest distance from x_max, y_max of image to x_max y_max of text on page as the image caption

                    if comp.label == "Picture":
                        min_distance = float('inf')
                        caption = None
                        for other_comp in comps_on_page:
                            if other_comp.label == "Text":
                                distance =  abs((comp.bbox.y_max - other_comp.bbox.y_max) )
                                if distance < min_distance:
                                    min_distance = distance
                                    caption = other_comp.text
                        comp.caption = caption
                
                # TODO: Filter out cropped images of bigger images if they exist
                # TODO: Some tables are identified as pictures, handle that case
                # TODO: Check for more edge cases and fix them
            
                all_components.extend(comps_on_page)

        finally:
            fitz_doc.close()

        # Save components to JSON for downstream use
        self._save_components_json(all_components)
        return all_components

    def _save_components_json(self, components: List[DetectedComponent]) -> None:
        """
        Save a JSON file summarizing all detected components.
        """
        import json

        data = [c.to_dict() for c in components]
        out_path = self.config.output_dir / "components.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"Saved components metadata to {out_path}")
