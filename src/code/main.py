from __future__ import annotations

import argparse
from pathlib import Path

from config import AppConfig
from pdf_component_extractor import PdfComponentExtractor


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Extract layout components from a PDF using YOLO DocLayNet and pdfplumber."
    )
    parser.add_argument(
        "--pdf",
        type=Path,
        required=True,
        help="Path to input PDF file.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to YOLO DocLayNet model weights (.pt).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory for images + components.json.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    #TODO: Can add new filters to tweak the extraction like zoom, dpi, etc.
    
    cfg = AppConfig(
        pdf_path=args.pdf,
        model_path=args.model,
        output_dir=args.output,
    )

    extractor = PdfComponentExtractor(cfg)
    components = extractor.run()

    print(f"Done. Total components detected: {len(components)}")


if __name__ == "__main__":    

    main()
