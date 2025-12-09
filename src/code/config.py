from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class AppConfig:
    """
    Simple configuration container for the PDF extraction app.
    """
    pdf_path: Path
    model_path: Path
    output_dir: Path
