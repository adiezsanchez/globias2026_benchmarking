"""Open the annotation image in Napari."""

from __future__ import annotations

from pathlib import Path

import utils

IMAGE_PATH = Path("raw_data") / "image" / "to_annotate.tif"


def main() -> None:
    """Load ``to_annotate.tif`` and block on the Napari event loop."""
    utils.open_image_in_napari(IMAGE_PATH, run=True)


if __name__ == "__main__":
    main()
