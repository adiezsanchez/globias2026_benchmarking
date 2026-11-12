"""Annotation consensus.

Load ``raw_data/image/to_annotate.tif`` and the annotator masks in
``raw_data/labels``. Predict nuclei with StarDist and Cellpose 3 on the CPU,
merge annotators and predictions into one ``{stack, names}`` bundle, align ids
to the first annotator mask, and review everything in Napari.

Label files are read in filename order. The first annotator file is the
reference.
"""

from __future__ import annotations

from pathlib import Path

import napari
import numpy as np

import utils

ROOT = Path.cwd()
IMAGE_PATH = ROOT / "raw_data" / "image" / "to_annotate.tif"
LABELS_DIR = ROOT / "raw_data" / "labels"
PREDICTIONS_DIR = ROOT / "predictions"


def main() -> None:
    """Run the annotation-consensus pipeline and open Napari."""
    utils.use_cpu()
    image = utils.load_image(IMAGE_PATH)

    # Predict nuclei (StarDist fluorescent-nuclei + Cellpose 3 nuclei, CPU).
    # Predictions are written to predictions/ and reused unless force=True.
    predictions = utils.predict_nuclei(image, IMAGE_PATH.name, PREDICTIONS_DIR)

    # Collect annotator labels as a LabelBundle; omit non-2D / mismatched planes.
    annotator_labels = utils.load_label_stack(LABELS_DIR)

    # Merge predictions into the label stack.
    prediction_labels = utils.label_bundle_from_mapping(predictions)
    labels = utils.combine_label_bundles(annotator_labels, prediction_labels)

    # Align label ids to the first annotator plane (reference).
    aligned = utils.align_label_stack(labels)

    # Inter-annotation uncertainty from annotator masks alone.
    annotator_aligned = utils.align_label_stack(annotator_labels)
    agreement = utils.label_agreement_map(annotator_aligned)
    masked = np.where(agreement == 1, 0.0, agreement)  # full agreement → 0
    uncertainty = 1.0 - masked
    uncertainty = np.where(uncertainty == 1, 0.0, uncertainty)  # former background → 0

    # Review in Napari: image, aligned masks, uncertainty, 3D volume.
    viewer = utils.show_in_napari(
        image,
        IMAGE_PATH.name,
        aligned,
    )
    viewer.add_image(uncertainty, colormap="inferno")

    # Height field: each (y, x) fills z-planes [0, height)
    heights = np.clip(np.rint(np.asarray(uncertainty, dtype=float) * 100), 0, 100).astype(
        np.int32
    )
    z = np.arange(100, dtype=np.int32)[:, None, None]
    # Voxel value = depth (1..100) so the colormap reads as a depth map; 0 = empty
    uncertainty_volume = np.where(z < heights, z + 1, 0).astype(np.float32)

    layer = viewer.add_image(
        uncertainty_volume,
        name="uncertainty_volume",
        colormap="inferno",
        rendering="mip",
        opacity=0.5,
        contrast_limits=(0, 100),
    )
    layer.colorbar.visible = True

    # 3D camera
    viewer.dims.ndisplay = 3
    cam = viewer.camera
    cam.orientation = ("towards", "down", "right")
    cam.zoom = 2.72
    cam.perspective = 45
    cam.angles = (-22, -7, -35)  # Z, Y, X
    cam.synced = True

    fig = utils.uncertainty_bars_3d(uncertainty * 100, threshold=0.0, stride=1)
    fig.show()

    napari.run()


if __name__ == "__main__":
    main()
