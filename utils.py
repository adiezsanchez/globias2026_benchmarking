"""CPU nuclei prediction and annotation alignment.

StarDist (TensorFlow) and Cellpose 3 (PyTorch) are pinned to the CPU.
Importing this module hides CUDA devices before either library is loaded.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TypedDict

import numpy as np
import tifffile
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from skimage.measure import regionprops
from skimage.segmentation import relabel_sequential as _relabel_sequential

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

IntegerArray = NDArray[np.integer]
TIFF_SUFFIXES = {".tif", ".tiff"}


class LabelBundle(TypedDict):
    """Label images paired with the filenames they came from."""

    stack: IntegerArray
    names: list[str]


def use_cpu() -> None:
    """Keep TensorFlow and PyTorch on the CPU.

    CUDA devices are hidden at import time. This function also drops any GPU
    TensorFlow still reports and makes CPU the default PyTorch device.
    Cellpose is constructed with an explicit CPU device on top of that.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    try:
        import tensorflow as tf
        import torch
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "TensorFlow/PyTorch are missing from this interpreter. "
            "Select the 'Python (globias2026_benchmarking)' kernel, or run "
            "`pixi run jupyter notebook 01_annotation_consensus.ipynb`."
        ) from error

    tf.config.set_visible_devices([], "GPU")
    torch.set_default_device("cpu")


def load_image(path: str | Path) -> NDArray[np.generic]:
    """Read a TIFF and drop singleton axes.

    Parameters
    ----------
    path:
        Path of the TIFF file.

    Returns
    -------
    numpy.ndarray
        Image array. A single-plane TIFF is returned as a 2D array.
    """
    return np.squeeze(tifffile.imread(Path(path)))


def save_labels(labels: IntegerArray, path: str | Path) -> Path:
    """Write an integer label image as a TIFF.

    Parameters
    ----------
    labels:
        Label image. Background is ``0``.
    path:
        Destination path. Parent directories are created when missing.

    Returns
    -------
    pathlib.Path
        Path of the written file.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    array = np.asarray(labels)
    if array.size and int(array.min()) >= 0 and int(array.max()) <= np.iinfo(np.uint16).max:
        array = array.astype(np.uint16, copy=False)
    tifffile.imwrite(destination, array)
    return destination


def load_label_stack(folder: str | Path) -> LabelBundle:
    """Stack the label TIFFs in ``folder``.

    Files are read in filename order. A file that is not a 2D image, or whose
    shape differs from the first label image, is left out of the stack.

    Parameters
    ----------
    folder:
        Directory that contains one TIFF per annotator.

    Returns
    -------
    LabelBundle
        ``stack`` has shape ``(n, y, x)``; ``names`` are the matching filenames.
    """
    paths = _tiff_paths(folder)
    if not paths:
        raise FileNotFoundError(f"No TIFF label images found in {folder}.")

    planes: list[IntegerArray] = []
    names: list[str] = []
    reference_shape: tuple[int, ...] | None = None
    for path in paths:
        image = load_image(path)
        if image.ndim != 2:
            print(
                f"Removing '{path.name}' from the stack: "
                f"expected a 2D label image, got shape {image.shape}."
            )
            continue
        if reference_shape is None:
            reference_shape = image.shape
        elif image.shape != reference_shape:
            print(
                f"Removing '{path.name}' from the stack: "
                f"shape {image.shape} does not match the reference shape {reference_shape}."
            )
            continue
        planes.append(np.asarray(image))
        names.append(path.name)

    if not planes:
        raise ValueError(f"No 2D label images could be stacked from {folder}.")
    return {"stack": np.stack(planes), "names": names}


def label_bundle_from_mapping(labels_by_name: Mapping[str, IntegerArray]) -> LabelBundle:
    """Build a label bundle from a filename-to-image mapping.

    Parameters
    ----------
    labels_by_name:
        Mapping of original filenames to 2D label images. Insertion order is
        kept so prediction order is recoverable.

    Returns
    -------
    LabelBundle
        Stacked labels with the corresponding filenames.
    """
    if not labels_by_name:
        raise ValueError("Cannot build a label bundle from an empty mapping.")

    names = list(labels_by_name)
    planes = [np.asarray(labels_by_name[name]) for name in names]
    shapes = {plane.shape for plane in planes}
    if len(shapes) != 1:
        raise ValueError(f"Label images must share one shape, got {sorted(shapes)}.")
    if planes[0].ndim != 2:
        raise ValueError(f"Expected 2D label images, got shape {planes[0].shape}.")
    return {"stack": np.stack(planes), "names": names}


def combine_label_bundles(*bundles: LabelBundle) -> LabelBundle:
    """Concatenate label bundles while preserving plane-to-filename pairing.

    Parameters
    ----------
    bundles:
        One or more label bundles. The first plane of the first bundle remains
        the alignment reference when the result is passed to
        :func:`align_label_stack`.

    Returns
    -------
    LabelBundle
        Combined ``stack`` and ``names`` in the given bundle order.
    """
    if not bundles:
        raise ValueError("At least one label bundle is required.")

    names = [name for bundle in bundles for name in bundle["names"]]
    planes = [plane for bundle in bundles for plane in np.asarray(bundle["stack"])]
    if len(names) != len(planes):
        raise ValueError("Each bundle must keep stack planes aligned with names.")

    shapes = {plane.shape for plane in planes}
    if len(shapes) != 1:
        raise ValueError(f"Bundles must share one plane shape, got {sorted(shapes)}.")
    return {"stack": np.stack(planes), "names": names}


def predict_stardist(
    image: NDArray[np.generic],
    model_name: str = "2D_versatile_fluo",
) -> IntegerArray:
    """Predict nuclear instances with the CPU build of StarDist.

    Parameters
    ----------
    image:
        Single-channel 2D fluorescence image.
    model_name:
        Name of a pretrained ``StarDist2D`` model.

    Returns
    -------
    numpy.ndarray
        Instance labels with shape ``image.shape`` and background ``0``.
    """
    use_cpu()
    from csbdeep.utils import normalize
    from stardist.models import StarDist2D

    model = StarDist2D.from_pretrained(model_name)
    labels, _ = model.predict_instances(normalize(np.asarray(image)))
    return np.asarray(labels, dtype=np.int32)


def predict_cellpose(
    image: NDArray[np.generic],
    model_type: str = "nuclei",
) -> IntegerArray:
    """Predict nuclear instances with Cellpose 3 on the CPU.

    Parameters
    ----------
    image:
        Single-channel 2D image. Cellpose normalizes intensities itself.
    model_type:
        Pretrained Cellpose model. ``"nuclei"`` is the nuclear model.

    Returns
    -------
    numpy.ndarray
        Instance labels with shape ``image.shape`` and background ``0``.
    """
    use_cpu()
    import torch
    from cellpose import models

    model = models.Cellpose(
        gpu=False,
        model_type=model_type,
        device=torch.device("cpu"),
    )
    masks, _, _, _ = model.eval(
        np.asarray(image),
        diameter=None,
        channels=[0, 0],
        normalize=True,
    )
    return np.asarray(masks, dtype=np.int32)


PREDICTION_MODELS: tuple[str, ...] = ("stardist", "cellpose")


def prediction_paths(source_name: str, directory: str | Path) -> dict[str, Path]:
    """Return the expected StarDist and Cellpose prediction paths.

    Parameters
    ----------
    source_name:
        Original image filename. Its stem prefixes the prediction files.
    directory:
        Folder that stores ``{stem}_stardist.tif`` and ``{stem}_cellpose.tif``.

    Returns
    -------
    dict
        Mapping of model name to the expected TIFF path.
    """
    folder = Path(directory)
    stem = Path(source_name).stem
    return {
        model_name: folder / f"{stem}_{model_name}.tif"
        for model_name in PREDICTION_MODELS
    }


def load_predictions(
    source_name: str,
    directory: str | Path,
) -> dict[str, IntegerArray] | None:
    """Load prediction TIFFs from disk when every expected file is present.

    Parameters
    ----------
    source_name:
        Original image filename used to build the prediction filenames.
    directory:
        Folder that may already contain the prediction TIFFs.

    Returns
    -------
    dict or None
        Filename-to-label mapping when all predictions exist; otherwise
        ``None`` so the caller can recompute them.
    """
    paths = prediction_paths(source_name, directory)
    missing = [path.name for path in paths.values() if not path.is_file()]
    if missing:
        return None

    loaded: dict[str, IntegerArray] = {}
    for path in paths.values():
        loaded[path.name] = np.asarray(load_image(path), dtype=np.int32)
    return loaded


def predict_nuclei(
    image: NDArray[np.generic],
    source_name: str,
    directory: str | Path,
    *,
    force: bool = False,
) -> dict[str, IntegerArray]:
    """Run StarDist and Cellpose and store both label images.

    Files are named ``{stem}_stardist.tif`` and ``{stem}_cellpose.tif``.
    When those files already exist, they are loaded instead of recomputed.

    Parameters
    ----------
    image:
        Image passed to both predictors.
    source_name:
        Original image filename. Its stem is used in the output names.
    directory:
        Folder that receives the prediction TIFFs.
    force:
        When ``True``, recompute and overwrite even if predictions exist.

    Returns
    -------
    dict
        Mapping of prediction filename to label image.
    """
    if not force:
        cached = load_predictions(source_name, directory)
        if cached is not None:
            print(f"Loading existing predictions from {Path(directory)}.")
            return cached

    folder = Path(directory)
    predictors = {
        "stardist": predict_stardist,
        "cellpose": predict_cellpose,
    }
    saved: dict[str, IntegerArray] = {}
    for model_name, path in prediction_paths(source_name, folder).items():
        labels = predictors[model_name](image)
        save_labels(labels, path)
        saved[path.name] = labels
    return saved


def relabel_sequential(labels: IntegerArray) -> IntegerArray:
    """Renumber foreground labels to ``1..N`` in raster order.

    Parameters
    ----------
    labels:
        Label image. Zero stays background.

    Returns
    -------
    numpy.ndarray
        Relabeled image. Object shapes are unchanged.
    """
    relabeled, _, _ = _relabel_sequential(np.asarray(labels))
    return relabeled


def align_label_stack(labels: LabelBundle) -> LabelBundle:
    """Align every label image to the first image in the bundle.

    The first image is relabeled sequentially and kept as the reference.
    Each remaining image must contain the same number of objects. Its labels
    are then replaced by a one-to-one nearest-centroid match against the
    reference, so a shared id is the same nucleus. An image with a different
    object count is removed and reported.

    Parameters
    ----------
    labels:
        Bundle whose ``names`` recover the source filename of each plane.

    Returns
    -------
    LabelBundle
        Aligned labels and the filenames that remain, reference first.
    """
    planes = np.asarray(labels["stack"])
    names = list(labels["names"])
    if planes.ndim != 3:
        raise ValueError(f"Expected a label stack of shape (n, y, x), got {planes.shape}.")
    if len(names) != planes.shape[0]:
        raise ValueError(
            f"Received {len(names)} filenames for a stack of {planes.shape[0]} images."
        )
    if planes.shape[0] == 0:
        raise ValueError("Label stack is empty.")

    reference_name = names[0]
    reference = relabel_sequential(planes[0])
    reference_count = _object_count(reference)
    kept_planes = [reference]
    kept_names = [reference_name]

    for plane, name in zip(planes[1:], names[1:], strict=True):
        count = _object_count(plane)
        if count != reference_count:
            print(
                f"Removing '{name}' from the stack: it contains {count} objects, "
                f"but the reference '{reference_name}' contains {reference_count}."
            )
            continue
        kept_planes.append(_match_centroids(reference, plane))
        kept_names.append(name)

    return {
        "stack": np.stack(kept_planes).astype(np.int32, copy=False),
        "names": kept_names,
    }


def show_in_napari(
    image: NDArray[np.generic],
    image_name: str,
    aligned: LabelBundle,
    *,
    show: bool = True,
) -> object:
    """Open the input image and the aligned label stack in Napari.

    Layer names are the original filenames. The aligned stack is one labels
    layer: moving its slider renames that layer to the filename of the
    visible plane.

    Parameters
    ----------
    image:
        Input image.
    image_name:
        Filename of ``image``.
    aligned:
        Aligned label bundle. ``names`` recovers each plane's source file.
    show:
        When ``False``, create the viewer without showing the window.

    Returns
    -------
    napari.Viewer
        Viewer with the image and the aligned label stack.
    """
    import napari

    planes = np.asarray(aligned["stack"])
    names = list(aligned["names"])
    if planes.ndim != 3:
        raise ValueError(f"Expected an aligned stack of shape (n, y, x), got {planes.shape}.")
    if len(names) != planes.shape[0]:
        raise ValueError(
            f"Received {len(names)} filenames for {planes.shape[0]} aligned images."
        )

    viewer = napari.Viewer(title="Annotation consensus", show=show)
    viewer.add_image(np.asarray(image), name=image_name, colormap="gray")
    layer = viewer.add_labels(
        planes,
        name=names[0],
        metadata={"filenames": names},
    )
    _rename_with_slider(viewer, layer, names)
    return viewer


def _tiff_paths(folder: str | Path) -> list[Path]:
    """Return TIFF files in ``folder``, sorted by filename.

    Parameters
    ----------
    folder:
        Directory to scan.

    Returns
    -------
    list of pathlib.Path
        Matching files. The first path is the reference annotation.
    """
    directory = Path(folder)
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in TIFF_SUFFIXES
    )


def _object_count(labels: IntegerArray) -> int:
    """Count foreground labels.

    Parameters
    ----------
    labels:
        Label image. Zero is background and is not counted.

    Returns
    -------
    int
        Number of distinct positive labels.
    """
    values = np.unique(labels)
    return int(values.size - np.count_nonzero(values == 0))


def _centroids(labels: IntegerArray) -> tuple[NDArray[np.int32], NDArray[np.float64]]:
    """Return foreground ids and ``(row, column)`` centroids.

    Parameters
    ----------
    labels:
        Label image.

    Returns
    -------
    ids:
        Label id of each object.
    points:
        Centroid coordinates with shape ``(n, 2)``.
    """
    properties = regionprops(np.asarray(labels))
    ids = np.array([prop.label for prop in properties], dtype=np.int32)
    points = np.array([prop.centroid for prop in properties], dtype=np.float64)
    return ids, points


def _match_centroids(reference: IntegerArray, labels: IntegerArray) -> IntegerArray:
    """Replace ids in ``labels`` with the nearest reference ids.

    Matching is one-to-one: the sum of centroid distances is minimized, and
    every reference id is used once. Both images must contain the same number
    of objects.

    Parameters
    ----------
    reference:
        Reference label image. Its ids are copied onto ``labels``.
    labels:
        Label image to remap.

    Returns
    -------
    numpy.ndarray
        Remapped labels. Background stays ``0``.
    """
    reference_ids, reference_points = _centroids(reference)
    label_ids, label_points = _centroids(labels)
    if reference_ids.size == 0:
        return np.zeros_like(labels, dtype=np.int32)

    assigned_labels, assigned_reference = linear_sum_assignment(
        cdist(label_points, reference_points)
    )
    lookup = np.zeros(int(np.max(label_ids)) + 1, dtype=np.int32)
    for label_index, reference_index in zip(assigned_labels, assigned_reference, strict=True):
        lookup[int(label_ids[label_index])] = int(reference_ids[reference_index])
    return lookup[np.asarray(labels)]


def _rename_with_slider(viewer: object, layer: object, names: Sequence[str]) -> None:
    """Rename ``layer`` to the filename of the current stack plane.

    Parameters
    ----------
    viewer:
        Napari viewer that owns the stack dimension.
    layer:
        Labels layer whose data is the aligned stack.
    names:
        Filename of each plane, in stack order.
    """

    def _apply(_event: object | None = None) -> None:
        index = int(viewer.dims.current_step[0])
        layer.name = names[index]

    viewer.dims.events.current_step.connect(_apply)
    _apply()
