# globias2026_benchmarking

[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD--3-green.svg)](./LICENSE)

Benchmarking scripts for the GloBIAS 2026 Dortmund edition. The workflow compares manual nuclei annotations against CPU predictions from [StarDist](https://github.com/stardist/stardist) and [Cellpose](https://github.com/MouseLand/cellpose) 3, aligns label IDs by centroid matching, and reviews the result in [Napari](https://napari.org).

<details>
<summary><h2>Environment setup</h2></summary>
1. Clone this repository:

   ```bash
   git clone https://github.com/adiezsanchez/globias2026_benchmarking
   ```

2. If you do not have Git installed, download the code as a `.zip` from the green **Code** button on GitHub.

3. Proceed with **Pixi** as the recommended environment manager.

> [!TIP]
> [Pixi](https://pixi.sh/latest/installation/) provides a reproducible environment from `pixi.toml` and `pixi.lock`. Supported platforms are **`linux-64`**, **`osx-arm64`**, and **`win-64`**. StarDist and Cellpose run on **CPU only** (no GPU acceleration).

**Install Pixi**

Windows PowerShell:

```powershell
powershell -ExecutionPolicy ByPass -c "irm -useb https://pixi.sh/install.ps1 | iex"
```

macOS / Linux:

```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

Close and reopen your terminal, then install dependencies:

```bash
cd globias2026_benchmarking && pixi install
```

</details>

<details>
<summary><h2>Workflow summary</h2></summary>

**Recommended folder layout**

```text
globias2026_benchmarking/
├── raw_data/
│   ├── image/
│   │   └── to_annotate.tif          # input fluorescence image
│   └── labels/
│       ├── to_annotate_anniek.tif   # manual annotations
│       ├── to_annotate_benjamin.tif
│       ├── to_annotate_iva.tif
│       └── to_annotate_yariv.tif
├── predictions/                     # StarDist / Cellpose outputs (created on run)
├── 01_annotation_consensus.ipynb
├── annotate.py
├── utils.py
├── pixi.toml
└── pixi.lock
```

**Minimal workflow**

1. **Annotate** — open `to_annotate.tif` in Napari and draw nuclei labels.
2. **Save labels** — store each annotator's mask under `raw_data/labels/` using the naming convention below.
3. **Consensus notebook** — run `01_annotation_consensus.ipynb` to:
   - predict nuclei with StarDist and Cellpose 3 (CPU; reuses files in `predictions/` when present);
   - collect annotator labels and merge them with the predictions;
   - align label IDs to the first annotator mask via centroid nearest-neighbor matching;
   - review the image and aligned stack in Napari.

> [!IMPORTANT]
> Save new annotations as TIFFs in **`raw_data/labels/`** with the same naming format as the existing files:
>
> ```text
> to_annotate_<annotator>.tif
> ```
>
> Examples: `to_annotate_anniek.tif`, `to_annotate_benjamin.tif`. Use your own name (or identifier) after the underscore. Labels should be integer instance masks (background = `0`).

Core helpers live in `utils.py`.

</details>

<details>
<summary><h2>Pixi tasks</h2></summary>

| Task | Command | Description |
|------|---------|-------------|
| Install environment | `pixi install` | Solve and install dependencies from `pixi.lock` |
| Annotate in Napari | `pixi run annotate` | Open `raw_data/image/to_annotate.tif` in Napari |
| Consensus notebook | `pixi run notebook` | Launch `01_annotation_consensus.ipynb` |

</details>
