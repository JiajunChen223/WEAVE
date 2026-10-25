# WEAVE

**WEAVE: Observation-Guided Whole-Spectrum Assembly for Hyperspectral Image Fusion**

Experimental settings and supporting code for WEAVE.
The core WEAVE implementation will be released immediately upon paper acceptance.

| Directory | Purpose |
|---|---|
| `configs/` | Four-dataset settings, twelve baseline recipes, and comparison/scale/efficiency designs |
| `preprocessing/` | Data conversion, normalization and simulated observations |
| `metrics/` | Mean band-wise PSNR, Wang SSIM, SAM and ERGAS |
| `training/` | Baseline training interface, seeds, EMA and checkpoints |
| `baselines/` | Upstream source pins and method-specific training adapters |
| `data/` | Example training manifest |
| `tests/` | Synthetic utility checks |

Training uses 10,000 successful optimizer updates, seed 0, effective batch size
one, and the final checkpoint. Optimizers, losses and EMA remain method-specific;
HSC-Sampling uses cosine learning-rate decay. Upstream architectures, WEAVE core
code, model weights and datasets are not bundled. Native baseline components
must be supplied through a `module:function` factory; a PSRT factory is included.

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests
```

MIT License for the support utilities. Obtain upstream baseline code separately
under its original license. This partial release is not a full reproduction package.
