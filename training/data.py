"""Uniform-scene sampling from an explicitly supplied training manifest."""
import json
from pathlib import Path
import numpy as np
import torch
from preprocessing.geometry import Geometry
from preprocessing.degradation import make_pair
from preprocessing.crops import augment


class TrainingBatches:
    def __init__(self, manifest, dataset_config, seed=0):
        manifest = Path(manifest).resolve()
        self.base, self.config = manifest.parent, dataset_config
        if dataset_config.get("mode") != "simulated":
            raise ValueError("This release supports the four simulated datasets")
        self.spec = json.loads(manifest.read_text(encoding="utf-8"))
        if self.spec.get("split") != "train":
            raise ValueError("Manifest must explicitly declare split=train")
        self.records = self.spec.get("records", [])
        if not self.records or any(r.get("split") != "train" for r in self.records):
            raise ValueError("Every training record must declare split=train")
        identifiers = [r["scene_id"] for r in self.records]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("Use one record per training scene/region")
        geometry = dataset_config["spatial_observation"]
        self.geometry = Geometry(geometry["scale"], geometry["kernel_size"],
                                 geometry["sigma"], tuple(geometry["phase"]))
        self.rng = np.random.default_rng(seed)
        self.arrays, self.origins = [], []
        self.response = None
        if dataset_config["mode"] == "simulated":
            self.response = self.load(self.spec["response"])
            shape = (dataset_config["msi_channels"], dataset_config["hsi_channels"])
            if (self.response.shape != shape or not np.isfinite(self.response).all()
                    or np.iscomplexobj(self.response) or np.min(self.response) < 0
                    or not np.allclose(self.response.sum(1), 1.0, atol=1e-5)):
                raise ValueError("A fixed nonnegative row-normalized response is required")
        crop = dataset_config["training_crop"]
        for record in self.records:
            with np.load(self.path(record["path"]), allow_pickle=False) as data:
                values = {k: data[k] for k in data.files}
            key = "target" if self.response is not None else "hsi"
            image = values[key]
            if (image.dtype != np.float32 or image.ndim != 3
                    or image.shape[2] != dataset_config["hsi_channels"]
                    or not np.isfinite(image).all() or image.min() < 0 or image.max() > 1):
                raise ValueError("Normalized float32 HWC training images are required")
            if self.response is None:
                ms = values["msi"]
                expected = (image.shape[0]*self.geometry.scale,
                            image.shape[1]*self.geometry.scale, dataset_config["msi_channels"])
                if (ms.shape != expected or ms.dtype != np.float32 or not np.isfinite(ms).all()
                        or ms.min() < 0 or ms.max() > 1):
                    raise ValueError("Observed MSI and HSI grids do not match")
            for key, image_key in (("mask", key), ("msi_mask", "msi")):
                if key in values and (values[key].dtype != np.bool_
                                     or values[key].shape != values[image_key].shape[:2]):
                    raise ValueError("Spatial validity masks must be boolean HW arrays")
            origins = self.load(record["origins"])
            if (origins.ndim != 2 or origins.shape[1] != 2 or len(origins) == 0
                    or not np.issubdtype(origins.dtype, np.integer) or np.any(origins < 0)
                    or np.any(origins[:, 0]+crop > image.shape[0])
                    or np.any(origins[:, 1]+crop > image.shape[1])
                    or len(np.unique(origins, axis=0)) != len(origins)):
                raise ValueError("Distinct legal integer crop coordinates are required")
            self.arrays.append(values)
            self.origins.append(origins)

    def path(self, relative):
        if Path(relative).is_absolute():
            raise ValueError("Manifest paths must be relative")
        path = (self.base / relative).resolve()
        if not path.is_relative_to(self.base):
            raise ValueError("Manifest path leaves its directory")
        return path

    def load(self, relative):
        return np.load(self.path(relative), allow_pickle=False)

    def __iter__(self):
        return self

    def __next__(self):
        i = int(self.rng.integers(len(self.records)))
        y, x = self.origins[i][int(self.rng.integers(len(self.origins[i])))]
        crop, code = self.config["training_crop"], int(self.rng.integers(8))
        values = self.arrays[i]
        key = "target" if self.response is not None else "hsi"
        target = augment(values[key][y:y+crop, x:x+crop], code)
        mask = values.get("mask", np.ones(values[key].shape[:2], dtype=bool))
        hs_mask = augment(mask[y:y+crop, x:x+crop], code)
        if not hs_mask.all():
            raise ValueError("Declared training crop includes invalid reference pixels")
        if self.response is not None:
            pair = make_pair(target, self.response, self.geometry)
        return {k: torch.from_numpy(np.ascontiguousarray(pair[k].transpose(2, 0, 1)))[None]
                for k in ("lr_hsi", "hr_msi", "target")}
