import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from preprocessing.geometry import Geometry
from preprocessing.normalize import normalize
from preprocessing.convert import convert
from preprocessing.crops import legal_origins, augment
from preprocessing.degradation import spatial_degrade, make_pair
from training.data import TrainingBatches


class PreprocessingTests(unittest.TestCase):
    def test_normalization_does_not_refit(self):
        image = np.array([[[100, 200], [300, 400]]], dtype=np.uint16)
        value = normalize(image, 1000)
        np.testing.assert_allclose(value, image/1000)
        self.assertEqual(value.dtype, np.float32)
        with self.assertRaises(ValueError): normalize(image, 100)

    def test_convert_layout(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); image = np.arange(24).reshape(2, 3, 4)
            np.save(root/"source.npy", image)
            shape = convert(root/"source.npy", root/"out.npy", layout="CHW", divisor=24)
            self.assertEqual(shape, (3, 4, 2))
            np.testing.assert_allclose(np.load(root/"out.npy"), np.moveaxis(image, 0, -1)/24)

    def test_circular_blur_and_phase_against_direct_sum(self):
        rng = np.random.default_rng(4); image = rng.random((12, 16, 3))
        geometry = Geometry(2, 4, 1.0, (1, 0))
        taps = np.exp(-((np.arange(4)-1.5)**2)/2)
        taps[1:3] = 1; taps /= taps.sum()
        direct = np.zeros_like(image)
        for i in range(4):
            for j in range(4):
                direct += taps[i]*taps[j]*np.roll(image, (i-2, j-2), (0, 1))
        np.testing.assert_allclose(spatial_degrade(image, geometry), direct[1::2, ::2], atol=6e-8)

    def test_pair_response_mixing(self):
        target = np.ones((16, 16, 4), np.float32)*0.3
        response = np.ones((2, 4))/4
        pair = make_pair(target, response, Geometry(4, 8, 2.0, (2, 2)))
        self.assertEqual(pair["lr_hsi"].shape, (4, 4, 4))
        np.testing.assert_allclose(pair["hr_msi"], 0.3)

    def test_coordinates_respect_invalid_support(self):
        mask = np.ones((8, 8), bool); mask[0, 0] = False
        origins = legal_origins(mask, 4, stride=2)
        self.assertEqual(len(origins), 8)
        self.assertFalse(any(np.all(x == [0, 0]) for x in origins))

    def test_paired_augmentation_preserves_alignment(self):
        image = np.arange(25).reshape(5, 5)
        for code in range(8):
            np.testing.assert_array_equal(augment(image*2, code), augment(image, code)*2)

    def test_training_manifest_and_dataset_binding(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); image = np.random.default_rng(11).random((24, 24, 3), dtype=np.float32)
            np.savez(root/"scene.npz", target=image)
            np.save(root/"origins.npy", np.array([[0, 0], [4, 4]]))
            np.save(root/"response.npy", np.ones((2, 3))/3)
            spec = dict(split="train", response="response.npy", records=[dict(
                split="train", scene_id="one", path="scene.npz", origins="origins.npy")])
            (root/"train.json").write_text(json.dumps(spec))
            config = dict(mode="simulated", hsi_channels=3, msi_channels=2,
                          training_crop=16, spatial_observation=dict(scale=4, kernel_size=8, sigma=2, phase=[2, 2]))
            a = TrainingBatches(root/"train.json", config, seed=0)
            b = TrainingBatches(root/"train.json", config, seed=0)
            first, second = next(a), next(b)
            for key, tensor in first.items():
                np.testing.assert_array_equal(tensor.numpy(), second[key].numpy())
            spec["split"] = "not_train"
            (root/"train.json").write_text(json.dumps(spec))
            with self.assertRaises(ValueError): TrainingBatches(root/"train.json", config)


if __name__ == "__main__":
    unittest.main()
