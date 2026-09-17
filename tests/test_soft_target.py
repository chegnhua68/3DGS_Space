import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from integrations.thermal3dgs.soft_target import SoftTargetStore, mix_soft_target


class SoftTargetTests(unittest.TestCase):
    def test_mix_soft_target_is_float32_and_respects_endpoints(self):
        observed = np.array([[[0.2, 0.4, 0.6]]], dtype=np.float32)
        teacher = np.array([[[0.8, 0.6, 0.4]]], dtype=np.float32)
        np.testing.assert_array_equal(mix_soft_target(observed, teacher, 0.0), observed)
        np.testing.assert_array_equal(mix_soft_target(observed, teacher, 1.0), teacher)
        target = mix_soft_target(observed, teacher, 0.75)
        self.assertEqual(target.dtype, np.dtype("float32"))
        np.testing.assert_allclose(target, np.array([[[0.65, 0.55, 0.45]]], dtype=np.float32))

    def test_store_validates_manifest_and_returns_detached_chw_tensor(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            array = np.full((2, 3, 3), 0.25, dtype=np.float32)
            target_path = root / "target_float" / "004.jpg.npy"
            target_path.parent.mkdir()
            with target_path.open("wb") as stream:
                np.save(stream, array, allow_pickle=False)
            file_hash = hashlib.sha256(target_path.read_bytes()).hexdigest()
            tensor_hash = hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()
            manifest = {
                "schema_version": 1,
                "cache_status": "completed",
                "mode": "denoised_soft_target",
                "cache_key": "test",
                "rho": 0.75,
                "dtype": "float32",
                "layout": "HWC",
                "data_range": [0.0, 1.0],
                "train_selected": ["004.jpg"],
                "views": [{
                    "id": "004.jpg",
                    "target_relative_path": "target_float/004.jpg.npy",
                    "target_file_sha256": file_hash,
                    "target_tensor_sha256": tensor_hash,
                    "shape": [2, 3, 3],
                }],
            }
            canonical = json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
            manifest["manifest_sha256"] = hashlib.sha256(canonical).hexdigest()
            manifest_path = root / "supervision_manifest.v1.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            store = SoftTargetStore(manifest_path)
            observed = torch.zeros(3, 2, 3)
            target = store.target_for("004.jpg", observed)
            self.assertEqual(tuple(target.shape), (3, 2, 3))
            self.assertFalse(target.requires_grad)
            self.assertTrue(torch.allclose(target, torch.full_like(target, 0.25)))


if __name__ == "__main__":
    unittest.main()
