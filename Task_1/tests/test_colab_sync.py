# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Verify complete Drive publication and selective checkpoint restore."""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cloud.colab import checkpoint_update, restore_results, sync_file, sync_small_results


class ColabSyncTest(unittest.TestCase):
    def test_save_and_restore_latest_checkpoint_with_midpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            local, drive, resumed = (root / name for name in ("local", "drive", "resumed"))
            checkpoint_dir = local / "checkpoints" / "full_ft"
            checkpoint_dir.mkdir(parents=True)
            for update in (312, 625, 937):
                path = checkpoint_dir / f"update_{update:04d}.pt"
                path.write_bytes(f"weights-{update}".encode())
                sync_file(path, local, drive, verify=True)
            history = local / "histories" / "full_ft.csv"
            history.parent.mkdir(parents=True)
            history.write_text("update,loss\n937,0.1\n", encoding="utf-8")
            sync_small_results(local, drive)
            restored = restore_results(resumed, drive)
            self.assertEqual(checkpoint_update("full_ft", resumed), 937)
            self.assertEqual(len([p for p in restored if p.suffix == ".pt"]), 2)
            self.assertFalse((resumed / "checkpoints/full_ft/update_0312.pt").exists())
            self.assertTrue((resumed / "checkpoints/full_ft/update_0625.pt").is_file())
            self.assertEqual((resumed / "histories/full_ft.csv").read_text(), history.read_text())

    def test_hash_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            local, drive = root / "local", root / "drive"
            path = local / "checkpoints" / "lora_r4" / "update_1250.pt"
            path.parent.mkdir(parents=True)
            path.write_bytes(b"expected")
            saved = sync_file(path, local, drive, verify=True)
            saved.write_bytes(b"tampered")
            with self.assertRaisesRegex(IOError, "hash mismatch"):
                restore_results(root / "resumed", drive)


if __name__ == "__main__":
    unittest.main()
