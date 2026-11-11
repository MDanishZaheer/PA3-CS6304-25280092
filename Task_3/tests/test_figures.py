# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Check that saved-result figure functions render without model or data downloads."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from analysis import figures


class FigureTests(unittest.TestCase):
    def test_required_figures_render(self):
        fixed = {}
        for dataset in ("math", "mathvista"):
            quality = "certainty_rwc" if dataset == "math" else "visconf_rwc"
            fixed[dataset] = pd.DataFrame([
                {"method": method, "budget": budget, "correct": 0.3 + 0.02 * budget / 32,
                 "oracle": 0.5 + 0.1 * budget / 32, "no_correct_candidate": 0.5,
                 "selection_miss": 0.2, "token_cost": 300 * budget}
                for method in ("first", "self_consistency", quality) for budget in (1, 2, 4, 8, 16, 32)])
        signals = {name: pd.DataFrame({"score": ["length", "certainty"],
                                      "top_bottom_difference": [0.1, 0.2]})
                   for name in fixed}
        live = pd.DataFrame({"strategy": ["single", "parallel", "refinement"],
                             "accuracy": [0.3, 0.4, 0.35], "output_tokens": [2000, 2700, 2400]})
        adaptive = {name: pd.DataFrame({"policy": ["agreement", "always32", "quality"],
                                        "correct": [0.3, 0.4, 0.39],
                                        "token_cost": [2000, 8000, 5000]})
                    for name in fixed}
        stops = {name: pd.DataFrame({"policy": ["quality"] * 4,
                                     "count": [4, 8, 16, 32], "fraction": [0.2, 0.2, 0.2, 0.4]})
                 for name in fixed}
        trajectories = [{"sample_id": str(index), "confidence": [0.2, 0.4],
                         "visual_engagement": [0.1, 0.3]} for index in (1, 2)]
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            paths = [figures.sampling_curves(fixed, output / "sampling.png"),
                     figures.failure_decomposition(fixed, output / "failures.png"),
                     figures.signal_associations(signals, output / "signals.png"),
                     figures.visual_trajectories(trajectories, output / "trajectories.png"),
                     figures.live_comparison(live, output / "live.png"),
                     figures.adaptive_comparison(adaptive, stops, output / "adaptive.png"),
                     figures.cost_frontier(fixed, adaptive, output / "frontier.png")]
            self.assertTrue(all(path.stat().st_size > 1000 for path in paths))


if __name__ == "__main__":
    unittest.main()
