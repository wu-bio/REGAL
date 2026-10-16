"""Regression checks for opt-in RSP activation fallback; no model training."""
from dataclasses import asdict
import unittest

import numpy as np

from RSP.common import activation_active_mask
from RSP.decomposition import _candidate_from_component
from RSP.io import build_rsp_activation_table, build_rsp_family_member_activation_table
from RSP.schema import RSPPipelineConfig, RSPScoringConfig


class ActivationFallbackTests(unittest.TestCase):
    def setUp(self):
        # The isolated peak fails the two-spot component requirement, while
        # the lower plateau provides a connected footprint at the base floor.
        self.values = np.array([1] + [.2] * 19, dtype=np.float32)
        self.neighbors = np.arange(1, 21).reshape(-1, 1)
        self.neighbors[-1, 0] = 18
        self.options = dict(min_peak_fraction=.1, quantile=.05, min_abs=1e-6, min_component_size=2)

    def config(self, enabled):
        return RSPScoringConfig(min_active_spots=2, activation_active_min_component_size=2,
                               activation_trim_enabled=False, activation_active_fallback_to_base=enabled)

    def test_default_is_disabled_and_serialized(self):
        self.assertIs(asdict(RSPPipelineConfig())["scoring"]["activation_active_fallback_to_base"], False)
        mask, threshold = activation_active_mask(self.values, self.neighbors, **self.options)
        self.assertFalse(mask.any())
        self.assertAlmostEqual(threshold, .6)

    def test_candidate_screening_obeys_switch(self):
        loading = np.array([.4, .35, .25], dtype=np.float32)
        matrix = np.outer(loading, self.values)
        for enabled in (False, True):
            with self.subTest(enabled=enabled):
                candidate = _candidate_from_component(
                    candidate_id="test", source_k=1, source_seed=2024, source_seed_offset=0,
                    source_component=0, initializer_type="test", initialization_strategy="test",
                    genes=np.array(["G1", "G2", "G3"]), loading=loading, activation_raw=self.values,
                    x=matrix, x_raw=matrix, reliability=np.ones_like(matrix),
                    gene_support_fraction=np.ones(3, dtype=np.float32), neighbors=self.neighbors,
                    scoring_cfg=self.config(enabled), reconstruction=matrix,
                    backend_solver="test", backend_init="test", factorization_path_id="test",
                    loading_sparsity_penalty=0, activation_sparsity_penalty=0, l1_ratio=.5,
                    component_loading_lambda=0, component_activation_lambda=0,
                )
                self.assertEqual(candidate["activation_active_spots"], len(self.values) if enabled else 0)
                self.assertEqual("too_few_active_spots" in candidate["drop_reason"], not enabled)
                self.assertEqual(candidate["candidate_status"], "candidate" if enabled else "dropped")

    def test_primary_and_family_exports_use_the_same_switch(self):
        spots = np.array([f"spot{i}" for i in range(len(self.values))])
        for enabled in (False, True):
            for build in (build_rsp_activation_table, build_rsp_family_member_activation_table):
                with self.subTest(enabled=enabled, builder=build.__name__):
                    candidate = {"candidate_id": "test", "rsp_id": "RSP0000", "family_id": "F0000",
                                 "candidate_status": "retained", "activation_raw": self.values.copy(),
                                 "activation_clean": self.values.copy()}
                    table = build([candidate], spots, self.config(enabled), self.neighbors)
                    self.assertEqual(int(table.activation_active.sum()), len(self.values) if enabled else 0)
                    expected = .2 if enabled else .6
                    np.testing.assert_allclose(table.activation_active_threshold, expected)
                    np.testing.assert_array_equal(table.activation, self.values)

    def test_valid_masks_and_empty_base_masks_do_not_change(self):
        for values, neighbors in ((np.full(len(self.values), .2), self.neighbors),
                                  (np.zeros(len(self.values)), self.neighbors),
                                  (self.values, np.arange(len(self.values)).reshape(-1, 1))):
            with self.subTest(values=values, neighbors=neighbors):
                off, off_threshold = activation_active_mask(values, neighbors, **self.options)
                on, on_threshold = activation_active_mask(values, neighbors, **self.options, fallback_to_base=True)
                np.testing.assert_array_equal(off, on)
                self.assertEqual(off_threshold, on_threshold)


if __name__ == "__main__":
    unittest.main()
