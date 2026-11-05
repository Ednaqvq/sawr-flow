import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from ucf.selective import adaptive_gains, benefit_labels, budget_gains, render
from wave_tools import haar


class SelectiveGateTest(unittest.TestCase):
    def test_anchor_interior_lll_preserves_detail_and_boundary(self):
        residual = torch.ones(1, 8, 1, 1, 2)
        interior = torch.tensor([True, False]).view(1, 1, 1, 1, 2)
        anchor = torch.ones(1, 1, 2, 2, 4)
        mask = torch.ones_like(anchor)
        mask[..., -1] = 0
        condition = dict(anchor=anchor * mask, mask=mask)
        model = SimpleNamespace(background_value=0.)
        gains = torch.full_like(residual[:, 1:], .5)
        old = render(model, condition, residual, interior, gains)
        new = render(model, condition, residual, interior, gains, anchor_interior_lll=True)
        torch.testing.assert_close(new[..., 2:], old[..., 2:])
        torch.testing.assert_close(haar(new)[:, :1, ..., :1], haar(condition['anchor'])[:, :1, ..., :1])
        torch.testing.assert_close(haar(new)[:, 1:, ..., :1], haar(old)[:, 1:, ..., :1])

    def test_gain_prefers_keep_or_partial_by_measured_benefit(self):
        detail = torch.tensor([1., 1., 1.]).view(1, 1, 1, 1, -1)
        truth = torch.tensor([-1., .25, 2.]).view_as(detail)
        scores, benefit = benefit_labels(truth, detail)
        self.assertTrue(torch.all(scores.abs() <= 1))
        self.assertTrue(torch.all(benefit[..., 0] < 0))
        torch.testing.assert_close(adaptive_gains(scores), torch.tensor([0., .25, 1.]).view_as(detail))

    def test_matched_budget_is_exact_for_each_band(self):
        weight = torch.tensor([[3., 1., 0.], [2., 0., 6.]]).view(1, 2, 1, 1, 3)
        rank = torch.tensor([[2., 1., 3.], [0., 2., 1.]]).view_as(weight)
        gains = budget_gains(rank, weight, .25)
        torch.testing.assert_close((gains * weight).sum(-1), weight.sum(-1) * .25)
        torch.testing.assert_close((gains * weight)[0, :, 0, 0], torch.tensor([[1., 0., 0.], [0., 0., 2.]]))
