import pathlib
import sys

import numpy as np
import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

torch = pytest.importorskip("torch")
pytest.importorskip("timm")  # train.py pulls in model.py, which needs timm

from rsna_knee.constants import TARGET_LABELS  # noqa: E402
from rsna_knee.imaging.train import score_predictions, weighted_bce  # noqa: E402


class TestWeightedBce:
    def test_zero_weight_study_is_ignored(self) -> None:
        logits = torch.tensor([[5.0] * 12, [-5.0] * 12])
        targets = torch.tensor([[1.0] * 12, [1.0] * 12])
        confident_only = weighted_bce(logits, targets, torch.tensor([1.0, 0.0]))
        both = weighted_bce(logits, targets, torch.tensor([1.0, 1.0]))
        assert float(confident_only) < float(both)

    def test_weights_are_normalized(self) -> None:
        logits = torch.zeros(2, 12)
        targets = torch.ones(2, 12)
        small = weighted_bce(logits, targets, torch.tensor([0.25, 0.25]))
        large = weighted_bce(logits, targets, torch.tensor([3.0, 3.0]))
        assert float(small) == pytest.approx(float(large), abs=1e-6)

    def test_gold_row_dominates(self) -> None:
        logits = torch.tensor([[3.0] * 12, [-3.0] * 12])
        targets = torch.tensor([[0.0] * 12, [1.0] * 12])
        gold_first = weighted_bce(logits, targets, torch.tensor([3.0, 0.25]))
        gold_second = weighted_bce(logits, targets, torch.tensor([0.25, 3.0]))
        assert float(gold_first) == pytest.approx(float(gold_second), abs=1e-5)


class TestScorePredictions:
    def test_perfect_separation(self) -> None:
        truth = np.zeros((4, len(TARGET_LABELS)))
        truth[2:] = 1.0
        preds = np.zeros((4, len(TARGET_LABELS)))
        preds[2:] = 0.9
        macro, per_label = score_predictions(truth, preds)
        assert macro == pytest.approx(1.0)
        assert len(per_label) == len(TARGET_LABELS)

    def test_single_class_targets_are_dropped(self) -> None:
        truth = np.zeros((4, len(TARGET_LABELS)))
        truth[2:, 0] = 1.0
        preds = np.random.RandomState(0).rand(4, len(TARGET_LABELS))
        macro, per_label = score_predictions(truth, preds)
        assert list(per_label) == [TARGET_LABELS[0]]
        assert np.isfinite(macro)

    def test_all_single_class_returns_nan(self) -> None:
        truth = np.zeros((4, len(TARGET_LABELS)))
        preds = np.random.RandomState(0).rand(4, len(TARGET_LABELS))
        macro, per_label = score_predictions(truth, preds)
        assert np.isnan(macro)
        assert per_label == {}


class TestSlotAttentionHead:
    def test_masked_slots_do_not_change_logits(self) -> None:
        from rsna_knee.imaging.model import SlotAttentionHead

        torch.manual_seed(0)
        head = SlotAttentionHead(embed_dim=16, n_targets=len(TARGET_LABELS), attn_dim=8).eval()
        embeddings = torch.randn(1, 4, 16)
        mask = torch.tensor([[1.0, 1.0, 0.0, 0.0]])

        with torch.no_grad():
            baseline = head(embeddings, mask)
            perturbed = embeddings.clone()
            perturbed[:, 2:] = torch.randn(1, 2, 16) * 50.0
            after = head(perturbed, mask)
        assert torch.allclose(baseline, after, atol=1e-5)

    def test_output_shape_and_no_nan_when_all_masked(self) -> None:
        from rsna_knee.imaging.model import SlotAttentionHead

        head = SlotAttentionHead(embed_dim=16, n_targets=len(TARGET_LABELS), attn_dim=8).eval()
        with torch.no_grad():
            out = head(torch.randn(2, 6, 16), torch.zeros(2, 6))
        assert out.shape == (2, len(TARGET_LABELS))
        assert torch.isfinite(out).all()


class TestModelForward:
    def test_forward_with_tiny_backbone(self) -> None:
        pytest.importorskip("timm")
        from rsna_knee.imaging.model import KneeSlotModel

        model = KneeSlotModel(
            backbone="vit_tiny_patch16_224",
            pretrained=False,
            img_size=224,
            unfrozen_blocks=2,
        ).eval()
        with torch.no_grad():
            logits = model(torch.rand(1, 2, 3, 224, 224), torch.tensor([[1.0, 0.0]]))
        assert logits.shape == (1, len(TARGET_LABELS))
        assert torch.isfinite(logits).all()

    def test_freezing_leaves_only_last_blocks_trainable(self) -> None:
        pytest.importorskip("timm")
        from rsna_knee.imaging.model import KneeSlotModel

        model = KneeSlotModel(
            backbone="vit_tiny_patch16_224", pretrained=False, img_size=224, unfrozen_blocks=2
        )
        blocks = list(model.backbone.blocks)
        assert not any(p.requires_grad for p in blocks[0].parameters())
        assert all(p.requires_grad for p in blocks[-1].parameters())
        assert all(p.requires_grad for p in model.head.parameters())


class TestCnnBackbone:
    BACKBONE = "efficientnet_b3"

    def _build(self, **kwargs):
        pytest.importorskip("timm")
        from rsna_knee.imaging.model import KneeSlotModel

        return KneeSlotModel(backbone=self.BACKBONE, pretrained=False, img_size=224, **kwargs)

    def test_forward_with_cnn_backbone(self) -> None:
        model = self._build(unfrozen_blocks=2).eval()
        assert model.is_token_backbone is False
        with torch.no_grad():
            logits = model(
                torch.rand(1, 6, 3, 224, 224),
                torch.tensor([[1.0, 1.0, 1.0, 1.0, 0.0, 0.0]]),
            )
        assert logits.shape == (1, len(TARGET_LABELS))
        assert torch.isfinite(logits).all()

    def test_cnn_freezing_unfreezes_last_stages(self) -> None:
        model = self._build(unfrozen_blocks=2)
        stages = list(model.backbone.blocks)
        assert not any(p.requires_grad for p in stages[0].parameters())
        assert all(p.requires_grad for p in stages[-1].parameters())

    def test_cnn_parameter_groups_split_lrs(self) -> None:
        from rsna_knee.imaging.model import parameter_groups

        model = self._build(unfrozen_blocks=2)
        groups = parameter_groups(model, backbone_lr=1e-5, head_lr=1e-3)
        assert [g["lr"] for g in groups] == [1e-5, 1e-3]
        head_ids = {id(p) for p in groups[1]["params"]}
        backbone_ids = {id(p) for p in groups[0]["params"]}
        assert head_ids.isdisjoint(backbone_ids)
        # The CNN projection layer must be trained at the head learning rate.
        assert all(id(p) in head_ids for p in model.slot_proj.parameters())
        assert all(g["params"] for g in groups)
