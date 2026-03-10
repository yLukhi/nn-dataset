# MobileAgeNet v2: Pretrained MobileNetV3-Large backbone for age estimation
# Transfer learning from ImageNet replaces the from-scratch v1 (~1.5M params).
# Architecture: pretrained features (5.5M) + regression head (≈250K) = ~5.7M total.
# Training strategy: 2-phase — freeze backbone for first _FREEZE_EPOCHS to warm up
# the head, then fine-tune all layers with differential LR (backbone 0.05× head).
# Expected improvement: MAE ~6.7 yrs (v1) → ≤3.5 yrs (v2 target).
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
from typing import Tuple, Dict, Any

# Number of epochs to keep the backbone frozen (head warm-up phase)
_FREEZE_EPOCHS = 10


def supported_hyperparameters():
    return {'lr', 'momentum', 'dropout'}


class Net(nn.Module):

    def train_setup(self, prm: Dict[str, Any]) -> None:
        self.to(self.device)
        self.epoch_max = prm.get('epoch_max', 50)
        self._current_epoch = 0
        self._base_lr = prm['lr']

        self.criteria = (nn.SmoothL1Loss(beta=0.5).to(self.device),)

        # Phase 1: head only — backbone frozen so pretrained features aren't corrupted
        # before the randomly-initialised head stabilises.
        self._set_backbone_grad(False)
        self.optimizer = torch.optim.AdamW(
            self.head.parameters(),
            lr=self._base_lr, weight_decay=1e-3, eps=1e-8,
        )
        freeze_epochs = min(_FREEZE_EPOCHS, self.epoch_max)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self._base_lr,
            epochs=freeze_epochs,
            steps_per_epoch=1,
            pct_start=0.3,
            anneal_strategy='cos',
            cycle_momentum=False,
            div_factor=3.0,
            final_div_factor=10.0,
        )

    def _set_backbone_grad(self, requires_grad: bool) -> None:
        for p in self.features.parameters():
            p.requires_grad = requires_grad

    def _init_finetune(self) -> None:
        """Switch to full fine-tune after head warm-up.

        Backbone gets 5% of head LR — enough to adapt ImageNet features to faces
        without destroying them. CosineAnnealingLR decays smoothly to near zero.
        """
        self._set_backbone_grad(True)
        remaining = max(1, self.epoch_max - _FREEZE_EPOCHS)
        lr = self._base_lr
        self.optimizer = torch.optim.AdamW([
            {'params': self.features.parameters(), 'lr': lr * 0.05},
            {'params': self.head.parameters(),     'lr': lr},
        ], weight_decay=5e-4, eps=1e-8)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=remaining, eta_min=lr * 1e-3,
        )

    def learn(self, train_data: Any) -> None:
        if self._current_epoch == _FREEZE_EPOCHS:
            self._init_finetune()

        self.train()
        for inputs, labels in train_data:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device).float()
            self.optimizer.zero_grad(set_to_none=True)
            outputs = self(inputs)
            loss = self.criteria[0](outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), max_norm=5.0)
            self.optimizer.step()
        self.scheduler.step()
        self._current_epoch += 1

    def __init__(self, in_shape: Tuple[int, ...], out_shape: Tuple[int, ...],
                 prm: Dict[str, Any], device: torch.device) -> None:
        super().__init__()
        self.device = device
        dropout = prm.get('dropout', 0.3)
        out_dim = out_shape[0] if out_shape else 1

        # ImageNet pretrained MobileNetV3-Large:
        #   features(x) → (B, 960, 7, 7) for 224×224 input
        #   avgpool(x)   → (B, 960, 1, 1)
        backbone = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V2)
        self.features = backbone.features
        self.avgpool  = backbone.avgpool

        # Lightweight regression head — backbone does the heavy lifting.
        # 960 → 256 → out_dim, with dropout for regularisation.
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(960, 256),
            nn.Hardswish(),
            nn.Dropout(p=dropout),
            nn.Linear(256, out_dim),
        )

        # Initialise head weights; backbone is already pretrained.
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
        # Seed the output bias at mean UTKFace age so initial predictions are
        # reasonable and the head converges quickly without exploding loss.
        self.head[-1].bias.data.fill_(33.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        return self.head(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
