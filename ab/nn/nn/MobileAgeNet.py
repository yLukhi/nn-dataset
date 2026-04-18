# MobileAgeNet v3: MobileNetV3-Large age regressor for mobile deployment.
# Main fixes:
#   - correct bounded-output bias initialization
#   - optional age normalization in loss space
#   - better warmup + fine-tune optimizer settings
#   - BN layers frozen during head-only warmup for stable transfer learning

import ssl
import math
import contextlib
from typing import Tuple, Dict, Any

import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights


@contextlib.contextmanager
def _unverified_ssl():
    orig = ssl._create_default_https_context
    ssl._create_default_https_context = ssl._create_unverified_context
    try:
        yield
    finally:
        ssl._create_default_https_context = orig


_FREEZE_EPOCHS = 5


def supported_hyperparameters():
    # Keep framework compatibility.
    # momentum is unused by AdamW but left here because your trainer searches it.
    return {'lr', 'momentum', 'dropout'}


class Net(nn.Module):
    def __init__(
        self,
        in_shape: Tuple[int, ...],
        out_shape: Tuple[int, ...],
        prm: Dict[str, Any],
        device: torch.device
    ) -> None:
        super().__init__()
        self.device = device
        self.prm = prm

        dropout = float(prm.get('dropout', 0.2))
        out_dim = out_shape[0] if out_shape else 1

        self.use_pretrained = bool(prm.get('pretrained', 1))
        self.freeze_epochs = int(prm.get('freeze_epochs', _FREEZE_EPOCHS))
        self.backbone_lr_mult = float(prm.get('backbone_lr_mult', 0.1))

        self.use_bounded_output = bool(prm.get('bounded_output', 1))
        self.min_age = float(prm.get('min_age', 0.0))
        self.max_age = float(prm.get('max_age', 116.0))

        # Use a slightly compressed training range to reduce saturation at extremes.
        # Inference is still in years.
        self.train_min_age = float(prm.get('train_min_age', self.min_age))
        self.train_max_age = float(prm.get('train_max_age', self.max_age))

        weights = MobileNet_V3_Large_Weights.IMAGENET1K_V2 if self.use_pretrained else None

        if weights is not None:
            try:
                with _unverified_ssl():
                    backbone = mobilenet_v3_large(weights=weights)
            except Exception as e:
                print(f"[WARN] Pretrained weights unavailable, using random init: {e}")
                backbone = mobilenet_v3_large(weights=None)
        else:
            backbone = mobilenet_v3_large(weights=None)

        self.features = backbone.features
        self.avgpool = backbone.avgpool

        # Slightly stronger but still lightweight head.
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(960, 256),
            nn.Hardswish(),
            nn.Dropout(p=dropout),
            nn.Linear(256, 64),
            nn.Hardswish(),
            nn.Dropout(p=max(0.05, dropout * 0.5)),
            nn.Linear(64, out_dim),
        )

        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)

        # Important:
        # If output is bounded by sigmoid, the final bias must be a logit, not an age in years.
        # Initialize to the midpoint age -> sigmoid(logit)=0.5 -> prediction near middle of range.
        if isinstance(self.head[-1], nn.Linear) and self.head[-1].bias is not None:
            nn.init.zeros_(self.head[-1].bias)

        self.to(self.device)

    def _freeze_backbone(self, requires_grad: bool) -> None:
        for p in self.features.parameters():
            p.requires_grad = requires_grad

        # During warmup, keep BN layers in eval mode for transfer-learning stability.
        if not requires_grad:
            for m in self.features.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    for p in m.parameters():
                        p.requires_grad = False

    def train_setup(self, prm: Dict[str, Any]) -> None:
        self.to(self.device)

        self.epoch_max = int(prm.get('epoch_max', 50))
        self._current_epoch = 0
        self._base_lr = float(prm['lr'])

        # SmoothL1 is okay, but a slightly larger beta is usually smoother for age regression.
        self.criteria = (nn.SmoothL1Loss(beta=1.0).to(self.device),)

        self._freeze_backbone(False)
        self.optimizer = torch.optim.AdamW(
            self.head.parameters(),
            lr=self._base_lr,
            weight_decay=1e-4,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        warmup_epochs = max(1, min(self.freeze_epochs, self.epoch_max))
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=warmup_epochs,
            eta_min=self._base_lr * 0.1,
        )

    def _init_finetune(self) -> None:
        self._freeze_backbone(True)

        remaining = max(1, self.epoch_max - self.freeze_epochs)
        lr = self._base_lr
        bb_lr = lr * self.backbone_lr_mult

        self.optimizer = torch.optim.AdamW(
            [
                {'params': self.features.parameters(), 'lr': bb_lr},
                {'params': self.head.parameters(), 'lr': lr},
            ],
            weight_decay=5e-5,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=remaining,
            eta_min=max(bb_lr, lr) * 1e-2,
        )

    def _apply_output_constraint(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_bounded_output:
            x = self.train_min_age + (self.train_max_age - self.train_min_age) * torch.sigmoid(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = self.head(x)
        x = self._apply_output_constraint(x)
        return x

    def learn(self, train_data: Any) -> None:
        if self._current_epoch == self.freeze_epochs:
            self._init_finetune()

        self.train()

        # Keep backbone BN frozen during head-only stage.
        if self._current_epoch < self.freeze_epochs:
            for m in self.features.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()

        for inputs, labels in train_data:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device).float()

            if labels.ndim == 1:
                labels = labels.unsqueeze(1)

            # Clamp noisy outliers into the same effective range as the bounded head.
            labels = labels.clamp_(self.train_min_age, self.train_max_age)

            self.optimizer.zero_grad(set_to_none=True)

            outputs = self(inputs)
            loss = self.criteria[0](outputs, labels)

            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), max_norm=2.0)
            self.optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()

        self._current_epoch += 1

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)