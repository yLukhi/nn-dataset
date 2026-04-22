# Auto-generated 4-Expert Heterogeneous MoE: AirNet + AirNext + FractalNet + UNet2D
# Four-expert mixture with learned gating, mixup, and LR scheduling

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def supported_hyperparameters():
    return {'dropout', 'lr', 'momentum'}



# ============================================================================
# EXPERT 1: AirNet
# ============================================================================


class AirInitBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.layers(x)


class AirUnit(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        self.downsample = (
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            ) if stride != 1 or in_channels != out_channels else nn.Identity()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.downsample(x)
        x = self.layers(x)
        return self.relu(x + residual)


class AirNetExpert(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.in_channels = in_shape[1]
        self.image_size = in_shape[2]
        self.num_classes = out_shape[0]
        self.learning_rate = prm['lr']
        self.momentum = prm['momentum']

        channels = [64, 128, 256, 512]
        init_block_channels = 64

        self.features = self.build_features(init_block_channels, channels)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(channels[-1], self.num_classes)

    def build_features(self, init_block_channels, channels):
        layers = [AirInitBlock(self.in_channels, init_block_channels)]
        for i, out_channels in enumerate(channels):
            layers.append(AirUnit(
                in_channels=init_block_channels if i == 0 else channels[i - 1],
                out_channels=out_channels,
                stride=1 if i == 0 else 2))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.learning_rate,
            momentum=self.momentum
        )

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()



# ============================================================================
# EXPERT 2: AirNext
# ============================================================================


class AirBlock(nn.Module):
    def __init__(self, in_channels, out_channels, groups=1, ratio=2):
        super(AirBlock, self).__init__()
        mid_channels = out_channels // ratio
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = torch.relu(self.bn2(self.conv2(x)))
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        x = self.bn3(self.conv3(x))
        x = self.sigmoid(x)
        return x


class AirNeXtUnit(nn.Module):
    def __init__(self, in_channels, out_channels, stride, cardinality, bottleneck_width, ratio):
        super(AirNeXtUnit, self).__init__()
        mid_channels = out_channels // 4
        D = int(math.floor(mid_channels * (bottleneck_width / 64.0)))
        group_width = cardinality * D
        self.use_air_block = (stride == 1 and mid_channels < 512)

        self.conv1 = nn.Conv2d(in_channels, group_width, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(group_width, group_width, kernel_size=3, stride=stride, padding=1, groups=cardinality, bias=False)
        self.conv3 = nn.Conv2d(group_width, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        if self.use_air_block:
            self.air = AirBlock(in_channels, group_width, groups=(cardinality // ratio), ratio=ratio)

        self.resize_identity = (in_channels != out_channels) or (stride != 1)
        if self.resize_identity:
            self.identity_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        self.activ = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.use_air_block:
            att = self.air(x)
            att = F.interpolate(att, size=x.shape[2:], mode="bilinear", align_corners=True)  # Ensure att matches x dimensions
        identity = self.identity_conv(x) if self.resize_identity else x
        x = self.conv1(x)
        x = self.conv2(x)
        if self.use_air_block:
            x = x * att
        x = self.conv3(x)
        x = x + identity
        x = self.activ(x)
        return x


class AirNextExpert(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super(AirNextExpert, self).__init__()
        self.device = device
        channel_number = in_shape[1]
        image_size = in_shape[2]
        class_number = out_shape[0]

        channels = [[64, 64, 64], [128, 128, 128], [256, 256, 256], [512, 512, 512]]
        init_block_channels = 64
        cardinality = 32
        bottleneck_width = 4
        ratio = 2

        self.in_size = image_size
        self.num_classes = class_number

        self.features = nn.Sequential(
            nn.Conv2d(channel_number, init_block_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(init_block_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        in_channels = init_block_channels
        for i, channels_per_stage in enumerate(channels):
            stage = nn.Sequential()
            for j, out_channels in enumerate(channels_per_stage):
                stride = 2 if (j == 0) and (i != 0) else 1
                stage.add_module("unit{}".format(j + 1), AirNeXtUnit(in_channels, out_channels, stride, cardinality, bottleneck_width, ratio))
                in_channels = out_channels
            self.features.add_module("stage{}".format(i + 1), stage)

        self.features.add_module("final_pool", nn.AdaptiveAvgPool2d(1))
        self.output = nn.Linear(in_channels, class_number)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.output(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=prm['lr'], weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.5)

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()
        self.scheduler.step()




# ============================================================================
# EXPERT 3: FractalNet
# ============================================================================


class FractalNetExpert(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        dropout_prob = prm['dropout']
        self.num_columns = 3
        self.glob_drop_ratio = 0.5
        self.loc_drop_prob = 0.15
        self.glob_num_columns = np.random.randint(0, self.num_columns, size=(1,))

        N = 5
        channels = [64 * (2 ** (i if i != 4 else i - 1)) for i in range(N)]
        dropout_probs = [min(0.5, dropout_prob + i * 0.1) for i in range(N)]

        self.features = nn.Sequential()
        in_channels = in_shape[1]
        for i, out_channels in enumerate(channels):
            self.features.add_module(f"unit{i + 1}", FractalUnit(
                in_channels=in_channels,
                out_channels=out_channels,
                num_columns=self.num_columns,
                loc_drop_prob=self.loc_drop_prob,
                dropout_prob=dropout_probs[i]))
            in_channels = out_channels

        # Compute the correct input size for the linear layer dynamically
        with torch.no_grad():
            dummy_input = torch.zeros(1, *in_shape[1:])
            dummy_output = self.features(dummy_input)
            in_features = dummy_output.view(1, -1).shape[1]

        self.output = nn.Linear(in_features=in_features, out_features=out_shape[0])
        self._init_params()

    def _init_params(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)  # Ensure input tensor is float32
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)  # Correct feature flattening
        x = self.output(x)
        return x

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=prm['lr'], momentum=prm['momentum'])

    def learn(self, train_data):
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device).to(torch.float32), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria[0](outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()


class FractalUnit(nn.Module):
    def __init__(self, in_channels, out_channels, num_columns, loc_drop_prob, dropout_prob):
        super().__init__()
        self.block = FractalBlock(in_channels, out_channels, num_columns, loc_drop_prob, dropout_prob)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.block(x)
        x = self.pool(x)
        return x


class FractalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_columns, loc_drop_prob, dropout_prob):
        super().__init__()
        self.num_columns = num_columns
        self.loc_drop_prob = loc_drop_prob
        self.blocks = nn.Sequential()
        depth = 2 ** (num_columns - 1)
        for i in range(depth):
            level_block = nn.Sequential()
            for j in range(num_columns):
                if (i + 1) % (2 ** j) == 0:
                    in_channels_ij = in_channels if (i + 1 == 2 ** j) else out_channels
                    level_block.add_module(f"subblock{j + 1}", drop_conv3x3_block(
                        in_channels=in_channels_ij,
                        out_channels=out_channels,
                        dropout_prob=dropout_prob))
            self.blocks.add_module(f"block{i + 1}", level_block)

    def forward(self, x):
        outs = [x] * self.num_columns
        for level_block in self.blocks:
            outs_i = [block(inputs) for block, inputs in zip(level_block, outs)]
            joined_out = torch.stack(outs_i).mean(dim=0)
            outs[:len(level_block)] = [joined_out] * len(level_block)
        return outs[0]


def drop_conv3x3_block(in_channels, out_channels, stride=1, padding=1, bias=False, dropout_prob=0.0):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding, bias=bias),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Dropout2d(p=dropout_prob) if dropout_prob > 0 else nn.Identity()
    )



# ============================================================================
# EXPERT 4: UNet2D
# ============================================================================


class UNet2DModel(nn.Module):
    def __init__(
            self,
            sample_size=32,
            in_channels=3,
            out_channels=128,
            layers_per_block=2,
            block_out_channels=(32, 64, 128),
            down_block_types=("DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
    ):
        super(UNet2DModel, self).__init__()
        self.sample_size = sample_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.layers_per_block = layers_per_block

        self.down_blocks = nn.ModuleList()
        in_ch = in_channels
        for out_ch, block_type in zip(block_out_channels, down_block_types):
            self.down_blocks.append(self._make_block(in_ch, out_ch, block_type))
            in_ch = out_ch

        self.up_blocks = nn.ModuleList()
        for out_ch, block_type in zip(block_out_channels[::-1], up_block_types):
            self.up_blocks.append(self._make_block(in_ch, out_ch, block_type))
            in_ch = out_ch

        self.final_conv = nn.Conv2d(in_ch, out_channels, kernel_size=1)

    def _make_block(self, in_channels, out_channels, block_type):
        if block_type.startswith("Down"):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(),
            )
        elif block_type.startswith("Up"):
            return nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels, out_channels, kernel_size=2, stride=2
                ),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(),
            )
        else:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(),
            )

    def forward(self, x, timesteps):
        down_features = []
        for block in self.down_blocks:
            x = block(x)
            down_features.append(x)

        for block in self.up_blocks:
            x = block(x)

        x = self.final_conv(x)
        return nn.Identity()(x)


class UNet2DExpert(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super(UNet2DExpert, self).__init__()
        self.device = device

        channel_number = in_shape[1]
        image_size = in_shape[2]
        class_number = out_shape[0]

        self.unet = UNet2DModel(
            sample_size=image_size,
            in_channels=channel_number,
            out_channels=128,
            layers_per_block=2,
            block_out_channels=(32, 64, 128),
            down_block_types=("DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D"), )
        self.classifier = nn.Sequential(nn.Dropout(prm['dropout']), nn.Linear(128, class_number))
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        batch_size = x.shape[0]
        timesteps = torch.full((batch_size,), 50, dtype=torch.long, device=x.device)
        unet_output = self.unet(x, timesteps)
        unet_output = unet_output.to(torch.float32)
        pooled_features = unet_output.mean(dim=(2, 3))
        logits = self.classifier(pooled_features)
        return logits

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)

        lr = prm['lr']
        momentum = prm['momentum']

        self.optimizer = optim.SGD(self.parameters(), lr=lr, momentum=momentum)

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()



# ============================================================================
# HETEROGENEOUS MOE GATE
# ============================================================================
class HeterogeneousGate(nn.Module):
    """Lightweight CNN-based gating network that routes inputs to 4 experts."""
    def __init__(self, input_channels, n_experts=4):
        super().__init__()
        self.n_experts = n_experts
        self.gate_features = nn.Sequential(
            nn.Conv2d(input_channels, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.gate = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, n_experts),
        )
        self.temperature = nn.Parameter(torch.ones(1) * 2.0)

    def forward(self, x):
        features = self.gate_features(x).flatten(1)
        logits = self.gate(features) / torch.clamp(self.temperature, 0.5, 5.0)
        if self.training:
            logits = logits + torch.randn_like(logits) * 0.1
        return F.softmax(logits, dim=-1), logits


# ============================================================================
# HETEROGENEOUS MOE NET — AirNet + AirNext + FractalNet + UNet2D
# ============================================================================
class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        n_experts = 4

        # Defaults for expert-specific hyperparameters
        pass  # no extra defaults needed
        self.experts = nn.ModuleList([
            AirNetExpert(in_shape, out_shape, prm, device),
            AirNextExpert(in_shape, out_shape, prm, device),
            FractalNetExpert(in_shape, out_shape, prm, device),
            UNet2DExpert(in_shape, out_shape, prm, device),
        ])
        self.gate = HeterogeneousGate(in_shape[1], n_experts=n_experts)

        self.load_balance_weight = 0.01
        self.label_smoothing = 0.1
        self.mixup_alpha = 0.2
        self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_weights, _ = self.gate(x)
        expert_outputs = torch.stack([e(x) for e in self.experts], dim=2)
        return torch.sum(expert_outputs * gate_weights.unsqueeze(1), dim=2)

    def _load_balance_loss(self):
        if not hasattr(self, '_last_gw'):
            return torch.tensor(0.0, device=self.device)
        gw = self._last_gw
        usage = gw.sum(dim=0)
        target = gw.sum() / len(self.experts)
        return F.mse_loss(usage, target.expand_as(usage))

    def _mixup_data(self, x, y):
        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha) if self.mixup_alpha > 0 else 1.0
        idx = torch.randperm(x.size(0), device=self.device)
        return lam * x + (1 - lam) * x[idx], y, y[idx], lam

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        expert_params, gate_params = [], list(self.gate.parameters())
        for e in self.experts:
            expert_params.extend(list(e.parameters()))
        self.optimizer = torch.optim.AdamW([
            {'params': expert_params, 'lr': prm.get('lr', 0.001), 'weight_decay': 5e-4},
            {'params': gate_params, 'lr': prm.get('lr', 0.001) * 2, 'weight_decay': 2.5e-4},
        ])
        self.warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer, start_factor=0.1, total_iters=5
        )
        self.main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=50, eta_min=1e-6
        )
        self._current_epoch = 0

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs = inputs.to(self.device, dtype=torch.float32)
            labels = labels.to(self.device, dtype=torch.long)
            if self.mixup_alpha > 0:
                inputs, labels_a, labels_b, lam = self._mixup_data(inputs, labels)
                self.optimizer.zero_grad()
                outputs = self(inputs)
                loss = lam * self.criteria(outputs, labels_a) + (1 - lam) * self.criteria(outputs, labels_b)
            else:
                self.optimizer.zero_grad()
                outputs = self(inputs)
                loss = self.criteria(outputs, labels)
            self._last_gw = self.gate(inputs)[0].detach()
            loss = loss + self.load_balance_weight * self._load_balance_loss()
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            self.optimizer.step()
        self._current_epoch += 1
        if self._current_epoch <= 5:
            self.warmup_scheduler.step()
        else:
            self.main_scheduler.step()

