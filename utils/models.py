import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from typing import Dict, Optional
import os

class ScalableModel(nn.Module):
    def __init__(self, num_classes: int, capacity: int, input_size: int, in_channels: int):
        super().__init__()
        self.num_classes = num_classes
        self.capacity = capacity
        self.input_size = input_size
        self.in_channels = in_channels

    # ---------- Mandatory architecture hooks ----------
    def forward_features(self, x):
        raise NotImplementedError

    def forward(self, x):
        return self.head(self.forward_features(x))

    # ---------- Utility ----------
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())
    
    def model_size_str(self):
        return (
            f"{self.__class__.__name__} | "
            f"capacity={self.capacity} | "
            f"params={self.num_parameters():,}"
        )

    # ==================================================
    # Training / Evaluation API
    # ==================================================

    def train_model(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        optimizer: Optimizer,
        device: torch.device,
        epochs: int,
        criterion=nn.CrossEntropyLoss(),
        scheduler=None,
        log_interval: int = 50,
        early_stop_on_zero_val_error: bool = True,
        ckpt_path: Optional[str] = None,
    ) -> Dict:

        self.to(device)
        history = {
            "train_acc": [],
            "val_acc": [],
            "train_loss": [],
            "val_loss": []
        }

        for epoch in range(epochs):
            self.train()
            correct, total, train_loss = 0, 0, 0.0

            for batch_idx, (x, y) in enumerate(train_loader):
                x, y = x.to(device), y.to(device)

                optimizer.zero_grad()
                logits = self(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                pred = logits.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)

                if batch_idx % log_interval == 0:
                    pass  # hook for logging if needed

            train_acc = correct / total
            train_loss /= len(train_loader)

            history["train_acc"].append(train_acc)
            history["train_loss"].append(train_loss)

            # ---------- Validation ----------
            if val_loader is not None:
                val_metrics = self.evaluate(val_loader, device, criterion)
                history["val_acc"].append(val_metrics["acc"])
                history["val_loss"].append(val_metrics["loss"])

                # Interpolation detection
                if early_stop_on_zero_val_error and val_metrics["acc"] == 1.0:
                    if ckpt_path:
                        self.save(ckpt_path)
                    break

            if scheduler is not None:
                scheduler.step()

        if ckpt_path:
            self.save(ckpt_path)

        return history

    @torch.no_grad()
    def evaluate(
        self,
        data_loader: DataLoader,
        device: torch.device,
        criterion=nn.CrossEntropyLoss()
    ) -> Dict:

        self.eval()
        self.to(device)

        correct, total_samples, loss_sum = 0, 0, 0.0

        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            logits = self(x)
            loss = criterion(logits, y)

            loss_sum += loss.item()
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total_samples += y.size(0)

        accuracy = correct / total_samples
        return {
            "accuracy": accuracy,
            "loss": loss_sum / total_samples,
            "error": 1.0 - accuracy
        }

    # ==================================================
    # Checkpointing
    # ==================================================

    def save(self, path: str, is_final: bool = True):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "state_dict": self.state_dict(),
            "capacity": self.capacity,
            "num_parameters": self.num_parameters(),
            "num_classes": self.num_classes,
            "model_class": self.__class__.__name__,
            "is_final": is_final
        }, path)

    def load(self, path: str, cap: int, map_location=None):
        ckpt = torch.load(path, map_location=map_location)
        assert cap == self.capacity, (
            f"Capacity mismatch: checkpoint={cap} "
            f"model={self.capacity}"
        )
        assert ckpt["num_classes"] == self.num_classes
        self.load_state_dict(ckpt["state_dict"])

def conv3x3(in_ch, out_ch, stride=1):
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            conv3x3(in_ch, out_ch, stride),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = conv3x3(in_ch, out_ch, stride)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_ch, out_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.shortcut = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return self.relu(out)



class ScalableResNet(ScalableModel):
    def __init__(
        self,
        capacity: int,
        num_classes: int = 10,
        input_size: int = 32,  
        in_channels: int = 3,
        base_width: int = 16,
        blocks_per_stage=(2, 2, 2)
    ):
        self.base_width = base_width
        self.input_size = input_size
        self.in_channels = in_channels
        self.blocks_per_stage = blocks_per_stage
        super().__init__(num_classes, capacity, input_size, in_channels)

        self._build_backbone()
        self._build_head()

    def _build_backbone(self):
        width = self.base_width * self.capacity

        self.stem = nn.Sequential(
            nn.Conv2d(3, width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True)
        )

        self.in_ch = width
        self.layer1 = self._make_layer(width, self.blocks_per_stage[0], stride=1)
        self.layer2 = self._make_layer(width * 2, self.blocks_per_stage[1], stride=2)
        self.layer3 = self._make_layer(width * 4, self.blocks_per_stage[2], stride=2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def _make_layer(self, out_ch, blocks, stride):
        layers = [BasicBlock(self.in_ch, out_ch, stride)]
        self.in_ch = out_ch
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_ch, out_ch))
        return nn.Sequential(*layers)

    def _build_head(self):
        self.head = nn.Linear(self.in_ch, self.num_classes)

    def forward_features(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x)
        return x.flatten(1)


class ScalableCNN(ScalableModel):
    def __init__(self, 
                capacity, 
                num_classes: int = 10, 
                in_channels: int = 1, 
                input_size: int = 28, 
                base_width = 2):
        self.in_channels = in_channels
        self.input_size = input_size
        self.base_width = base_width
        super().__init__(num_classes, capacity, input_size, in_channels)
        
        self.build_backbone()
        self.build_head()

    def build_backbone(self):
        # base_width = 10
        C = self.base_width * self.capacity
        self.features = nn.Sequential(
            ConvBlock(self.in_channels, C),
            ConvBlock(C, C),
            nn.MaxPool2d(2),
            ConvBlock(C, 2 * C),
            ConvBlock(2 * C, 2 * C),
            nn.MaxPool2d(2),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Infer feature dimension automatically
        with torch.no_grad():
            dummy = torch.zeros(1, self.in_channels, self.input_size, self.input_size)
            out = self.features(dummy)
            self.out_dim = out.view(1, -1).size(1)
            print("Inferred out_dim:", self.out_dim)


    def build_head(self):
        C = self.base_width * self.capacity
        self.out_dim = 2*C
        self.head = nn.Linear(self.out_dim, self.num_classes)

    def forward_features(self, x):
        x = self.features(x)
        x = self.gap(x)
        return x.view(x.size(0), -1)

class ScalableMLP(ScalableModel):
    def __init__(
        self,
        capacity: int,
        num_classes: int = 10,
        in_channels: int = 1,
        input_size: int = 28,
        base_width: int = 4,
    ):
        self.in_channels = in_channels
        self.input_size = input_size
        self.input_dim = in_channels * input_size * input_size
        self.base_width = base_width
        super().__init__(num_classes, capacity, input_size, in_channels)

        self.build_backbone()
        self.build_head()

    def build_backbone(self):
        C = self.base_width * self.capacity  # expansion width (analogous to base_width * capacity)

        self.features = nn.Sequential(
            nn.Linear(self.input_dim, C),
            nn.ReLU(inplace=True),
            nn.Linear(C, 2 * C),    # SG 8 ->2
            nn.ReLU(inplace=True),
        )

        # Feature dimension is fixed by construction
        self.out_dim = 2 * C        # SG 8 ->2

    def build_head(self):
        self.head = nn.Linear(self.out_dim, self.num_classes)

    def forward_features(self, x):
        # assert x.dim() == 4, \
        #     f"Expected input shape (B, C, H, W), got {x.shape}"

        x = x.view(x.size(0), -1)
        return self.features(x)

