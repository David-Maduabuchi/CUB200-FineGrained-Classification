import torch.nn as nn
import timm

from ..config import VISION_DIM, NUM_CLASSES   # ← if you make a config.py later


class ViTMixSwin(nn.Module):
    """
    Vision-only fine-grained classifier using Swin-B.
    """
    def __init__(self, freeze_vision: bool = True):
        super().__init__()

        # Backbone
        self.vision_encoder = timm.create_model(
            "swin_base_patch4_window7_224",
            pretrained=True,
            num_classes=0,          # remove head
            global_pool="avg"       # → (B, 1024)
        )

        if freeze_vision:
            for p in self.vision_encoder.parameters():
                p.requires_grad = False

        # Head
        self.classifier = nn.Sequential(
            nn.Linear(VISION_DIM, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(512, NUM_CLASSES),
        )

    def forward(self, images):
        features = self.vision_encoder(images)
        logits = self.classifier(features)
        return logits


    def unfreeze_last_stages(self, num_stages: int = 2):
        """Helper to unfreeze last N stages (Swin has layers.0, .1, .2, .3)"""
        stage_names = [f"layers.{i}" for i in range(4 - num_stages, 4)]
        for name, param in self.vision_encoder.named_parameters():
            if any(s in name for s in stage_names):
                param.requires_grad = True