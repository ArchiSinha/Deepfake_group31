import torch
import torch.nn as nn
import timm


class Discriminator(nn.Module):
    """
    Image classifier (discriminator) with a configurable timm backbone.
    Input:  (B, 3, 224, 224), ImageNet-normalized
    Output: (B, 1) raw logit. Apply sigmoid to get P(fake).
    Label convention: 0 = Real, 1 = Fake.
    """
    def __init__(self, backbone="efficientnet_b0", pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(
            backbone, pretrained=pretrained, num_classes=0
        )
        in_features = self.backbone.num_features  # 1280 for efficientnet_b0
        self.head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1),
        )

    def get_embeddings(self, x):
        """Pooled backbone features (B, num_features), for the future confidence head."""
        return self.backbone(x)

    def forward(self, x):
        return self.head(self.get_embeddings(x))


class VideoDiscriminator(nn.Module):
    """
    Temporal discriminator for video clips (not used in Stage 1).
    Input:  (B, N, 3, H, W)
    Output: (B, 1) raw logit, higher = more likely fake.
    """
    def __init__(self, backbone="efficientnet_b0", pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(
            backbone, pretrained=pretrained, num_classes=0
        )
        in_features = self.backbone.num_features
        self.temporal = nn.LSTM(
            input_size=in_features,
            hidden_size=512,
            num_layers=1,
            batch_first=True,
        )
        self.classifier = nn.Linear(512, 1)

    def forward(self, x):
        B, N, C, H, W = x.shape
        feats = self.backbone(x.view(B * N, C, H, W)).view(B, N, -1)
        _, (hidden, _) = self.temporal(feats)
        return self.classifier(hidden.squeeze(0))