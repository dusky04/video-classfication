import torch
from torch import nn
from torchvision.models import VGG, vgg16


class VGGLSTModel(nn.Module):
    def __init__(
        self,
        feature_extractor_model: VGG,
        hidden_dim: int = 256,
        num_lstm_layers: int = 1,
        num_classes: int = 10,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        # we will pass in the desired model to be used as a feature extractor
        # this time, we using VGG16, perhaps can extend to other models

        # configure the VGG16
        # VGG16 classifier has structure: classifier[0] (Linear 25088->4096), classifier[1] (ReLU),
        # classifier[2] (Dropout), classifier[3] (Linear 4096->4096), classifier[4] (ReLU),
        # classifier[5] (Dropout), classifier[6] (Linear 4096->1000)
        # get the output size from the penultimate layer of VGG16 classifier
        self.in_features = feature_extractor_model.classifier[
            0
        ].in_features  # 25088 for VGG16

        # create our own feature extractor by removing the classifier
        # VGG has 'features' (conv layers) and 'classifier' (fc layers)
        self.feature_extractor: nn.Module = feature_extractor_model.features
        # add adaptive pooling to get consistent output size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))

        self.hidden_dim: int = hidden_dim

        # create an LSTM for temporal modelling
        self.lstm = nn.LSTM(
            input_size=self.in_features,
            hidden_size=hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
        )

        self.dropout = nn.Dropout(dropout)

        # final projection layer
        self.linear = nn.Linear(in_features=hidden_dim, out_features=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input dim: [batch_size, frame, C, H, W]
        B, T, C, H, W = x.shape
        # since we are passing through conv2D layers in VGG16
        # have to convert dims to [batch_size * frame, C, H, W]
        x_reshaped = x.view(B * T, C, H, W)

        # pass through VGG16 feature extractor (conv layers only)
        features = self.feature_extractor(x_reshaped)  # [B*T, 512, H', W']

        # apply adaptive pooling to get consistent size
        features = self.adaptive_pool(features)  # [B*T, 512, 7, 7]

        # flatten the spatial dimensions
        features = features.view(B * T, -1)  # [B*T, 25088]

        # LSTM expects (batch_size, sequence_length, input_size) as input
        features = features.view(B, T, -1)  # [batch_size, sequence_length, 25088]
        # output of LSTM dims: [batch_size, sequence_length, hidden_dim]
        features, _ = self.lstm(features)
        features = features[:, -1, :]  # [batch_size, hidden_dim]

        features = self.dropout(features)

        output: torch.Tensor = self.linear(features)
        return output


def build_vgg_lstm_model(
    hidden_dim: int, num_lstm_layers: int, num_classes: int
) -> VGGLSTModel:
    vgg: VGG = vgg16(weights="DEFAULT")
    # freeze all the parameters
    for param in vgg.parameters():
        param.requires_grad = False

    return VGGLSTModel(
        feature_extractor_model=vgg,
        hidden_dim=hidden_dim,
        num_lstm_layers=num_lstm_layers,
        num_classes=num_classes,
    )
