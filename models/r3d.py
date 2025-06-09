import torch
from torch import nn
from torchvision.models.video import r3d_18


class R3DModel(nn.Module):
    def __init__(
        self,
        num_classes: int = 10,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        # get the output size from the penultimate layer of ResNet
        model = r3d_18()
        model.fc = nn.Linear(model.fc.in_features, num_classes)

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
        # since we are passing through conv2D layers in ResNet
        # have to convert dims to [batch_size * frame, C, H, W]
        # output dims: [batch_size, 512 (in_features), 1, 1]
        features: torch.Tensor = self.feature_extractor(x.view(B * T, C, H, W))

        # LSTM expects (batch_size, sequence_length, input_size) as input
        features = features.view(B, T, -1)  # [batch_size, sequence_length, 512]
        # output of LSTM dims: [batch_size, sequence_length, hidden_dim]
        features, _ = self.lstm(features)
        features = features[:, -1, :]  # [batch_size, hidden_dim]

        features = self.dropout(features)

        output: torch.Tensor = self.linear(features)
        return output


def build_resnet_lstm_model(
    hidden_dim: int, num_lstm_layers: int, num_classes: int
) -> ResnetLSTModel:
    resnet: ResNet = resnet18(weights="DEFAULT")
    # freeze all the parameters
    for param in resnet.parameters():
        param.requires_grad = False

    return ResnetLSTModel(
        feature_extractor_model=resnet,
        hidden_dim=hidden_dim,
        num_lstm_layers=num_lstm_layers,
        num_classes=num_classes,
    )
