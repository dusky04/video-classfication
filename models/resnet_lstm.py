import torch
from torch import nn
from torchvision.models import ResNet, resnet18

from config import Config


class ResnetLSTModel(nn.Module):
    def __init__(
        self,
        config: Config,
        feature_extractor_model: ResNet,
    ) -> None:
        super().__init__()
        # we will pass in the desired model to be used as a feature extractor
        # this time, we using ResNet18, perhaps can extend to VGG
        # configure the ResNet
        # get the output size from the penultimate layer of ResNet
        self.in_features = feature_extractor_model.fc.in_features
        # create our own feature extractor by removing 'fc' layer
        self.feature_extractor = nn.Sequential(
            *list(feature_extractor_model.children())[:-1]
        )

        self.batch_norm = nn.BatchNorm2d(num_features=512)

        self.lstm = nn.LSTM(
            input_size=self.in_features,
            hidden_size=config.LSTM_HIDDEN_DIM,
            num_layers=config.LSTM_NUM_LAYERS,
            dropout=config.LSTM_DROPOUT,
            batch_first=True,
        )

        self.dropout = nn.Dropout(config.LSTM_DROPOUT)

        self.linear = nn.Linear(
            in_features=config.LSTM_HIDDEN_DIM, out_features=config.NUM_CLASSES
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input dim: [batch_size, frame, C, H, W]
        B, T, C, H, W = x.shape
        # since we are passing through conv2D layers in ResNet
        # have to convert dims to [batch_size * frame, C, H, W]
        # output dims: [batch_size, 512 (in_features), 1, 1]
        features = self.feature_extractor(x.view(B * T, C, H, W))
        features = self.batch_norm(features)

        # LSTM expects (batch_size, sequence_length, input_size) as input
        features = features.view(B, T, -1)  # [batch_size, sequence_length, 512]
        # output of LSTM dims: [batch_size, sequence_length, hidden_dim]
        features, _ = self.lstm(features)
        features = features[:, -1, :]  # [batch_size, hidden_dim]

        features = self.dropout(features)

        output = self.linear(features)
        return output


def build_resnet_lstm_model(config: Config) -> ResnetLSTModel:
    resnet: ResNet = resnet18(weights="DEFAULT")

    for param in resnet.parameters():
        param.requires_grad = False
    # fine tune the lower layer
    for param in resnet.layer4.parameters():
        param.requires_grad = True

    return ResnetLSTModel(config, feature_extractor_model=resnet)
