import torch
from torch import nn


# expected input shape: (batch_size, 3, 180, 180)
class FeatureExtractorCNN(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

        # Block 1
        self.conv_1 = nn.Conv2d(
            in_channels=3, out_channels=64, kernel_size=3, padding=1
        )
        self.batch_norm_1 = nn.BatchNorm2d(num_features=64)
        self.max_pool_1 = nn.MaxPool2d(kernel_size=2)

        # Block 2
        self.conv_2 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, padding=1
        )
        self.conv_3 = nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=3, padding=1
        )
        self.batch_norm_2 = nn.BatchNorm2d(num_features=128)
        self.max_pool_2 = nn.MaxPool2d(kernel_size=2)

        # Block 3
        self.conv_4 = nn.Conv2d(
            in_channels=128, out_channels=256, kernel_size=3, padding=1
        )
        self.conv_5 = nn.Conv2d(
            in_channels=256, out_channels=256, kernel_size=3, padding=1
        )
        self.batch_norm_3 = nn.BatchNorm2d(256)
        self.max_pool_3 = nn.MaxPool2d(kernel_size=2)

        # Block 4
        self.conv_6 = nn.Conv2d(
            in_channels=256, out_channels=384, kernel_size=3, padding=1
        )
        self.conv_7 = nn.Conv2d(
            in_channels=384, out_channels=384, kernel_size=3, padding=1
        )
        self.batch_norm_4 = nn.BatchNorm2d(384)
        self.max_pool_4 = nn.MaxPool2d(kernel_size=2)

        # Block 5
        self.conv_8 = nn.Conv2d(
            in_channels=384, out_channels=480, kernel_size=3, padding=1
        )
        self.conv_9 = nn.Conv2d(
            in_channels=480, out_channels=480, kernel_size=3, padding=1
        )
        self.batch_norm_5 = nn.BatchNorm2d(480)
        self.max_pool_5 = nn.MaxPool2d(kernel_size=2)

        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Block 1
        x = self.relu(self.conv_1(x))
        x = self.batch_norm_1(x)
        x = self.max_pool_1(x)  # Shape: [N, 64, 90, 90]

        # Block 2
        x = self.relu(self.conv_2(x))
        x = self.relu(self.conv_3(x))
        x = self.batch_norm_2(x)
        x = self.max_pool_2(x)  # Shape: [N, 128, 45, 45]

        # Block 3
        x = self.relu(self.conv_4(x))
        x = self.relu(self.conv_5(x))
        x = self.batch_norm_3(x)
        x = self.max_pool_3(x)  # Shape: [N, 256, 22, 22]

        # Block 4
        x = self.relu(self.conv_6(x))
        x = self.relu(self.conv_7(x))
        x = self.batch_norm_4(x)
        x = self.max_pool_4(x)  # Shape: [N, 384, 11, 11]

        # Block 5
        x = self.relu(self.conv_8(x))
        x = self.relu(self.conv_9(x))
        x = self.batch_norm_5(x)
        x = self.max_pool_5(x)  # Shape: [N, 480, 5, 5]

        return x


class Mo(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.feature_extractor = FeatureExtractorCNN()

        self.gru = nn.GRU(input_size=15 * 12000, hidden_size=180, batch_first=True)

        self.classifier = nn.Sequential(
            nn.Linear(in_features=180, out_features=512),
            nn.Linear(in_features=512, out_features=128),
            nn.Linear(in_features=128, out_features=64),
            nn.Linear(in_features=64, out_features=10),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input shape: [B, T, C, H, W]
        B, T, C, H, W = x.shape

        # extract the features from the CNN - [B * T, 480, 5, 5]
        x = self.feature_extractor(x.view(B * T, C, H, W).contiguous())
        x = x.view(B, -1).contiguous()  # [B, T * 12000]

        # pass through the GRU for sequence modelling
        x, _ = self.gru(x)  # [B, something]

        x = self.classifier(x)
        return x
