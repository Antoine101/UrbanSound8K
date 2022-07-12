import torch
import torch.nn as nn
import torch.nn.functional as F


class UrbanSound8KModel(nn.Module):
    def __init__(self, input_height, input_width, output_neurons) -> None:
        super().__init__()
        # Definition of the model
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2)
        self.conv2_drop = nn.Dropout2d(p=0.3)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=self.count_neurons(image_dim=(input_height, input_width)), out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=256)
        self.fc3 = nn.Linear(in_features=256, out_features=output_neurons)

    def count_neurons(self, image_dim):
        x = torch.rand(1, 1, *image_dim)
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), kernel_size=2))
        x = self.flatten(x)
        return x.size(1)        

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), kernel_size=2))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        return logits 