import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNBranch(nn.Module):
    """
    CNN branch to extract local features (edges, textures, fine details)
    from ultrasound images. Input: (batch, 3, 640, 640). Output: (batch, 128)
    """
    def __init__(self):
        super(CNNBranch, self).__init__()
        # Block 1: (3,640,640) -> (32,320,320)
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        # Block 2: (32,320,320) -> (64,160,160)
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        # Block 3: (64,160,160) -> (128,80,80)
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
    
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        # Global average pooling to convert (batch,128,80,80) to (batch,128)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        return x.view(x.size(0), -1)

if __name__ == "__main__":
    model = CNNBranch()
    dummy_input = torch.randn(4, 3, 640, 640)
    output = model(dummy_input)
    print("CNN Branch Output Shape:", output.shape)  # Expected: (4, 128)
