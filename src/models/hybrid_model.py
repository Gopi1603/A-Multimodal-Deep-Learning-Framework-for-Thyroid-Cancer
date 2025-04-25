import torch
import torch.nn as nn
from .cnn import CNNBranch
from .vit import ViTBranch

class HybridImagingModel(nn.Module):
    """
    Hybrid model that integrates CNN and ViT branches.
    Baseline: simple concatenation of CNN (128-dim) and ViT (768-dim) outputs.
    Projects concatenated features to a unified 512-dimensional feature vector.
    """
    def __init__(self):
        super(HybridImagingModel, self).__init__()
        self.cnn_branch = CNNBranch()  # Output: (batch, 128)
        self.vit_branch = ViTBranch()    # Output: (batch, 768)
        self.fc = nn.Linear(128 + 768, 512)  # Projection layer
        
    def forward(self, x):
        cnn_features = self.cnn_branch(x)  # (batch, 128)
        vit_features = self.vit_branch(x)    # (batch, 768)
        fused_features = torch.cat((cnn_features, vit_features), dim=1)  # (batch, 896)
        out = self.fc(fused_features)  # (batch, 512)
        return out

if __name__ == "__main__":
    model = HybridImagingModel()
    dummy_input = torch.randn(4, 3, 640, 640)
    output = model(dummy_input)
    print("Hybrid Imaging Model Output Shape:", output.shape)  # Expected: (4, 512)
