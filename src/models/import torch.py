import torch
import torch.nn as nn

class ViTBranch(nn.Module):
    """
    Vision Transformer branch to extract global features.
    Input: (batch, 3, 640, 640) 
    Process: Divide into patches (16x16), embed, add positional encoding,
             and pass through transformer encoders.
    Output: (batch, embed_dim) with embed_dim typically 768.
    """
    def __init__(self, image_size=640, patch_size=16, in_channels=3, embed_dim=768, num_layers=7, num_heads=8):
        super(ViTBranch, self).__init__()
        self.patch_size = patch_size
        num_patches = (image_size // patch_size) ** 2  # For 640x640 with 16x16 patches: 1600 patches
        # Use a Conv2d layer to project image patches
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        # Learned positional embeddings
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, embed_dim))
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=0.1)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    
    def forward(self, x):
        # x shape: (batch, 3, 640, 640)
        x = self.proj(x)  # Shape: (batch, embed_dim, H', W') where H'=W'=image_size/patch_size
        x = x.flatten(2).transpose(1, 2)  # Shape: (batch, num_patches, embed_dim)
        x = x + self.pos_embedding  # Add positional encoding
        x = self.transformer(x)  # Process with transformer
        # Aggregate features (mean pooling over patch dimension)
        x = x.mean(dim=1)
        return x

if __name__ == "__main__":
    model = ViTBranch()
    dummy_input = torch.randn(4, 3, 640, 640)
    output = model(dummy_input)
    print("ViT Branch Output Shape:", output.shape)  # Expected: (4, 768)
