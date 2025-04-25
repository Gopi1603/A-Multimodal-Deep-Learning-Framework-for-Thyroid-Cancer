import torch
import torch.nn as nn
import torch.nn.functional as F

class FusionModule(nn.Module):
    """
    Fusion module with dynamic gating to integrate imaging and clinical features.
    For a given input, it learns a gating vector (values between 0 and 1) to weigh
    the contribution from imaging and clinical modalities.
    
    Let f_img be the imaging feature vector (e.g., 512-dim from HybridImagingModel)
    and f_clin be the clinical feature vector (e.g., 32-dim from ClinicalModel).
    
    We first project both to the same dimension, then compute a gate vector and fuse:
        f_fused = g * f_img_projected + (1 - g) * f_clin_projected
    """
    def __init__(self, img_dim=512, clin_dim=32, fused_dim=256):
        super(FusionModule, self).__init__()
        # Project imaging features to fused_dim
        self.img_proj = nn.Linear(img_dim, fused_dim)
        # Project clinical features to fused_dim
        self.clin_proj = nn.Linear(clin_dim, fused_dim)
        # Gating network: takes concatenated projected features and outputs gating vector
        self.gate_fc = nn.Sequential(
            nn.Linear(fused_dim * 2, fused_dim),
            nn.ReLU(inplace=True),
            nn.Linear(fused_dim, fused_dim),
            nn.Sigmoid()  # Outputs values between 0 and 1
        )
        # Final classifier layer (can be extended in multi-task setup)
        self.classifier = nn.Linear(fused_dim, 2)  # Example: binary classification
        
    def forward(self, f_img, f_clin):
        # f_img: (batch, img_dim), f_clin: (batch, clin_dim)
        img_feat = self.img_proj(f_img)   # (batch, fused_dim)
        clin_feat = self.clin_proj(f_clin)  # (batch, fused_dim)
        
        # Concatenate projected features for gating
        combined = torch.cat((img_feat, clin_feat), dim=1)  # (batch, fused_dim*2)
        gate = self.gate_fc(combined)  # (batch, fused_dim), values in [0,1]
        
        # Fuse features dynamically using gate (element-wise multiplication)
        fused_feature = gate * img_feat + (1 - gate) * clin_feat  # (batch, fused_dim)
        # Optionally, pass through an additional FC layer
        output = self.classifier(fused_feature)
        return output, fused_feature  # Return classifier output and fused features

if __name__ == "__main__":
    fusion_module = FusionModule()
    dummy_img_feat = torch.randn(4, 512)
    dummy_clin_feat = torch.randn(4, 32)
    out, fused = fusion_module(dummy_img_feat, dummy_clin_feat)
    print("Fusion Module Output Shape (classification logits):", out.shape)
    print("Fused Feature Shape:", fused.shape)
