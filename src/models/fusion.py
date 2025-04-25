# src/models/fusion.py
import torch
import torch.nn as nn

class FusionModule(nn.Module):
    """
    Fusion module that dynamically fuses imaging and clinical features.
    Projects imaging features (512-dim) and clinical features (32-dim) into a common space,
    applies a gating mechanism, and then fuses them. Also produces multi-task outputs:
    classification logits and a risk score.
    """
    def __init__(self, img_dim=512, clin_dim=32, fused_dim=256, num_classes=2):
        super(FusionModule, self).__init__()
        self.img_proj = nn.Linear(img_dim, fused_dim)
        self.clin_proj = nn.Linear(clin_dim, fused_dim)
        
        self.gate_fc = nn.Sequential(
            nn.Linear(2 * fused_dim, fused_dim),
            nn.ReLU(inplace=True),
            nn.Linear(fused_dim, fused_dim),
            nn.Sigmoid()
        )
        
        # Multi-task heads:
        self.classifier = nn.Linear(fused_dim, num_classes)
        self.risk_predictor = nn.Linear(fused_dim, 1)
        
    def forward(self, f_img, f_clin):
        img_feat = self.img_proj(f_img)
        clin_feat = self.clin_proj(f_clin)
        combined = torch.cat((img_feat, clin_feat), dim=1)
        gate = self.gate_fc(combined)
        fused_feature = gate * img_feat + (1 - gate) * clin_feat
        
        class_logits = self.classifier(fused_feature)
        risk_score = self.risk_predictor(fused_feature)
        return class_logits, risk_score, fused_feature

if __name__ == "__main__":
    # Test the fusion module with dummy data
    dummy_img_feat = torch.randn(4, 512)
    dummy_clin_feat = torch.randn(4, 32)
    fusion_module = FusionModule(num_classes=2)
    class_logits, risk_score, fused_feature = fusion_module(dummy_img_feat, dummy_clin_feat)
    print("Fusion Module Classification Output Shape:", class_logits.shape)
    print("Risk Prediction Output Shape:", risk_score.shape)
    print("Fused Feature Shape:", fused_feature.shape)
