import torch
from models.cnn import CNNBranch
from models.vit import ViTBranch
from models.hybrid_model import HybridImagingModel
from models.clinical_model import ClinicalModel
from models.fusion import FusionModule
from models.multitask import MultiTaskHead

def test_full_pipeline():
    # Simulated ultrasound image (batch of 4, RGB, 640x640)
    img_input = torch.randn(4, 3, 640, 640)
    
    # Simulated clinical data (batch of 4, 20 features)
    clinical_input = torch.randn(4, 20)

    # Initialize models
    cnn = CNNBranch()
    vit = ViTBranch()
    hybrid = HybridImagingModel()
    clinical = ClinicalModel(input_dim=20)
    fusion = FusionModule()
    multitask = MultiTaskHead(input_dim=256)

    # Forward pass through models
    cnn_output = cnn(img_input)
    vit_output = vit(img_input)
    hybrid_output = hybrid(img_input)
    clinical_output = clinical(clinical_input)

    fused_output, fused_features = fusion(hybrid_output, clinical_output)

    class_logits, risk_score = multitask(fused_features)

    # Print shapes
    print("CNN Output Shape:", cnn_output.shape)
    print("ViT Output Shape:", vit_output.shape)
    print("Hybrid Model Output Shape:", hybrid_output.shape)
    print("Clinical Model Output Shape:", clinical_output.shape)
    print("Fusion Module Output Shape:", fused_output.shape)
    print("Fused Feature Shape:", fused_features.shape)
    print("Classification Output Shape:", class_logits.shape)
    print("Risk Prediction Output Shape:", risk_score.shape)

if __name__ == "__main__":
    test_full_pipeline()
