import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiTaskHead(nn.Module):
    """
    Multi-task learning head that takes a fused feature vector and produces:
      - Classification output (e.g., tumor subtype)
      - Risk prediction output (a continuous value)
      - (Optional) Segmentation output (if spatial information is available)
    
    For this example, we'll implement classification and risk prediction.
    """
    def __init__(self, input_dim, num_classes=3, risk_output_dim=1):
        super(MultiTaskHead, self).__init__()
        # Shared fully connected layer for feature refinement
        self.shared_fc = nn.Linear(input_dim, 256)
        # Classification head
        self.classifier = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes)
        )
        # Risk prediction head
        self.risk_predictor = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(256, risk_output_dim)
        )
        # (Optional) You could add a segmentation head here if spatial output is needed.
    
    def forward(self, x):
        shared_features = F.relu(self.shared_fc(x))
        class_logits = self.classifier(shared_features)
        risk_score = self.risk_predictor(shared_features)
        return class_logits, risk_score

if __name__ == "__main__":
    model = MultiTaskHead(input_dim=256, num_classes=3, risk_output_dim=1)
    dummy_input = torch.randn(4, 256)
    class_logits, risk_score = model(dummy_input)
    print("Classification Output Shape:", class_logits.shape)  # Expected: (4, 3)
    print("Risk Prediction Output Shape:", risk_score.shape)    # Expected: (4, 1)
