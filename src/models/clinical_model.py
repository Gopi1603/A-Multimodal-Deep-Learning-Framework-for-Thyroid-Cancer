import torch
import torch.nn as nn
import torch.nn.functional as F

class ClinicalModel(nn.Module):
    """
    MLP for processing clinical data.
    Input: Feature vector from clinical CSV data.
    Architecture: Two hidden layers with ReLU activations, projecting to a 32-dimensional vector.
    """
    def __init__(self, input_dim, hidden_dim1=128, hidden_dim2=64, output_dim=32):
        super(ClinicalModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, output_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

if __name__ == "__main__":
    # Example: suppose clinical data has 20 features
    model = ClinicalModel(input_dim=20)
    dummy_input = torch.randn(4, 20)
    output = model(dummy_input)
    print("Clinical Model Output Shape:", output.shape)  # Expected: (4, 32)
