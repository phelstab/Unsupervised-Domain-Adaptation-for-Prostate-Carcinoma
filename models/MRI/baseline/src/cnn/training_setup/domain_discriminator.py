import torch.nn as nn


class DomainDiscriminator(nn.Module):
    def __init__(self, feature_dim=512, hidden_dim=256):
        super().__init__()
        
        self.discriminator = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim // 2, 2)
        )
    
    def forward(self, features):
        return self.discriminator(features)
