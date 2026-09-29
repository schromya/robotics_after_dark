import torch
import torch.nn as nn
from torchvision.models.resnet import ResNet, BasicBlock


class BinaryClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        # TODO (@schromya): hil-serl uses a more complex backbone (learned pooling).
        #                   May need to upgrade later.
        self.backbone = ResNet(BasicBlock, [1, 1, 1, 1])  # Resnet-10
        feature_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
    
        self.head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.Dropout(0.1),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)


model = BinaryClassifier(num_outputs=1)
images = torch.randn(2, 3, 224, 224)
outputs = model(images)
print(outputs.shape)