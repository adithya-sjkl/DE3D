import torch

from efficientnet_pytorch import EfficientNet
import torch

# Load EfficientNet (e.g., EfficientNet-B0)
efficientnet = EfficientNet.from_pretrained('efficientnet-b0')

class FeatureExtractor(nn.Module):
    def __init__(self, efficientnet):
        super(FeatureExtractor, self).__init__()
        self.efficientnet = efficientnet
        self.feature_blocks = nn.Sequential(*list(efficientnet.children())[:-2])  # Extract features after the last block

    def forward(self, x):
        return self.feature_blocks(x)
    
    image = torch.randn(224,224,3)

    print(image)