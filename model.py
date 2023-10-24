import torch
from torch import nn

def initialize_model(Pytorch_file_path):

    """Initializes the TinyVGG Model for Inference

    Args:
        Pytorch_file_path: A filepath to a pytorch model file (.pt or .pth)

    Returns:
        A pre-trained TinyVGG PyTorch model for inference

    Raises:
        FileNotFound: An error occurred accessing the directory.
    """

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    class TinyVGG(nn.Module):
        def __init__(self,
                    in_features,
                    out_features,
                    hidden_units):
            super().__init__()
            self.conv_block_1 = nn.Sequential(
                nn.Conv2d(in_channels=in_features,
                                    out_channels=hidden_units,
                                    kernel_size=3,
                                    padding=1,
                                    stride=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=hidden_units,
                        out_channels=hidden_units,
                        kernel_size=2,
                        padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2)
            )
            self.conv_block_2 = nn.Sequential(
                nn.Conv2d(in_channels=hidden_units,
                        out_channels=hidden_units,
                        kernel_size=3,
                        stride=1,
                        padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2)
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(in_features=7*7*hidden_units,
                        out_features=out_features)
            )

        def forward(self, X):
            X = self.conv_block_1(X)
            X = self.conv_block_2(X)
            X = self.classifier(X)
            return (X)

    model = TinyVGG(in_features=1,
                        out_features=10,
                        hidden_units=10).to(device)
    
    model.load_state_dict(torch.load(f=Pytorch_file_path,map_location=torch.device(device)))
    
    return model
                                    