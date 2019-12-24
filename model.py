import torch
import torch.nn as nn
import torchvision.models as models

class MNISTClassifier(nn.Module):
    def __init__(self, pretrained = False, num_classes = 10):
        super(MNISTClassifier, self).__init__()

        self.num_classes = num_classes

        # vanilla model (inspired by LeNet) -- 97.65%
        '''self.model = nn.Sequential(
            nn.Conv2d(in_channels = 1, out_channels = 6, kernel_size = (5, 5), stride = 1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size = (2, 2), stride = 2),
            nn.Conv2d(in_channels = 6, out_channels = 16, kernel_size = (5, 5), stride = 1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size = (2, 2), stride = 2),
            nn.Conv2d(in_channels = 16, out_channels = 120, kernel_size = (4, 4), stride = 1),
            nn.Flatten(),
            nn.Linear(in_features = 120, out_features = 84, bias = True),
            nn.Tanh(),
            nn.Linear(in_features = 84, out_features = 10, bias = True)
        )'''

        # improved model (deeper version of LeNet) -- 99.2%
        self.model = nn.Sequential(
            nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size = (3, 3)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = (2, 2)),
            nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = (3, 3)),
            nn.ReLU(),
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (3, 3)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = (2, 2)),
            nn.Flatten(),
            nn.Linear(in_features = 1024, out_features = 100, bias = True),
            nn.ReLU(),
            nn.Linear(in_features = 100, out_features = 10, bias = True)
        )

    def forward(self, img):
        scores = self.model(img)
        return scores
