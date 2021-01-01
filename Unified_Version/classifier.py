from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, input_channels, input_height, input_width, num_classes):
        super().__init__()
        self.num_classes = num_classes
        
        self.conv_1 = nn.Conv2d(in_channels=input_channels,
                            out_channels=32,
                            kernel_size=5, 
                            padding=2)
        self.batchnorm1 = nn.BatchNorm2d(32)
        self.conv_2 = nn.Conv2d(in_channels=32, 
                            out_channels=64,
                            kernel_size=5, 
                            padding=2)
        self.batchnorm2 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        
        self.conv2_drop = nn.Dropout2d()

        self.dense_features = 64 * input_height//2 * input_width//2
        self.dense = nn.Linear(in_features=self.dense_features, 
                                out_features=50,
                                bias=True)
        self.dropout = nn.Dropout(p=0.5)
        
        self.dense_out = nn.Linear(in_features=50,
                             out_features=num_classes,
                                bias=False)

    def forward(self, x):
        x = F.relu(self.conv_1(x))
        x = self.batchnorm1(x)
        x = F.relu(self.conv_2(x))
        x = self.batchnorm2(x)
        x = self.maxpool(x)

        x = self.conv2_drop(x)
        x = x.view(-1, self.dense_features)
        x = F.relu(self.dense(x))

        return F.log_softmax(self.dense_out(x), dim=1)
