import torch
import torch.nn.functional as F
from torch import nn


class LeNet5(nn.Module):
    def __init__(self, drop):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.droput = nn.Dropout(drop)

    def forward(self, x):
        x = self.maxpool(F.relu(self.conv1(x), inplace=True))
        x = self.maxpool(F.relu(self.conv2(x), inplace=True))
        x = x.view(x.shape[0], -1)
        x = self.droput(x)
        x = F.relu(self.fc1(x))
        x = self.droput(x)
        x = F.relu(self.fc2(x))
        x = self.droput(x)
        output = self.fc3(x)

        return output


class LeNet5C(nn.Module):
    def __init__(self):
        super(LeNet5C, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv3 = nn.Conv2d(16, 120, 5)
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.maxpool(F.relu(self.conv1(x), inplace=True))
        x = self.maxpool(F.relu(self.conv2(x), inplace=True))
        x = F.relu(self.conv3(x), inplace=True)
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        output = self.fc2(x)
        
        return output


class LeNet5S(nn.Module):
    def __init__(self):
        super(LeNet5S, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64*4*4, 500)
        self.fc2 = nn.Linear(500, 10)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.maxpool(F.relu(self.conv1(x), inplace=True))
        x = self.maxpool(F.relu(self.conv2(x), inplace=True))
        x = self.maxpool(F.relu(self.conv3(x), inplace=True))
        # x = x.view(x.shape[0], -1)
        x = x.view(-1, 64 * 4 * 4)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        output = self.fc2(x)

        return output


class LeNet5f(nn.Module):
    def __init__(self, drop):
        super(LeNet5f, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 5)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 128, 5)
        self.fc1 = nn.Linear(128*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.dropout = nn.Dropout(drop)

    def forward(self, x):
        x = self.maxpool(F.relu(self.conv1(x), inplace=True))
        x = self.maxpool(F.relu(self.conv2(x), inplace=True))
        x = x.view(x.shape[0], -1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        output = self.fc3(x)

        return output

# if __name__ == "__main__":

#     model = LeNet5()

#     input_ = torch.randn(1, 3, 32, 32)

#     output1 = model(input_)
#     output2 = model.forward(input_)

#     print(output1)
#     print(output2)