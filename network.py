import torch
import torch.nn.functional as F
from torch import nn


class CAE(nn.Module):
    def __init__(self):
        super(CAE, self).__init__()
        self.conv1a = nn.Conv2d(3, 32, (3, 3), padding=1)
        self.conv2a = nn.Conv2d(32, 64, (3, 3), padding=1)
        self.conv3a = nn.Conv2d(64, 128, (3, 3), padding=1)
        self.conv4a = nn.Conv2d(128, 256, (3, 3), padding=1)
        self.conv5a = nn.Conv2d(256, 512, (3, 3), padding=1)
        self.maxpool = nn.MaxPool2d((2, 2))
        self.deconv5a = nn.ConvTranspose2d(512, 256, (3, 3), padding=1)
        self.deconv4a = nn.ConvTranspose2d(256, 128, (3, 3), padding=1)
        self.deconv3a = nn.ConvTranspose2d(128, 64, (3, 3), padding=1)
        self.deconv2a = nn.ConvTranspose2d(64, 32, (3, 3), padding=1)
        self.deconv1a = nn.ConvTranspose2d(32, 3, (3, 3), padding=1)

    def forward(self, x):
        x = self.maxpool(F.relu(self.conv1a(x), inplace=True))
        x = self.maxpool(F.relu(self.conv2a(x), inplace=True))
        x = self.maxpool(F.relu(self.conv3a(x), inplace=True))
        x = self.maxpool(F.relu(self.conv4a(x), inplace=True))
        x = self.maxpool(F.relu(self.conv5a(x), inplace=True))
        x = self.deconv5a(F.interpolate(x, scale_factor=2))
        x = self.deconv4a(F.interpolate(x, scale_factor=2))
        x = self.deconv3a(F.interpolate(x, scale_factor=2))
        x = self.deconv2a(F.interpolate(x, scale_factor=2))
        x = self.deconv1a(F.interpolate(x, scale_factor=2))

        return x


class CAESKC(nn.Module):
    def __init__(self):
        self.skip_con = []
        super(CAESKC, self).__init__()
        self.conv1a = nn.Conv2d(3, 32, (3, 3), padding=1)
        self.conv2a = nn.Conv2d(32, 64, (3, 3), padding=1)
        self.conv3a = nn.Conv2d(64, 128, (3, 3), padding=1)
        self.conv4a = nn.Conv2d(128, 256, (3, 3), padding=1)
        self.conv5a = nn.Conv2d(256, 512, (3, 3), padding=1)
        self.maxpool = nn.MaxPool2d((2, 2))
        self.deconv5a = nn.ConvTranspose2d(512+256, 256, (3, 3), padding=1)
        self.deconv4a = nn.ConvTranspose2d(256+128, 128, (3, 3), padding=1)
        self.deconv3a = nn.ConvTranspose2d(128+64, 64, (3, 3), padding=1)
        self.deconv2a = nn.ConvTranspose2d(64+32, 32, (3, 3), padding=1)
        self.deconv1a = nn.ConvTranspose2d(32, 3, (3, 3), padding=1)

    def forward(self, x):
        x = self.maxpool(F.relu(self.conv1a(x), inplace=True))
        self.skip_con.append(x)
        x = self.maxpool(F.relu(self.conv2a(x), inplace=True))
        self.skip_con.append(x)
        x = self.maxpool(F.relu(self.conv3a(x), inplace=True))
        self.skip_con.append(x)
        x = self.maxpool(F.relu(self.conv4a(x), inplace=True))
        self.skip_con.append(x)
        x = self.maxpool(F.relu(self.conv5a(x), inplace=True))

        x = torch.cat((self.skip_con.pop(), F.interpolate(x, scale_factor=2)), dim=1)
        x = self.deconv5a(x)
        x = torch.cat((self.skip_con.pop(), F.interpolate(x, scale_factor=2)), dim=1)
        x = self.deconv4a(x)
        x = torch.cat((self.skip_con.pop(), F.interpolate(x, scale_factor=2)), dim=1)
        x = self.deconv3a(x)
        x = torch.cat((self.skip_con.pop(), F.interpolate(x, scale_factor=2)), dim=1)
        x = self.deconv2a(x)
        x = self.deconv1a(F.interpolate(x, scale_factor=2))

        return x


class SUperNet(nn.Module):
    def __init__(self):
        self.skip_con = []
        super(SUperNet, self).__init__()
        self.conv1a = nn.Conv2d(3, 32, (3, 3), padding=1)
        self.conv2a = nn.Conv2d(32, 48, (3, 3), padding=1)
        self.conv3a = nn.Conv2d(48, 48, (3, 3), padding=1)
        self.conv4a = nn.Conv2d(48, 96, (3, 3), padding=1)
        self.conv5a = nn.Conv2d(96, 96, (3, 3), padding=1)
        self.maxpool = nn.MaxPool2d((2, 2))
        self.deconv5a = nn.ConvTranspose2d(96+96, 96, (3, 3), padding=1)
        self.deconv4a = nn.ConvTranspose2d(96+48, 48, (3, 3), padding=1)
        self.deconv3a = nn.ConvTranspose2d(48+48, 48, (3, 3), padding=1)
        self.deconv2a = nn.ConvTranspose2d(48+32, 32, (3, 3), padding=1)
        self.deconv1a = nn.ConvTranspose2d(32, 3, (3, 3), padding=1)
        self.fc1a = nn.Linear(96, 10)

    def forward(self, x):
        # Encoder
        x = self.maxpool(F.relu(self.conv1a(x)))
        self.skip_con.append(x)
        x = self.maxpool(F.relu(self.conv2a(x)))
        self.skip_con.append(x)
        x = self.maxpool(F.relu(self.conv3a(x)))
        self.skip_con.append(x)
        x = self.maxpool(F.relu(self.conv4a(x)))
        self.skip_con.append(x)
        x = self.maxpool(F.relu(self.conv5a(x)))
        # CLass scores
        scores = self.fc1a(x.view(x.shape[0], -1))
        # Decoder
        x = torch.cat((self.skip_con.pop(), F.interpolate(x, scale_factor=2)), dim=1)
        x = self.deconv5a(x)
        x = torch.cat((self.skip_con.pop(), F.interpolate(x, scale_factor=2)), dim=1)
        x = self.deconv4a(x)
        x = torch.cat((self.skip_con.pop(), F.interpolate(x, scale_factor=2)), dim=1)
        x = self.deconv3a(x)
        x = torch.cat((self.skip_con.pop(), F.interpolate(x, scale_factor=2)), dim=1)
        x = self.deconv2a(x)
        x = self.deconv1a(F.interpolate(x, scale_factor=2))

        return x, scores


class SUperNet2(nn.Module):
    def __init__(self):
        self.skip_con = []
        super(SUperNet2, self).__init__()
        self.conv1a = nn.Conv2d(3, 32, (3, 3), padding=1)
        self.conv2a = nn.Conv2d(32, 48, (3, 3), padding=1)
        self.conv3a = nn.Conv2d(48, 48, (3, 3), padding=1)
        self.conv4a = nn.Conv2d(48, 96, (3, 3), padding=1)
        self.conv5a = nn.Conv2d(96, 96, (3, 3), padding=1)
        self.conv1x = nn.Conv2d(96, 10, (1, 1))
        self.maxpool = nn.MaxPool2d((2, 2))
        self.deconv5a = nn.ConvTranspose2d(96+96, 96, (3, 3), padding=1)
        self.deconv4a = nn.ConvTranspose2d(96+48, 48, (3, 3), padding=1)
        self.deconv3a = nn.ConvTranspose2d(48+48, 48, (3, 3), padding=1)
        self.deconv2a = nn.ConvTranspose2d(48+32, 32, (3, 3), padding=1)
        self.deconv1a = nn.ConvTranspose2d(32, 3, (3, 3), padding=1)
        self.fc1a = nn.Linear(96, 10)

    def forward(self, x):
        # Encoder
        x = self.maxpool(F.relu(self.conv1a(x)))
        self.skip_con.append(x)
        x = self.maxpool(F.relu(self.conv2a(x)))
        self.skip_con.append(x)
        x = self.maxpool(F.relu(self.conv3a(x)))
        self.skip_con.append(x)
        x = self.maxpool(F.relu(self.conv4a(x)))
        self.skip_con.append(x)
        x = self.maxpool(F.relu(self.conv5a(x)))
        # CLass scores
        scores = F.interpolate(self.conv1x(x), scale_factor=32)
        # Decoder
        x = torch.cat((self.skip_con.pop(), F.interpolate(x, scale_factor=2)), dim=1)
        x = self.deconv5a(x)
        x = torch.cat((self.skip_con.pop(), F.interpolate(x, scale_factor=2)), dim=1)
        x = self.deconv4a(x)
        x = torch.cat((self.skip_con.pop(), F.interpolate(x, scale_factor=2)), dim=1)
        x = self.deconv3a(x)
        x = torch.cat((self.skip_con.pop(), F.interpolate(x, scale_factor=2)), dim=1)
        x = self.deconv2a(x)
        x = self.deconv1a(F.interpolate(x, scale_factor=2))
        x = torch.cat([x, scores], dim=1)

        return x


class SUperNet3(nn.Module):
    def __init__(self):
        self.skip_con = []
        super(SUperNet3, self).__init__()
        self.conv1a = nn.Conv2d(3, 32, (3, 3), padding=1)
        self.bn1a = nn.BatchNorm2d(32)
        self.conv2a = nn.Conv2d(32, 48, (3, 3), padding=1)
        self.bn2a = nn.BatchNorm2d(48)
        self.conv3a = nn.Conv2d(48, 48, (3, 3), padding=1)
        self.bn3a = nn.BatchNorm2d(48)
        self.conv4a = nn.Conv2d(48, 96, (3, 3), padding=1)
        self.bn4a = nn.BatchNorm2d(96)
        self.conv5a = nn.Conv2d(96, 96, (3, 3), padding=1)
        self.bn5a = nn.BatchNorm2d(96)
        self.conv1x = nn.Conv2d(96, 10, (1, 1))
        self.maxpool = nn.MaxPool2d((2, 2))
        self.deconv5a = nn.ConvTranspose2d(96+96, 96, (3, 3), padding=1)
        self.deconv4a = nn.ConvTranspose2d(96+48, 48, (3, 3), padding=1)
        self.deconv3a = nn.ConvTranspose2d(48+48, 48, (3, 3), padding=1)
        self.deconv2a = nn.ConvTranspose2d(48+32, 32, (3, 3), padding=1)
        self.deconv1a = nn.ConvTranspose2d(32, 3, (3, 3), padding=1)
        self.fc1a = nn.Linear(96, 10)

    def forward(self, x):
        # Encoder
        x = self.maxpool(F.relu(self.bn1a(self.conv1a(x))))
        self.skip_con.append(x)
        x = self.maxpool(F.relu(self.bn2a(self.conv2a(x))))
        self.skip_con.append(x)
        x = self.maxpool(F.relu(self.bn3a(self.conv3a(x))))
        self.skip_con.append(x)
        x = self.maxpool(F.relu(self.bn4a(self.conv4a(x))))
        self.skip_con.append(x)
        x = self.maxpool(F.relu(self.bn5a(self.conv5a(x))))
        # CLass scores
        scores = F.interpolate(self.conv1x(x), scale_factor=32)
        # Decoder
        x = torch.cat((self.skip_con.pop(), F.interpolate(x, scale_factor=2)), dim=1)
        x = self.deconv5a(x)
        x = torch.cat((self.skip_con.pop(), F.interpolate(x, scale_factor=2)), dim=1)
        x = self.deconv4a(x)
        x = torch.cat((self.skip_con.pop(), F.interpolate(x, scale_factor=2)), dim=1)
        x = self.deconv3a(x)
        x = torch.cat((self.skip_con.pop(), F.interpolate(x, scale_factor=2)), dim=1)
        x = self.deconv2a(x)
        x = self.deconv1a(F.interpolate(x, scale_factor=2))
        x = torch.cat([x, scores], dim=1)

        return x


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight)
    if isinstance(m, nn.ConvTranspose2d):
        nn.init.kaiming_normal_(m.weight)
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)

# model.apply(weights_init)
# if __name__ == "__main__":

#     model = LeNet5()

#     input_ = torch.randn(1, 3, 32, 32)

#     output1 = model(input_)
#     output2 = model.forward(input_)

#     print(output1)
#     print(output2)