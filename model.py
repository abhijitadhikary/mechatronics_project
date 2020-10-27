import torchvision
import torch
import torch.nn as nn
import numpy as np


class CoarseModule(nn.Module):
    def __init__(self, image_size=224, num_channels=3, num_keypoints=16):
        super(CoarseModule, self).__init__()
        rennet18 = torchvision.models.resnet18(pretrained=True)
        layers_rennet18 = list(rennet18.children())[:6]
        self.backbone_layer = nn.Sequential(*layers_rennet18)

        num_channels_resnet_out = 128

        self.conv_up_1 = nn.ConvTranspose2d(in_channels=num_channels_resnet_out, out_channels=num_channels_resnet_out // 2, kernel_size=2, stride=2)
        self.bn_up_1 = nn.BatchNorm2d(num_channels_resnet_out // 2)

        self.conv_up_2 = nn.ConvTranspose2d(in_channels=num_channels_resnet_out // 2, out_channels=num_channels_resnet_out // 4, kernel_size=2, stride=2)
        self.bn_up_2 = nn.BatchNorm2d(num_channels_resnet_out // 4)

        self.conv_up_3 = nn.ConvTranspose2d(in_channels=num_channels_resnet_out // 4, out_channels=num_keypoints, kernel_size=2, stride=2)

        self.relu = nn.ReLU()

        self.dropout = nn.Dropout2d(0.5)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.backbone_layer(x)
        x = self.relu(self.bn_up_1(self.dropout(self.conv_up_1(x))))
        x = self.relu(self.bn_up_2(self.dropout(self.conv_up_2(x))))
        x = self.sigmoid(self.conv_up_3(x))
        return x


class FineModule(nn.Module):
    def __init__(self, image_size=224, num_channels=3, num_keypoints=16, num_features=64):
        super(FineModule, self).__init__()

        self.conv_down_1 = nn.Conv2d(in_channels=num_channels + num_keypoints, out_channels=num_features, kernel_size=3, stride=2, padding=1)
        self.conv_down_2 = nn.Conv2d(in_channels=num_features, out_channels=num_features * 2, kernel_size=3, stride=2, padding=1)
        self.conv_down_3 = nn.Conv2d(in_channels=num_features * 2, out_channels=num_features * 4, kernel_size=3, stride=2, padding=1)
        self.conv_down_4 = nn.Conv2d(in_channels=num_features * 4, out_channels=num_features * 8, kernel_size=3, stride=2, padding=1)

        self.bottleneck_1 = nn.Conv2d(in_channels=num_features * 8, out_channels=num_features * 8, kernel_size=1, stride=1, padding=0)
        self.bottleneck_2 = nn.Conv2d(in_channels=num_features * 8, out_channels=num_features * 8, kernel_size=1, stride=1, padding=0)
        self.bottleneck_3 = nn.Conv2d(in_channels=num_features * 8, out_channels=num_features * 8, kernel_size=1, stride=1, padding=0)

        self.conv_up_4 = nn.ConvTranspose2d(in_channels=num_features * 8, out_channels=num_features * 4, kernel_size=2, stride=2, padding=0)
        self.conv_up_3 = nn.ConvTranspose2d(in_channels=num_features * 4 * 2, out_channels=num_features * 2 * 2, kernel_size=2, stride=2, padding=0)
        self.conv_up_2 = nn.ConvTranspose2d(in_channels=num_features * 2 * 2+num_features*2, out_channels=num_features * 2, kernel_size=2, stride=2, padding=0)
        self.conv_up_1 = nn.ConvTranspose2d(in_channels=num_features * 2+num_features, out_channels=num_keypoints, kernel_size=2, stride=2, padding=0)

        self.bn_down_1 = nn.BatchNorm2d(num_features)
        self.bn_down_2 = nn.BatchNorm2d(num_features*2)
        self.bn_down_3 = nn.BatchNorm2d(num_features*4)
        self.bn_down_4 = nn.BatchNorm2d(num_features*8)

        self.bn_bottleneck_1 = nn.BatchNorm2d(num_features*8)
        # self.bn_bottleneck_2 = nn.BatchNorm2d(num_features*8)

        self.bn_up_4 = nn.BatchNorm2d(num_features*4)
        self.bn_up_3 = nn.BatchNorm2d(num_features*2*2)
        self.bn_up_2 = nn.BatchNorm2d(num_features*2)


        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x):
        down_1 = self.leaky_relu(self.bn_down_1(self.conv_down_1(x)))
        down_2 = self.leaky_relu(self.bn_down_2(self.conv_down_2(down_1)))
        down_3 = self.leaky_relu(self.bn_down_3(self.conv_down_3(down_2)))
        down_4 = self.leaky_relu(self.bn_down_4(self.conv_down_4(down_3)))

        bottleneck_1 = self.leaky_relu(self.bn_bottleneck_1(self.bottleneck_1(down_4)))
        bottleneck_2 = self.leaky_relu(self.bn_bottleneck_1(self.bottleneck_2(bottleneck_1)))
        bottleneck_3 = self.leaky_relu(self.bn_bottleneck_1(self.bottleneck_3(bottleneck_2)))

        up_3 = self.relu(self.bn_up_4(self.conv_up_4(bottleneck_3)))

        up_3 = torch.cat((up_3, down_3), dim=1)

        up_2 = self.relu(self.bn_up_3(self.conv_up_3(up_3)))

        up_2 = torch.cat((up_2, down_2), dim=1)

        up_1 = self.relu(self.bn_up_2(self.conv_up_2(up_2)))

        up_1 = torch.cat((up_1, down_1), dim=1)

        out = self.sigmoid(self.conv_up_1(up_1))

        return out

class KeypointNet(nn.Module):

    def __init__(self, image_size=224, num_channels=3, num_keypoints=16):
        super(KeypointNet, self).__init__()
        self.coarse_module = CoarseModule(image_size, num_channels, num_keypoints)
        self.fine_module = FineModule(image_size, num_channels, num_keypoints)

    def forward(self, input):
        heatmaps_coarse = self.coarse_module(input)

        heatmaps_concat = torch.cat((input, heatmaps_coarse), dim=1)
        heatmaps_fine = self.fine_module(heatmaps_concat)

        # coordinates = self.block_out(out_block_1)

        return heatmaps_coarse, heatmaps_fine

class FCN_Resnet101(nn.Module):
    def __init__(self, image_size=224, num_channels=3, num_keypoints=16, device=torch.device('cuda')):
        super(FCN_Resnet101, self).__init__()
        self.coarse_module = torchvision.models.segmentation.fcn_resnet101(pretrained=True, progress=True, num_classes=21, aux_loss=None).to(device)
        self.coarse_module.classifier[4] = nn.Conv2d(in_channels=512, out_channels=16, kernel_size=1, stride=1).to(device)
        self.coarse_module.aux_classifier[4] = nn.Conv2d(in_channels=256, out_channels=16, kernel_size=1, stride=1).to(device)

        self.fine_module = torchvision.models.segmentation.fcn_resnet101(pretrained=True, progress=True, num_classes=21, aux_loss=None).to(device).to(device)
        self.fine_module.backbone.conv1 = nn.Conv2d(in_channels=19, out_channels=64, kernel_size=6, stride=2, padding=3, bias=False)
        self.fine_module.classifier[4] = nn.Conv2d(in_channels=512, out_channels=16, kernel_size=1, stride=1).to(device)
        self.fine_module.aux_classifier[4] = nn.Conv2d(in_channels=256, out_channels=16, kernel_size=1, stride=1).to(device)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        heatmaps_coarse = self.sigmoid(self.coarse_module(input)['out'])

        heatmaps_concat = torch.cat((input, heatmaps_coarse), dim=1)
        heatmaps_fine = self.sigmoid(self.fine_module(heatmaps_concat)['out'])

        # coordinates = self.block_out(out_block_1)

        return heatmaps_coarse, heatmaps_fine