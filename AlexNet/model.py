import torch
import torch.nn as nn
import torch.nn.functional as F

from module import *
import module


class AlexNet(nn.Module):

    def __init__(self, num_channels=3, num_classes=10):
        super(AlexNet, self).__init__()

        # original size 32x32
        self.conv1 = nn.Conv2d(num_channels, 32, kernel_size=3, padding=1, bias=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)                  # output[48, 27, 27]
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=True)           # output[128, 27, 27]
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)                  # output[128, 13, 13]
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=True)          # output[192, 13, 13]
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=True)          # output[192, 13, 13]
        self.relu4 = nn.ReLU(inplace=True)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=True)          # output[128, 13, 13]
        self.relu5 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.drop1 = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(256 * 3 * 3, 1024, bias=True)
        self.relu6 = nn.ReLU(inplace=True)
        self.drop2 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(1024, 512, bias=True)
        self.relu7 = nn.ReLU(inplace=True)
        self.fc3 = nn.Linear(512, num_classes, bias=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.conv5(x)
        x = self.relu5(x)
        x = self.pool5(x)

        x = torch.flatten(x, start_dim=1)

        x = self.drop1(x)
        x = self.fc1(x)
        x = self.relu6(x)
        x = self.drop2(x)
        x = self.fc2(x)
        x = self.relu7(x)
        x = self.fc3(x)
        return x

    def quantize(self, quant_type, num_bits=8, e_bits=3): 
        # e_bits仅当使用FLOAT量化时用到

        self.qconv1 = QConv2d(quant_type, self.conv1, qi=True, qo=True, num_bits=num_bits, e_bits=e_bits)
        self.qrelu1 = QReLU(quant_type, num_bits=num_bits, e_bits=e_bits)
        self.qpool1 = QMaxPooling2d(quant_type, kernel_size=2, stride=2, padding=0, num_bits=num_bits, e_bits=e_bits)
        self.qconv2 = QConv2d(quant_type, self.conv2, qi=False, qo=True, num_bits=num_bits, e_bits=e_bits)
        self.qrelu2 = QReLU(quant_type, num_bits=num_bits, e_bits=e_bits)
        self.qpool2 = QMaxPooling2d(quant_type, kernel_size=2, stride=2, padding=0, num_bits=num_bits, e_bits=e_bits)
        self.qconv3 = QConv2d(quant_type, self.conv3, qi=False, qo=True, num_bits=num_bits, e_bits=e_bits)
        self.qrelu3 = QReLU(quant_type, num_bits=num_bits, e_bits=e_bits)
        self.qconv4 = QConv2d(quant_type, self.conv4, qi=False, qo=True, num_bits=num_bits, e_bits=e_bits)
        self.qrelu4 = QReLU(quant_type, num_bits=num_bits, e_bits=e_bits)
        self.qconv5 = QConv2d(quant_type, self.conv5, qi=False, qo=True, num_bits=num_bits, e_bits=e_bits)
        self.qrelu5 = QReLU(quant_type, num_bits=num_bits, e_bits=e_bits)
        self.qpool5 = QMaxPooling2d(quant_type, kernel_size=3, stride=2, padding=0, num_bits=num_bits, e_bits=e_bits)

        self.qfc1 = QLinear(quant_type, self.fc1, qi=False, qo=True, num_bits=num_bits, e_bits=e_bits)
        self.qrelu6 = QReLU(quant_type, num_bits=num_bits, e_bits=e_bits)
        self.qfc2 = QLinear(quant_type, self.fc2, qi=False, qo=True, num_bits=num_bits, e_bits=e_bits)
        self.qrelu7 = QReLU(quant_type, num_bits=num_bits, e_bits=e_bits)
        self.qfc3 = QLinear(quant_type, self.fc3, qi=False, qo=True, num_bits=num_bits, e_bits=e_bits)

    def quantize_forward(self, x):
        x = self.qconv1(x)
        x = self.qrelu1(x)
        x = self.qpool1(x)
        x = self.qconv2(x)
        x = self.qrelu2(x)
        x = self.qpool2(x)
        x = self.qconv3(x)
        x = self.qrelu3(x)
        x = self.qconv4(x)
        x = self.qrelu4(x)
        x = self.qconv5(x)
        x = self.qrelu5(x)
        x = self.qpool5(x)

        x = torch.flatten(x, start_dim=1)

        x = self.drop1(x)
        x = self.qfc1(x)
        x = self.qrelu6(x)
        x = self.drop2(x)
        x = self.qfc2(x)
        x = self.qrelu7(x)
        x = self.qfc3(x)
        return x

    def freeze(self):
        self.qconv1.freeze()
        self.qrelu1.freeze(self.qconv1.qo)
        self.qpool1.freeze(self.qconv1.qo)
        self.qconv2.freeze(self.qconv1.qo)
        self.qrelu2.freeze(self.qconv2.qo)
        self.qpool2.freeze(self.qconv2.qo)
        self.qconv3.freeze(self.qconv2.qo)
        self.qrelu3.freeze(self.qconv3.qo)
        self.qconv4.freeze(self.qconv3.qo)
        self.qrelu4.freeze(self.qconv4.qo)
        self.qconv5.freeze(self.qconv4.qo)
        self.qrelu5.freeze(self.qconv5.qo)
        self.qpool5.freeze(self.qconv5.qo)
        self.qfc1.freeze(self.qconv5.qo)
        self.qrelu6.freeze(self.qfc1.qo)
        self.qfc2.freeze(self.qfc1.qo)
        self.qrelu7.freeze(self.qfc2.qo)
        self.qfc3.freeze(self.qfc2.qo)

    def quantize_inference(self, x):
        qx = self.qconv1.qi.quantize_tensor(x)
        qx = self.qconv1.quantize_inference(qx)
        qx = self.qrelu1.quantize_inference(qx)
        qx = self.qpool1.quantize_inference(qx)
        qx = self.qconv2.quantize_inference(qx)
        qx = self.qrelu2.quantize_inference(qx)
        qx = self.qpool2.quantize_inference(qx)
        qx = self.qconv3.quantize_inference(qx)
        qx = self.qrelu3.quantize_inference(qx)
        qx = self.qconv4.quantize_inference(qx)
        qx = self.qrelu4.quantize_inference(qx)
        qx = self.qconv5.quantize_inference(qx)
        qx = self.qrelu5.quantize_inference(qx)
        qx = self.qpool5.quantize_inference(qx)

        qx = torch.flatten(qx, start_dim=1)

        qx = self.qfc1.quantize_inference(qx)
        qx = self.qrelu6.quantize_inference(qx)
        qx = self.qfc2.quantize_inference(qx)
        qx = self.qrelu7.quantize_inference(qx)
        qx = self.qfc3.quantize_inference(qx)
        out = self.qfc3.qo.dequantize_tensor(qx)
        return out
