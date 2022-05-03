import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import alexnet


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv2d_drop = nn.Dropout(p=0.2)
        self.fc1 = nn.Linear(16 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 100)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        # x = self.pool(F.relu(self.conv2d_drop(self.conv1(x))))
        # x = self.pool(F.leaky_relu(self.conv2d_drop(self.conv1(x))))
        x = self.pool(F.relu(self.conv2(x)))
        # x = self.pool(F.relu(self.conv2d_drop(self.conv2(x))))
        # x = self.pool(F.leaky_relu(self.conv2d_drop(self.conv2(x))))
        x = x.view(-1, 16 * 4 * 4)
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = self.fc3(x)
        return x


class SoftmaxClassifier(nn.Module):
    def __init__(self):
        super(SoftmaxClassifier, self).__init__()
        self.fc = nn.Linear(28 * 28 * 3, 14, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_().mul_(0.005)

    def forward(self, x):
        x = x.view(-1, 28 * 28 * 3)
        x = self.fc(x)
        return x

class Softmax20(nn.Module):
    def __init__(self):
        super(Softmax20, self).__init__()
        self.fc = nn.Linear(20 * 20 * 3, 14, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_().mul_(0.005)

    def forward(self, x):
        x = x.view(-1, 20 * 20 * 3)
        x = self.fc(x)
        return x

class AlexNetSoftMax(nn.Module):
    # https://github.com/hemanthmayaluru/Image-Classification-using-CNNs-AlexNet-VGG16-SVM-transfer-learning
    def __init__(self):
        super(AlexNetSoftMax, self).__init__()
        self.cnn_layers = nn.Sequential()
        self.fc_layers = nn.Sequential()
        self.loss_criterion = None

        model_ft = alexnet(pretrained = True)
        self.cnn_layers = model_ft.features 
        self.fc_layers = model_ft.classifier
        self.fc_layers[-1] = nn.Linear(4096, 15)

        # for layer in self.cnn_layers:
        #     if isinstance(layer, nn.Conv2d):
        #         layer.weight.requires_grad = False
        #         layer.bias.requires_grad = False

        # for layer in self.fc_layers:
        #     if isinstance(layer, nn.Linear):
        #         if layer.out_features == 15:
        #             continue
        #         layer.weight.requires_grad = False
        #         layer.bias.requires_grad = False
        for param in self.cnn_layers.parameters():
                param.requires_grad = False

        
    def forward(self, x):
        output = self.cnn_layers(x)
        output = output.view(output.shape[0], -1)
        model_output = self.fc_layers(output)
        return model_output

class AlexNetSVM(nn.Module):
    def __init__(self):
        super(AlexNetSVM, self).__init__()
        self.cnn_layers = nn.Sequential()
        self.fc_layers = nn.Sequential()
        self.loss_criterion = None

        model_ft = alexnet(pretrained = True)
        self.cnn_layers = model_ft.features 
        # Exclude the last fully connected layer
        self.fc_layers = model_ft.classifier[:-1]
        # self.fc_layers[-1] = nn.Linear(4096, 15)

        for param in self.cnn_layers.parameters():
                param.requires_grad = False

        
    def forward(self, x):
        output = self.cnn_layers(x)
        output = output.view(output.shape[0], -1)
        model_output = self.fc_layers(output)
        return model_output