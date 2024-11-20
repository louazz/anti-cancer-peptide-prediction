import torch
import torch.nn as nn


class Block(nn.Module):
    def __init__(self, in_channel):
        super().__init__()

        self.fc1 = nn.Linear(in_channel, 44)
        self.elu = nn.ELU()

        self.fc2 = nn.Linear(44, in_channel)
        self.elu2 = nn.ELU()

    def forward(self, inputs):
        x = self.fc1(inputs)
        x = self.elu(x)

        x = self.fc2(x)
        x = self.elu2(x)

        return x


class NeuralNetwork(nn.Module):
    def __init__(self, input_features, output_channel):
        super().__init__()

        self.fc1 = nn.Linear(input_features, 44)
        self.bn1 = nn.BatchNorm1d(44)
        self.elu1 = nn.ELU()

        self.fc2 = nn.Linear(44, 22)
        self.bn2 = nn.BatchNorm1d(22)
        self.elu2 = nn.ELU()

        self.dropout = nn.Dropout(0.3)

        self.block = Block(input_features)

        self.fc3 = nn.Linear(110, 55)
        self.bn3 = nn.BatchNorm1d(55)
        self.elu3 = nn.ELU()

        self.fc4 = nn.Linear(55, 11)
        self.bn4 = nn.BatchNorm1d(11)
        self.elu4 = nn.ELU()

        self.fc5 = nn.Linear(11, output_channel)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        x = self.fc1(inputs)
        x = self.bn1(x)
        x = self.elu1(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.elu2(x)

        y = self.block(inputs)

        x = torch.cat([x, y], axis=1)

        x = self.fc3(x)
        x = self.bn3(x)
        x = self.elu3(x)

        x = self.fc4(x)
        x = self.bn4(x)
        x = self.elu4(x)

        x = self.fc5(x)
        x = self.sigmoid(x)
        return x
