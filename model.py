import torch.nn as nn 
import torch.nn.functional as F

class CRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        # self.conv7 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=2, stride=1, padding=0)
        self.conv7 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        

        self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.max_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.max_pool3 = nn.MaxPool2d(kernel_size=(1, 2), stride=2)
        self.max_pool4 = nn.MaxPool2d(kernel_size=(1, 2), stride=2)

        self.batch_norm1 = nn.BatchNorm2d(num_features=512)
        self.batch_norm2 = nn.BatchNorm2d(num_features=512)

        self.lstm1 = nn.LSTM(input_size=512, hidden_size=256, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=512, hidden_size=256, bidirectional=True, batch_first=True)

        self.avg_pool = nn.AdaptiveAvgPool1d(output_size=1)
            
        self.transcription = nn.Linear(in_features=512, out_features=26)

        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        x = self.max_pool1(F.relu(self.conv1(x)))
        x = self.max_pool2(F.relu(self.conv2(x)))

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        x = self.max_pool3(x)

        x = F.relu(self.conv5(x))
        x = self.batch_norm1(x)

        x = F.relu(self.conv6(x))
        x = self.batch_norm2(x)

        x = self.max_pool4(F.relu(self.conv7(x)))

        # feature map 2 sequence
        batch_size, channels, height, width = x.size()
        x = x.permute(0, 2, 3, 1).view(batch_size, height * width, channels)
        
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)

        x = x.permute(0, 2, 1)
        x = self.avg_pool(x)
        x = x.squeeze(-1)

        # transcription
        x = self.transcription(x)
        
        # probabilities
        x = self.log_softmax(x)

        return x
