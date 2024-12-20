import torch
import torch.nn.functional as F

class PolicyNet(torch.nn.Module):
    ''' MLP策略网 '''
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)


class ValueNet(torch.nn.Module):
    ''' MLP价值网 '''
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)
    

# class ConvPolicyNet(torch.nn.Module):
#     ''' 卷积策略网 '''
#     def __init__(self, input_channels, action_dim):
#         super(ConvPolicyNet, self).__init__()
#         # 定义卷积层
#         self.conv1 = torch.nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
#         self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, padding=1)
#         self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=3, padding=1)
#         # self.conv4 = torch.nn.Conv2d(128, 128, kernel_size=3, padding=1)
        
#         # 添加批归一化层
#         self.bn1 = torch.nn.BatchNorm2d(32)
#         self.bn2 = torch.nn.BatchNorm2d(64)
#         self.bn3 = torch.nn.BatchNorm2d(128)
#         # self.bn4 = torch.nn.BatchNorm2d(128)
        
#         # 全连接层
#         self.fc1 = torch.nn.Linear(128 * 7 * 7, 512)
#         self.fc2 = torch.nn.Linear(512, 256)
#         self.fc3 = torch.nn.Linear(256, action_dim)
        
#         # Dropout层防止过拟合
#         self.dropout = torch.nn.Dropout(0.3)

#     def forward(self, x):
#         # 卷积层 + 批归一化 + ReLU
#         x = F.relu(self.bn1(self.conv1(x)))
#         x = F.relu(self.bn2(self.conv2(x)))
#         x = F.relu(self.bn3(self.conv3(x)))
#         # x = F.relu(self.bn4(self.conv4(x)))
        
#         x = x.view(x.size(0), -1)  # 展平
        
#         # 全连接层
#         x = F.relu(self.fc1(x))
#         x = self.dropout(x)
#         x = F.relu(self.fc2(x))
#         x = self.dropout(x)
#         return F.softmax(self.fc3(x), dim=1)

class ConvPolicyNet(torch.nn.Module):
    ''' 卷积策略网 '''
    def __init__(self, input_channels, action_dim):
        super(ConvPolicyNet, self).__init__()
        # 定义卷积层
        self.conv1 = torch.nn.Conv2d(input_channels, 16, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = torch.nn.Linear(32 * 7 * 7, 128)  # 棋盘大小为7x7
        self.fc2 = torch.nn.Linear(128, action_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        # x = torch.flatten(x, start_dim=0)  # 展平
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)


class ConvValueNet(torch.nn.Module):
    ''' 卷积价值网 '''
    def __init__(self, input_channels):
        super(ConvValueNet, self).__init__()
        # 定义卷积层
        self.conv1 = torch.nn.Conv2d(input_channels, 16, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = torch.nn.Linear(32 * 7 * 7, 128)  # 棋盘大小为7x7
        self.fc2 = torch.nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        # x = torch.flatten(x, start_dim=0)  # 展平
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
