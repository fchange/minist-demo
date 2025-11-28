#!/usr/bin/env python3
"""
MNIST 模型训练脚本
训练一个简单的CNN模型并导出为ONNX格式
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'使用设备: {device}')

# 定义CNN模型
class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        # 第一个卷积层: 1 -> 32 通道
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        # 第二个卷积层: 32 -> 64 通道
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        # 池化层
        self.pool = nn.MaxPool2d(2, 2)
        # Dropout
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        # 全连接层
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # 第一个卷积块
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.pool(x)  # 28x28 -> 14x14

        # 第二个卷积块
        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.pool(x)  # 14x14 -> 7x7

        x = self.dropout1(x)

        # 展平
        x = x.view(-1, 64 * 7 * 7)

        # 全连接层
        x = torch.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)

        return x


def train_model():
    # 数据预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # 加载数据集
    print('正在下载/加载 MNIST 数据集...')
    train_dataset = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    # 创建模型
    model = MNISTNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练
    epochs = 5
    print(f'\n开始训练 ({epochs} 个 epoch)...')

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if batch_idx % 200 == 199:
                print(f'  Epoch {epoch + 1}, Batch {batch_idx + 1}, Loss: {running_loss / 200:.4f}')
                running_loss = 0.0

        # 测试
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        accuracy = 100 * correct / total
        print(f'Epoch {epoch + 1} 完成, 测试准确率: {accuracy:.2f}%')

    return model


def export_onnx(model, filename='mnist.onnx'):
    """导出模型为ONNX格式（使用传统方法确保单文件）"""
    model.eval()
    model.to('cpu')

    # 创建示例输入
    dummy_input = torch.randn(1, 1, 28, 28)

    # 导出为ONNX - 使用 dynamo=False 确保生成单个文件
    print(f'\n正在导出模型到 {filename}...')

    # 删除旧文件
    if os.path.exists(filename):
        os.remove(filename)
    if os.path.exists(filename + '.data'):
        os.remove(filename + '.data')

    torch.onnx.export(
        model,
        dummy_input,
        filename,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        },
        dynamo=False  # 使用传统导出方法，生成单个文件
    )

    # 验证导出的模型
    import onnx
    onnx_model = onnx.load(filename)
    onnx.checker.check_model(onnx_model)

    file_size = os.path.getsize(filename) / 1024
    print(f'模型导出成功! 文件大小: {file_size:.1f} KB')

    return filename


def main():
    print('=' * 50)
    print('MNIST CNN 模型训练与导出')
    print('=' * 50)

    # 训练模型
    model = train_model()

    # 导出为ONNX
    export_onnx(model, 'mnist.onnx')

    print('\n' + '=' * 50)
    print('完成! 模型已保存为 mnist.onnx')
    print('现在可以在浏览器中使用 ONNX Runtime Web 加载此模型')
    print('=' * 50)


if __name__ == '__main__':
    main()
