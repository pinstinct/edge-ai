import os.path

import torch
import torch.nn as nn

"""
Model Quantization: 모델 양자화
- 32비트 부동 소수점 숫자를 8비트 정수 또는 그 보다 더 낮은 정밀도 형식으로 모델 매개변수의 정밀도를 낮추는 것을 포함한다.
- 양자화는 8비트만 사용하여 메모리 요구 사항을 4배 줄이고 추론 시간을 단축하는 경우가 많다.
- 양자화 기술
  - Post-training Quantization(PTQ)
  - Quantization-Aware Training(QTA)
  - Dynamic Quantization: 모델 크기를 줄이고, CPU 추론 속도를 높이는 기법
  - Static Quantization 
"""


# 모델 정의
class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.fc1 = nn.Linear(100, 50)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.fc = nn.Linear(16 * 8 * 8, 10)  # 8x8 이미지 기준

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = torch.nn.functional.max_pool2d(x, 2)
        x = self.conv2(x)
        x = self.relu2(x)
        x = torch.nn.functional.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def run_model():
    # set quantization engine
    if torch.backends.quantized.engine == "none":
        torch.backends.quantized.engine = "qnnpack"
    print("Using quantization engine: ", torch.backends.quantized.engine)

    # init and save model
    linear_model = LinearModel()
    torch.save(linear_model, "linear_model.pth")

    conv_model = ConvNet()
    torch.save(conv_model, "cnn_model.pth")

    # load model
    # model = torch.load("large_model.pth")

    # dynamic quantization
    q_linear_model = quantized_model(linear_model)
    q_conv_model = quantized_model(conv_model)

    # save dynamic quantization
    torch.save(q_linear_model, "q_linear_model.pth")
    torch.save(q_conv_model, "q_conv_model.pth")

    # compare memory usage
    original_size = os.path.getsize("linear_model.pth")
    quantized_size = os.path.getsize("q_linear_model.pth")
    print(f"Original model size: {original_size / 1024:.2f} KB")
    print(f"Quantized model size: {quantized_size / 1024:.2f} KB")
    print(f"Memory reduction (모델 크기): {original_size / quantized_size:.2f}x")

    # load sample data for test
    x = torch.randn(1, 100)
    output_original = linear_model(x)
    output_quantized = q_linear_model(x)
    print("Original output:", output_original)
    print("Quantized output:", output_quantized)

    x = torch.randn(1, 3, 32, 32)  # 3채널, 32x32 이미지
    q_conv_model.eval()
    with torch.no_grad():
        output_fp32 = conv_model(x)
        output_int8 = q_conv_model(x)
    print("Output (FP32 model):", output_fp32)
    print("Output (Quantized INT8 model):", output_int8)
    diff = torch.mean(torch.abs(output_fp32 - output_int8))
    print(f"Average output difference (출력 차이): {diff.item():.6f}")


# Quantize the model to INT8
def quantized_model(model):
    return torch.quantization.quantize_dynamic(
        model,
        {nn.Linear, nn.Conv2d},  # layers to quantize, 신경망 모델 구축을 위한 레이어
        dtype=torch.qint8  # quantization data type
    )
