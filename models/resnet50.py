import torch
import torch.nn as nn
from torchvision import models

class CustomResNet50(nn.Module):
    def __init__(self, num_classes=4):
        super(CustomResNet50, self).__init__()
        # ResNet50モデルの読み込み、事前学習済みモデルを使用（今回は使わない設定）
        self.model = models.resnet50(pretrained=False)
        
        # 元の最初の畳み込み層を取得
        original_first_layer = self.model.conv1
        
        # 1チャンネル入力用の新しい畳み込み層を作成
        self.model.conv1 = nn.Conv2d(1, original_first_layer.out_channels,
                                     kernel_size=original_first_layer.kernel_size,
                                     stride=original_first_layer.stride,
                                     padding=original_first_layer.padding,
                                     bias=False)
        
        # 元の層の重みの平均を取って新しい畳み込み層に設定
        self.model.conv1.weight.data = original_first_layer.weight.data.mean(dim=1, keepdim=True)
        
        # 最終層の出力ユニット数を4に変更
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)
    
    def forward(self, x):
        # モデルを通じて入力xを伝播
        return self.model(x)