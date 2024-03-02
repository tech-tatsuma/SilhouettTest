import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision.transforms import functional as TF
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from models.resnet50 import CustomResNet50
from torchvision import transforms
from PIL import Image
import argparse

from datasets.dataset import SilhouetteImageDataset

# Grad-CAMを生成するためのクラス定義
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model # 使用するモデル
        self.target_layer = target_layer # 注目する層
        self.gradients = None # 勾配を保持するための変数
        self.forward_output = None # forward時の出力を保持するための変数

        # 対象層にフックを設定
        target_layer.register_forward_hook(self.save_forward_output) # forward時の出力を保存
        target_layer.register_full_backward_hook(self.save_gradients) # 勾配を保存

    # forward時の出力を保存するためのメソッド
    def save_forward_output(self, module, input, output):
        self.forward_output = output

    # 勾配を保存するためのメソッド
    def save_gradients(self, module, input_grad, output_grad):
        self.gradients = output_grad[0]

    # CAMを生成するメソッド
    def generate_cam(self, input_image, target_class=None):
        # モデルを評価モードに設定
        self.model.eval()
        output = self.model(input_image) # モデルに入力して予測を行う
        if target_class is None:
            target_class = output.argmax().item() # 対象クラスが指定されていない場合は最も革新度の高いクラスを選択
        output[:, target_class].backward() # 対象クラスに対する勾配を計算

        # 勾配と特徴マップを取得し、重みを計算
        gradients = self.gradients.detach() # 勾配を取得
        forward_output = self.forward_output.detach() # 特徴マップを取得
        weights = torch.mean(gradients, dim=[2, 3], keepdim=True) # 勾配の平均を計算して重みを求める

        # クラスアクティベーションマッピングを計算
        cam = torch.mul(forward_output, weights).sum(dim=1, keepdim=True)
        cam = F.relu(cam) # 正の影響のみを取り出す
        cam = F.interpolate(cam, input_image.shape[2:], mode='bilinear', align_corners=False) # 入力画像のサイズに合わせて拡大
        cam = cam - cam.min() # 正規化のダメ最小値を引く
        cam = cam / cam.max() # 正規化のため最大値で割る

        return cam

# CAM画像と元画像を重ねて可視化する関数
def visualize_cam(cam_image, input_image, save_path=None):
    cam_image = cam_image.cpu().squeeze().numpy() # CAMをnumpy配列に変換
    input_image = TF.to_pil_image(input_image.cpu().squeeze()) # 入力画像をPIL形式に変換
    plt.imshow(input_image, alpha=0.5) 
    plt.imshow(cam_image, cmap='jet', alpha=0.5)
    plt.axis('off')
    if save_path is not None:
        plt.savefig(save_path)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--image", type=str, required=True)
    argparser.add_argument("--model", type=str, required=True)
    argparser.add_argument("--output_path", type=str, required=True)
    args = argparser.parse_args()

    temp_transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])
    all_dataset = SilhouetteImageDataset(directory='archive', transform=temp_transform)

    # データローダの取得
    temp_loader = DataLoader(all_dataset, batch_size=len(all_dataset), shuffle=False)

    # データセットから平均と標準偏差を計算
    data = next(iter(temp_loader))[0]
    mean = data.mean([0, 2, 3])
    std = data.std([0, 2, 3])

    # モデルと対象層を指定
    model = CustomResNet50(num_classes=4)
    model_path = args.model
    model.load_state_dict(torch.load(model_path))
    target_layer = model.model.layer4

    # Grad-CAMを初期化
    grad_cam = GradCAM(model, target_layer)

    # 画像の読み込みと前処理
    def preprocess_image(image_path, size=(128, 128)):
        image = Image.open(image_path).convert('RGB')
        # 画像が3チャンネルのRGB画像であればグレースケールに変換
        if image.mode == 'RGB':
            image = image.convert('L')  # RGBをグレースケールに変換
        transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[mean], std=[std]),  # 1チャンネル用の平均値と標準偏差に調整
        ])
        image = transform(image).unsqueeze(0)  # バッチ次元を追加
        return image

    # 画像パス（ここに適切なパスを設定してください）
    image_path = args.image

    # 画像を前処理
    input_image = preprocess_image(image_path)

    # Grad-CAMの実行
    cam = grad_cam.generate_cam(input_image)

    # CAMの可視化
    visualize_cam(cam, input_image, save_path=args.output_path)