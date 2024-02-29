import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn.functional as F
from torch.utils.data import random_split
from torch import nn

import argparse
import random
import numpy as np
from tqdm import tqdm
import sys
from matplotlib import pyplot as plt
from setproctitle import setproctitle
import os
import optuna
import time

from models.mlpmixer import MLPMixer
from models.resnet50 import CustomResNet50
from models.vit import VisionTransformer
from datasets.dataset import SilhouetteImageDataset

from models.crossvit import VisionTransformer as crossvit

# シードの設定を行う関数
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 訓練のための関数
def train(opt, trial=None):
    print(opt)
    # シードの設定
    seed_everything(opt.seed)

    # デバイスの設定
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # ハイパーパラメータの設定
    epochs = opt.epochs
    patience = opt.patience
    learning_rate = opt.learning_rate
    batch_size = opt.batch_size

    # データセットの読み込み
    temp_transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])
    all_dataset = SilhouetteImageDataset(directory='archive', transform=temp_transform)

    # データローダの取得
    temp_loader = DataLoader(all_dataset, batch_size=len(all_dataset), shuffle=False)
    # データセットから平均と標準偏差を計算
    data = next(iter(temp_loader))[0]
    mean = data.mean([0, 2, 3])
    std = data.std([0, 2, 3])

    # 正規化を含む変換処理の定義
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    all_dataset = SilhouetteImageDataset(directory='archive', transform=transform)

    # 訓練用、評価用のデータサイズを設定
    train_size = int(0.8 * len(all_dataset))
    val_size = len(all_dataset) - train_size

    # データの分割
    train_dataset, val_dataset = random_split(all_dataset, [train_size, val_size])

    # データローダの取得
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # モデルの定義
    num_classes = 4

    if opt.modelname == "MLP-Mixer":
        model = MLPMixer()
    elif opt.modelname == "ResNet50":
        model = CustomResNet50(num_classes=num_classes)
    elif opt.modelname == "VisionTransformer":
        model = VisionTransformer(
            image_size=128,
            patch_size=16,
            in_channels=1,
            embedding_dim=768,
            num_layers=12,
            num_heads=12,
            mlp_ratio=4.0,
            num_classes=4,
            hidden_dims=[64, 128, 256])
    elif opt.modelname == "crossvit":
        model = crossvit()

    # モデルの並列化
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    # モデルの転送
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # 早期終了のためのパラメータの初期化
    val_loss_min = None
    val_loss_min_epoch = 0

    # 学習曲線のための配列の初期化
    train_losses = []
    val_losses = []

    # モデルの訓練
    for epoch in tqdm(range(epochs)):

        # 各種パラメータの初期化
        train_loss = 0.0
        val_loss = 0.0

        # モデルをtrainモードに設定
        model.train()

        # trainデータのロード
        for inputs, labels in train_loader:
            
            inputs, labels = inputs.to(device), labels.to(device)

            # 勾配の初期化
            optimizer.zero_grad()

            # モデルの適用
            outputs = model(inputs)

            # 損失関数の計算
            loss = F.cross_entropy(outputs, labels)
            loss.backward()

            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # モデルを評価モードの設定
        model.eval()

        val_correct = 0  # 正解数
        val_total = 0    # 総数
        
        # 検証データでの評価
        with torch.no_grad():
            for inputs, labels in val_loader:
                
                # データをGPUに転送
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)

                loss = F.cross_entropy(outputs, labels)
                val_loss += loss.item()

                # 精度の計算
                _, predicted = torch.max(outputs, 1)  # 最も高いスコアを持つクラスを予測値とする
                val_total += labels.size(0)          # バッチサイズ（ラベルの数）を総数に加算
                val_correct += (predicted == labels).sum().item()  # 正解数を更新

        val_loss /= len(val_loader)  # 平均損失を計算
        val_accuracy = val_correct / val_total  # 精度を計算
        val_losses.append(val_loss)  # 損失を記録

        print(f'Epoch {epoch+1}, Training loss: {train_loss:.4f}, Validation loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')
        sys.stdout.flush()

        # メモリーを最適化する
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # バリデーションロスが下がった時は結果を保存する
        if val_loss_min is None or val_loss < val_loss_min:
            model_save_directory = './crossvit'
            model_save_name = f'./crossvit/lr{learning_rate}_ep{epochs}_pa{patience}intweak.pt'
            if not os.path.exists(model_save_directory):
                os.makedirs(model_save_directory)
            torch.save(model.state_dict(), model_save_name)
            val_loss_min = val_loss
            val_loss_min_epoch = epoch
            
        # もしバリデーションロスが一定期間下がらなかったらその時点で学習を終わらせる
        elif (epoch - val_loss_min_epoch) >= patience:
            print('Early stopping due to validation loss not improving for {} epochs'.format(patience))
            break

    # 学習プロセスをグラフ化し、保存する
    plt.figure(figsize=(15, 5))
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

    plt.plot(train_losses, label='Training loss')
    plt.plot(val_losses, label='Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.title("Training and Validation Loss")
    graph_save_directory = './crossvit'
    graph_save_name = f'{graph_save_directory}/lr{learning_rate}_ep{epochs}_pa{patience}intweak.png'

    if not os.path.exists(graph_save_directory):
        os.makedirs(graph_save_directory)

    plt.savefig(graph_save_name)


    return val_loss_min

def objective(trial):
    args = argparse.Namespace(
        epochs=100,
        learning_rate=trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True),
        weight_decay=trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True),
        patience=20,
        islearnrate_search=True,
        seed=42,
        batch_size=20,
        modelname='crossvit'
    )
    return train(args, trial)

if __name__ == '__main__':
    # プロセス名の設定
    setproctitle("crossvit")

    # オプションを標準出力する
    print('-----biginning training-----')
    start_time = time.time()
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=5)

    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    end_time = time.time()
    # 処理時間（秒）を計算
    elapsed_time = end_time - start_time
    print(f'-----completing training in {elapsed_time:.2f} seconds-----')