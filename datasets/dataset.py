import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class SilhouetteImageDataset(Dataset):
    def __init__(self, directory, transform=None):
        # データセットが格納されているディレクトリのパスを初期化
        self.directory = directory
        # 画像に適用する前処理を指定
        self.transform = transform
        # 対象となるクラスのリストを定義
        self.classes = ['bending', 'lying', 'sitting', 'standing']
        # クラス名をインデックスに変換する辞書を作成
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        # データセット内の全サンプルのリストを作成
        self.samples = self._make_dataset()

    def _make_dataset(self):
        # 空のリストを初期化．これにサンプルを追加していく
        samples = []
        for target in self.classes:
            # 各クラスに対応するディレクトリのパスを取得
            target_dir = os.path.join(self.directory, target)
            # ディレクトリが存在しない場合はスキップ
            if not os.path.isdir(target_dir):
                continue
            # ディレクトリ内の全てのファイル名を走査
            for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
                for fname in sorted(fnames):
                    # 各ファイルの完全なパスを生成
                    path = os.path.join(root, fname)
                    # パスと対応するクラスのインデックスをタプルとしてリストに追加
                    item = (path, self.class_to_idx[target])
                    samples.append(item)
        return samples

    # データセット内のサンプル数を返す
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # 指定されたインデックスに対応するサンプルを取得
        path, target = self.samples[idx]
        # 画像ファイルを開き、PIL.Imageオブジェクトとして読み込みます。
        image = Image.open(path).convert('RGB')
        # 画像が3チャンネルのRGB画像であればグレースケールに変換
        if image.mode == 'RGB':
            image = image.convert('L')  # RGBをグレースケールに変換
        if self.transform:
            image = self.transform(image)
        return image, target