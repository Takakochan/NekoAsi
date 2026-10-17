"""NekoAsi の最初のMLデモ：記録データ → モデル → 学習、を一通り体験する

本番用の学習（LeRobotのACTなど）ではなく、仕組みが見える最小構成。
今は flutter_01 の1エピソード（413フレーム）しか無いので、これは「データの形に慣れる」
ためのもので、汎化できるモデルにはならない（同じ数十フレームを繰り返し見て覚えるだけ ＝
過学習 overfitting）。データが増えたら本番用の学習に進む。

やること：
  1. Hugging Face の NekoAsi/nekoasi-motions を読み込む
  2. 画像＋今の関節角度（state） → 次の関節角度（action）を予測する小さい CNN を学習する
  3. 損失（loss）が下がる様子と、予測値 vs 実際の値を表示する

使い方:
  python ml_demo.py
"""
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from lerobot.datasets.lerobot_dataset import LeRobotDataset

REPO_ID = "NekoAsi/nekoasi-motions"
IMAGE_KEY = "observation.images.cam"
IMAGE_SIZE = (90, 160)  # 元は 720x1280。小さい CNN で十分学習できるサイズに縮める


class TinyVisionPolicy(nn.Module):
    """画像とstateから次のactionを予測する、できるだけ小さいモデル"""

    def __init__(self, state_dim: int = 6, action_dim: int = 6):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=2), nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),  # 画像のサイズによらず (32, 4, 4) にそろえる
        )
        self.head = nn.Sequential(
            nn.Linear(32 * 4 * 4 + state_dim, 64), nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, image: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        feat = self.cnn(image).flatten(1)
        return self.head(torch.cat([feat, state], dim=1))


def main() -> None:
    ds = LeRobotDataset(REPO_ID)
    print(f"データセット: {len(ds)} フレーム（{ds.num_episodes} エピソード）")

    loader = DataLoader(ds, batch_size=32, shuffle=True)

    model = TinyVisionPolicy()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    print("\n学習中（このデータだけを繰り返し見て覚える＝過学習の確認）")
    for epoch in range(20):
        total_loss = 0.0
        for batch in loader:
            image = F.interpolate(batch[IMAGE_KEY], size=IMAGE_SIZE)
            pred = model(image, batch["observation.state"])
            loss = F.mse_loss(pred, batch["action"])

            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        print(f"  epoch {epoch + 1:2d}: loss = {total_loss / len(loader):.3f}")

    print("\n予測 vs 実際（関節角度、度）の例：")
    model.eval()
    sample = next(iter(DataLoader(ds, batch_size=5, shuffle=True)))
    with torch.no_grad():
        image = F.interpolate(sample[IMAGE_KEY], size=IMAGE_SIZE)
        pred = model(image, sample["observation.state"])
    for i in range(len(pred)):
        print(f"  予測:   {pred[i].numpy().round(1)}")
        print(f"  実際:   {sample['action'][i].numpy().round(1)}")
        print()


if __name__ == "__main__":
    main()
