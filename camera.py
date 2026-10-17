"""
USB カメラの録画（アームの制御を止めないよう、別スレッドで撮り続ける）

動画と一緒に <名前>_frames.csv に各フレームの時刻（time.time()）を残す。
再生のイベントログも同じ時計なので、あとで「どのパターンのときの映像か」を切り出せる。

カメラの確認（5秒撮って、解像度と実際の fps を表示）:
  python camera.py --camera 0 --seconds 5

このカメラは自動露出（auto exposure）だと fps が 20 くらいに落ちる。
--exposure と --gain で手動露出に固定すると 30fps を維持できる（要 v4l2-ctl）。
部屋の明るさで値は変わるので、まず明るさを見て調整すること。目安：
  python camera.py --camera 0 --seconds 5 --exposure 220 --gain 36
"""
import argparse
import csv
import subprocess
import sys
import threading
import time
from pathlib import Path

import cv2


def parse_source(s: str):
    """'0' → カメラ番号 0、それ以外は動画ファイルのパス（テスト用）"""
    return int(s) if s.isdigit() else s


def set_manual_exposure(camera_index: int, exposure: int | None, gain: int | None) -> None:
    """auto_exposure を手動に固定して露光とゲインを設定する（v4l2-ctl が要る）"""
    device = f"/dev/video{camera_index}"
    cmd = ["v4l2-ctl", "-d", device, "-c", "auto_exposure=1"]
    if exposure is not None:
        cmd += ["-c", f"exposure_time_absolute={exposure}"]
    if gain is not None:
        cmd += ["-c", f"gain={gain}"]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except FileNotFoundError:
        print("  ⚠️ v4l2-ctl が見つかりません（sudo apt install v4l-utils）。自動露出のまま使います")
    except subprocess.CalledProcessError as e:
        print(f"  ⚠️ 露出の設定に失敗: {e.stderr.strip()}")


class CameraRecorder:
    def __init__(
        self,
        source,
        out_path: Path,
        width: int = 1920,
        height: int = 1080,
        fps: int = 30,
        exposure: int | None = None,
        gain: int | None = None,
    ):
        self.out_path = Path(out_path)
        self.frames_path = self.out_path.with_name(self.out_path.stem + "_frames.csv")
        self.n_frames = 0
        self._stop = threading.Event()
        self._thread = None

        if isinstance(source, int) and sys.platform.startswith("linux"):
            self.cap = cv2.VideoCapture(source, cv2.CAP_V4L2)
        else:
            self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise SystemExit(f"カメラ {source} を開けませんでした（v4l2-ctl --list-devices で番号を確認）")
        if isinstance(source, int):
            # MJPG にしないと、多くの USB カメラは 1080p で数 fps しか出ない
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.cap.set(cv2.CAP_PROP_FPS, fps)
            if (exposure is not None or gain is not None) and sys.platform.startswith("linux"):
                set_manual_exposure(source, exposure, gain)

        ok, frame = self.cap.read()
        if not ok:
            raise SystemExit(f"カメラ {source} から映像を読めませんでした")
        self.height, self.width = frame.shape[:2]
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or fps
        if (self.width, self.height) != (width, height):
            print(f"  ⚠️ カメラの解像度は {self.width}x{self.height} です（{width}x{height} を指定）")

        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        codec = "MJPG" if self.out_path.suffix == ".avi" else "mp4v"
        self.writer = cv2.VideoWriter(
            str(self.out_path), cv2.VideoWriter_fourcc(*codec), self.fps, (self.width, self.height)
        )
        if not self.writer.isOpened():
            self.cap.release()
            raise SystemExit(f"{self.out_path} に書き出せません（OpenCV が {codec} に対応していない可能性）")

    def start(self) -> None:
        self._t_first = self._t_last = None
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print(f"● 録画開始: {self.out_path}（{self.width}x{self.height}）")

    def _loop(self) -> None:
        with self.frames_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["frame", "time"])
            while not self._stop.is_set():
                ok, frame = self.cap.read()
                if not ok:
                    break
                t = time.time()
                self.writer.write(frame)
                w.writerow([self.n_frames, f"{t:.4f}"])
                self._t_first = self._t_first or t
                self._t_last = t
                self.n_frames += 1

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join()
        self.writer.release()
        self.cap.release()
        if self.n_frames > 1:
            sec = self._t_last - self._t_first
            print(f"■ 録画終了: {self.n_frames} フレーム / {sec:.1f}秒（実際 {(self.n_frames - 1) / sec:.1f} fps）")
        else:
            print("■ 録画終了: フレームが撮れていません")


def main() -> None:
    p = argparse.ArgumentParser(description="カメラの確認（短く撮って fps を表示）")
    p.add_argument("--camera", default="0", help="カメラ番号（例: 0）か動画ファイル")
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--height", type=int, default=1080)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--seconds", type=float, default=5)
    p.add_argument("--out", default="sessions/camera_test.mp4")
    p.add_argument("--exposure", type=int, default=None, help="手動露光時間（例: 220）。指定すると自動露出をオフにする")
    p.add_argument("--gain", type=int, default=None, help="手動ゲイン（0〜63）。暗いときに明るさを足す")
    args = p.parse_args()

    rec = CameraRecorder(
        parse_source(args.camera), Path(args.out), args.width, args.height, args.fps, args.exposure, args.gain
    )
    rec.start()
    time.sleep(args.seconds)
    rec.stop()


if __name__ == "__main__":
    main()
