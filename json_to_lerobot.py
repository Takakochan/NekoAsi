"""direct_teach.py の記録 JSON を LeRobotDataset に変換して、Hugging Face にアップロードする

使い方:
  python json_to_lerobot.py --repo-id NekoAsi/nekoasi-motions --files recordings/*.json
  python json_to_lerobot.py --repo-id NekoAsi/nekoasi-motions --files recordings/*.json --push

sessions/<日時>_events.csv に、そのファイルを Vino の前でカメラ付きで再生した記録
（play_start〜play_end）があれば、その区間の動画フレームも observation.images.cam として
同じエピソードに入れる。無ければそのファイルは関節角度だけでは中途半端になるのでスキップする
（--files に残りのファイルだけ渡し直せば関節角度のみのデータセットも作れる）。
"""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

MOTORS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
IMAGE_KEY = "observation.images.cam"


def resample(path: Path, fps: int) -> tuple[np.ndarray, np.ndarray]:
    """記録を fps のきっちりした間隔に線形補間する。(grid, 関節角度) を返す。
    grid は記録自身の時間軸（再生速度とは無関係。壁時計時刻にするには speed で割る）"""
    data = json.loads(path.read_text())
    t = np.array([f["t"] for f in data["frames"]])
    P = np.array([f["pos"] for f in data["frames"]])
    grid = np.arange(0.0, t[-1], 1.0 / fps)
    S = np.stack([np.interp(grid, t, P[:, j]) for j in range(P.shape[1])], axis=1)
    return grid, S.astype(np.float32)


def find_play_event(file_name: str, sessions_dir: Path):
    """file_name をカメラ付きで再生したイベントのうち、一番新しいものを探す。
    (video_path, frames_csv_path, play_start, play_end, speed) を返す。無ければ None"""
    best = None
    for events_path in sorted(sessions_dir.glob("*_events.csv")):
        session_ts = events_path.name[: -len("_events.csv")]
        video_path = sessions_dir / f"{session_ts}.mp4"
        frames_path = sessions_dir / f"{session_ts}_frames.csv"
        if not (video_path.exists() and frames_path.exists()):
            continue
        with events_path.open() as f:
            for row in csv.DictReader(f):
                if row["file"] != file_name or not row.get("play_start") or not row.get("play_end"):
                    continue
                if row["interrupted"] == "True":
                    continue
                if int(row["loop"]) != 1:
                    print(f"  ⚠️ {file_name}: loop={row['loop']} のイベントは動画同期が未対応なのでスキップ")
                    continue
                best = (video_path, frames_path, float(row["play_start"]), float(row["play_end"]), float(row["speed"]))
    return best


def extract_frames(video_path: Path, frames_csv_path: Path, wall_times: np.ndarray) -> list:
    """wall_times（壁時計時刻）に一番近い動画フレームを、先頭から順番に読んで集める。
    mp4 はシーク(POS_FRAMES)よりも順番に読むほうが安全（過去の調査でフレームが
    乱れることがあると分かったため）"""
    rows = list(csv.DictReader(frames_csv_path.open()))
    frame_times = np.array([float(r["time"]) for r in rows])
    idx = np.searchsorted(frame_times, wall_times)
    idx = np.clip(idx, 0, len(frame_times) - 1)
    prev = np.clip(idx - 1, 0, None)
    closer_to_prev = (idx > 0) & (np.abs(frame_times[prev] - wall_times) < np.abs(frame_times[idx] - wall_times))
    idx = np.where(closer_to_prev, idx - 1, idx)

    cap = cv2.VideoCapture(str(video_path))
    frames_by_idx = {}
    cur = -1
    frame = None
    for target in sorted(set(idx.tolist())):
        while cur < target:
            ok, frame = cap.read()
            cur += 1
            if not ok:
                raise RuntimeError(f"{video_path}: フレーム {target} まで読めなかった")
        frames_by_idx[target] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cap.release()
    return [frames_by_idx[i] for i in idx]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-id", required=True, help="例: NekoAsi/nekoasi-motions")
    p.add_argument("--files", nargs="+", required=True)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--sessions-dir", default="sessions")
    p.add_argument("--push", action="store_true", help="Hugging Face にアップロードする")
    p.add_argument("--public", action="store_true", help="公開にする（省略すると非公開）")
    args = p.parse_args()

    sessions_dir = Path(args.sessions_dir)
    names = [f"{m}.pos" for m in MOTORS]
    features = {
        "observation.state": {"dtype": "float32", "shape": (len(MOTORS),), "names": names},
        "action": {"dtype": "float32", "shape": (len(MOTORS),), "names": names},
    }

    plan = []  # (path, grid, S, event)
    image_shape = None
    for f in args.files:
        path = Path(f)
        grid, S = resample(path, args.fps)
        event = find_play_event(path.name, sessions_dir)
        if event is None:
            print(f"  ⚠️ {path.name}: カメラ付きで再生した記録が見つからないのでスキップ")
            continue
        plan.append((path, grid, S, event))
        if image_shape is None:
            cap = cv2.VideoCapture(str(event[0]))
            ok, frame = cap.read()
            cap.release()
            if ok:
                image_shape = frame.shape  # (H, W, 3)

    if not plan:
        raise SystemExit("カメラ付きで再生した記録が1つも無いので、変換できるファイルがありません")

    features[IMAGE_KEY] = {"dtype": "video", "shape": image_shape, "names": ["height", "width", "channels"]}

    ds = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=args.fps,
        features=features,
        robot_type="so101_follower",
        use_videos=True,
    )

    for path, grid, S, event in plan:
        A = np.vstack([S[1:], S[-1:]])
        video_path, frames_path, play_start, play_end, speed = event
        wall_times = play_start + grid / speed
        if wall_times[-1] > play_end + 0.5:
            print(f"  ⚠️ {path.name}: 計算した再生終了時刻が記録の play_end と合わないのでスキップ")
            continue
        images = extract_frames(video_path, frames_path, wall_times)

        for state, action, image in zip(S, A, images):
            ds.add_frame({"observation.state": state, "action": action, IMAGE_KEY: image, "task": path.stem})
        ds.save_episode()
        print(f"{path.name}: {len(S)} フレームを1エピソードとして保存（映像つき）")

    ds.finalize()  # これを呼ばないとデータセットが壊れる
    print(f"ローカルに保存: {ds.root}")

    if args.push:
        ds.push_to_hub(private=not args.public, tags=["NekoAsi", "so101", "LeRobot"])
        print(f"アップロード完了: https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
