"""
SO-101 フォロワーアーム用 ダイレクトティーチング（手で動かして記録 → 再生）

使い方:
  記録: python direct_teach.py record --port /dev/ttyACM0 --id my_follower --out recordings/flick.json
  再生: python direct_teach.py play   --port /dev/ttyACM0 --id my_follower --file recordings/flick.json --speed 1.5
  再生して反応をメモ: python direct_teach.py play ... --log
  範囲チェック: python direct_teach.py check --id my_follower recordings/*.json

--id はキャリブレーション時に使った id と同じものにしてください。
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

MOTORS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
CALIB_DIRS = [
    Path.home() / ".cache/huggingface/lerobot/calibration/robots/so_follower",
    Path(__file__).parent / "calibration",  # リポジトリに保存したコピー
]


def connect(port: str, robot_id: str):
    # check だけならアームなしで使えるよう、LeRobot はここで読み込む
    try:
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
    except ImportError as e:
        raise SystemExit(
            "LeRobot の SO101Follower を読み込めませんでした。"
            "LeRobot のバージョンによってモジュールの場所が違う可能性があります。\n" + str(e)
        )
    robot = SO101Follower(SO101FollowerConfig(port=port, id=robot_id))
    robot.connect()
    return robot


def read_pos(robot) -> list[float]:
    pos = robot.bus.sync_read("Present_Position")
    return [float(pos[m]) for m in MOTORS]


def hold_current_pose(robot) -> list[float]:
    """トルクONで急に動かないよう、目標位置を現在位置に合わせてからトルクを入れる"""
    robot.bus.disable_torque()
    current = robot.bus.sync_read("Present_Position")
    robot.bus.sync_write("Goal_Position", current)
    robot.bus.enable_torque()
    return [float(current[m]) for m in MOTORS]


def send(robot, values) -> None:
    robot.send_action({f"{m}.pos": float(v) for m, v in zip(MOTORS, values)})


# ---------------------------------------------------------------- 範囲チェック
def joint_limits(robot_id: str) -> dict[str, tuple[float, float]] | None:
    """キャリブレーションの範囲を「度」に直す（LeRobot と同じく範囲の中心が 0°）"""
    path = next((d / f"{robot_id}.json" for d in CALIB_DIRS if (d / f"{robot_id}.json").exists()), None)
    if path is None:
        return None
    calib = json.loads(path.read_text())
    limits = {}
    for m in MOTORS:
        if m in ("wrist_roll", "gripper"):  # wrist_roll は1回転、gripper は 0〜100 表示なので対象外
            continue
        lo, hi = calib[m]["range_min"], calib[m]["range_max"]
        mid = (lo + hi) / 2
        limits[m] = ((lo - mid) * 360 / 4095, (hi - mid) * 360 / 4095)
    return limits


def check_range(frames: list[dict], robot_id: str) -> bool:
    """範囲の外に出た関節を表示する。全部収まっていれば True"""
    limits = joint_limits(robot_id)
    if limits is None:
        print(f"  （キャリブレーションファイル {robot_id}.json が見つからないので範囲チェックを省略）")
        return True
    P = np.array([f["pos"] for f in frames])
    ok = True
    for j, m in enumerate(MOTORS):
        if m not in limits:
            continue
        lo, hi = limits[m]
        if P[:, j].min() < lo or P[:, j].max() > hi:
            ok = False
            print(
                f"  ⚠️ {m} が範囲の外: 記録 {P[:, j].min():.1f}〜{P[:, j].max():.1f}° / "
                f"範囲 {lo:.1f}〜{hi:.1f}°（再生では端で止まります）"
            )
    if ok:
        print("  ✅ 全関節が可動域に収まっています")
    return ok


# ---------------------------------------------------------------- 記録
def record_frames(robot, fps: int, duration: float) -> list[dict]:
    """トルクOFFの状態で、Ctrl+C か duration 秒まで関節角度を記録する"""
    for i in (3, 2, 1):
        print(f"  {i}...")
        time.sleep(1)
    print("● 記録中（Ctrl+C で終了）")

    frames = []
    period = 1.0 / fps
    t0 = time.perf_counter()
    try:
        while True:
            t = time.perf_counter() - t0
            if duration and t > duration:
                break
            frames.append({"t": t, "pos": read_pos(robot)})
            sleep = period - (time.perf_counter() - t0 - t)
            if sleep > 0:
                time.sleep(sleep)
    except KeyboardInterrupt:
        pass
    return frames


def save_recording(frames: list[dict], fps: int, out: Path, robot_id: str) -> bool:
    if not frames:
        print("■ フレームが記録されなかったので保存しませんでした")
        return False
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"fps": fps, "motors": MOTORS, "frames": frames}, indent=1))
    print(f"■ {len(frames)} フレーム（{frames[-1]['t']:.1f}秒）を {out} に保存しました")
    check_range(frames, robot_id)
    return True


def record(args) -> None:
    robot = connect(args.port, args.id)
    robot.bus.disable_torque()
    print("トルクOFFにしました。アームを手で動かせます。")
    input("開始位置にアームを置いて Enter → 3秒後に記録開始")
    frames = record_frames(robot, args.fps, args.duration)
    robot.disconnect()
    save_recording(frames, args.fps, Path(args.out), args.id)


# ---------------------------------------------------------------- 再生
def load_motion(file: str) -> tuple[np.ndarray, np.ndarray]:
    data = json.loads(Path(file).read_text())
    t = np.array([f["t"] for f in data["frames"]])
    P = np.array([f["pos"] for f in data["frames"]])  # (N, 6)
    return t, P


def move_to(robot, goal: np.ndarray, sec: float, fps: int) -> None:
    """今の位置から goal まで sec 秒かけてゆっくり移動する"""
    start = np.array(read_pos(robot))
    steps = max(1, int(sec * fps))
    for k in range(1, steps + 1):
        send(robot, start + (goal - start) * k / steps)
        time.sleep(1.0 / fps)


def play_frames(robot, t, P, speed: float, loop: int, fps: int, approach_sec: float) -> None:
    """トルクONの状態で、開始位置へ移動してから記録を再生する"""
    # 1) 現在位置から記録の開始位置までゆっくり移動
    print("開始位置へ移動中...")
    move_to(robot, P[0], approach_sec, fps)

    # 2) 時間を speed 倍に伸縮して再生（関節ごとに線形補間）
    for n in range(loop):
        print(f"▶ 再生 {n + 1}/{loop}（速度 x{speed}）")
        t0 = time.perf_counter()
        while True:
            tau = (time.perf_counter() - t0) * speed
            if tau > t[-1]:
                break
            target = [np.interp(tau, t, P[:, j]) for j in range(len(MOTORS))]
            send(robot, target)
            time.sleep(1.0 / fps)
        if n < loop - 1:  # ループの継ぎ目で飛ばないよう開始位置に戻す
            move_to(robot, P[0], approach_sec, fps)


def release(robot) -> None:
    """アームを支えてもらってからトルクを切る。Ctrl+C でもトルクを入れたまま終わらない"""
    while True:
        try:
            input("再生終了。アームを手で支えてから Enter（トルクがOFFになります）")
            break
        except KeyboardInterrupt:
            print("\nCtrl+C ではなく、アームを支えてから Enter を押してください")
    robot.disconnect()


def play(args) -> None:
    t, P = load_motion(args.file)

    robot = connect(args.port, args.id)
    try:
        try:
            print("アームを手で支えてください（トルクを入れます）")
            input("準備ができたら Enter")
            hold_current_pose(robot)
            play_frames(robot, t, P, args.speed, args.loop, args.fps, args.approach_sec)
        except KeyboardInterrupt:
            print("\n■ 中断しました")
    finally:
        # 中断・エラー時もトルクが入ったままにならないよう、必ずここを通す
        release(robot)

    if args.log:
        from reactions import ask_and_log

        ask_and_log(args.file, args.speed)


def check(args) -> None:
    for f in args.files:
        print(f)
        check_range(json.loads(Path(f).read_text())["frames"], args.id)


def main() -> None:
    p = argparse.ArgumentParser(description="SO-101 ダイレクトティーチング")
    sub = p.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--port", required=True, help="例: /dev/ttyACM0（lerobot-find-port で確認）")
    common.add_argument("--id", required=True, help="キャリブレーション時の id")
    common.add_argument("--fps", type=int, default=30)

    r = sub.add_parser("record", parents=[common])
    r.add_argument("--out", default="demo.json")
    r.add_argument("--duration", type=float, default=0, help="秒。0なら Ctrl+C まで")

    pl = sub.add_parser("play", parents=[common])
    pl.add_argument("--file", required=True)
    pl.add_argument("--speed", type=float, default=1.0, help="1.5 なら1.5倍速")
    pl.add_argument("--loop", type=int, default=1)
    pl.add_argument("--approach-sec", type=float, default=3.0, help="開始位置までの移動時間")
    pl.add_argument("--log", action="store_true", help="再生後に Vino の反応を reactions.csv に記録する")

    c = sub.add_parser("check", help="記録が可動域に収まっているか調べる（アーム不要）")
    c.add_argument("--id", default="my_follower", help="キャリブレーション時の id")
    c.add_argument("files", nargs="+")

    args = p.parse_args()
    {"record": record, "play": play, "check": check}[args.cmd](args)


if __name__ == "__main__":
    main()
