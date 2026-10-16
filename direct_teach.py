"""
SO-101 フォロワーアーム用 ダイレクトティーチング（手で動かして記録 → 再生）

使い方:
  記録: python direct_teach.py record --port /dev/ttyACM0 --id my_follower --out recordings/flick.json
  再生: python direct_teach.py play   --port /dev/ttyACM0 --id my_follower --file recordings/flick.json --speed 1.5

--id はキャリブレーション時に使った id と同じものにしてください。
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

try:
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
except ImportError as e:
    raise SystemExit(
        "LeRobot の SO101Follower を読み込めませんでした。"
        "LeRobot のバージョンによってモジュールの場所が違う可能性があります。\n" + str(e)
    )

MOTORS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def connect(port: str, robot_id: str) -> SO101Follower:
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


# ---------------------------------------------------------------- 記録
def record(args) -> None:
    robot = connect(args.port, args.id)
    robot.bus.disable_torque()
    print("トルクOFFにしました。アームを手で動かせます。")
    input("開始位置にアームを置いて Enter → 3秒後に記録開始（Ctrl+C で終了）")
    for i in (3, 2, 1):
        print(f"  {i}...")
        time.sleep(1)
    print("● 記録中")

    frames = []
    period = 1.0 / args.fps
    t0 = time.perf_counter()
    try:
        while True:
            t = time.perf_counter() - t0
            if args.duration and t > args.duration:
                break
            frames.append({"t": t, "pos": read_pos(robot)})
            sleep = period - (time.perf_counter() - t0 - t)
            if sleep > 0:
                time.sleep(sleep)
    except KeyboardInterrupt:
        pass

    robot.disconnect()

    if not frames:
        print("■ フレームが記録されなかったので保存しませんでした")
        return
    out = Path(args.out)
    out.write_text(json.dumps({"fps": args.fps, "motors": MOTORS, "frames": frames}, indent=1))
    print(f"■ {len(frames)} フレーム（{frames[-1]['t']:.1f}秒）を {out} に保存しました")


# ---------------------------------------------------------------- 再生
def play(args) -> None:
    data = json.loads(Path(args.file).read_text())
    t = np.array([f["t"] for f in data["frames"]])
    P = np.array([f["pos"] for f in data["frames"]])  # (N, 6)

    robot = connect(args.port, args.id)
    try:
        try:
            print("アームを手で支えてください（トルクを入れます）")
            input("準備ができたら Enter")
            current = np.array(hold_current_pose(robot))

            # 1) 現在位置から記録の開始位置までゆっくり移動
            print("開始位置へ移動中...")
            steps = int(args.approach_sec * args.fps)
            for k in range(1, steps + 1):
                send(robot, current + (P[0] - current) * k / steps)
                time.sleep(1.0 / args.fps)

            # 2) 時間を speed 倍に伸縮して再生（関節ごとに線形補間）
            for n in range(args.loop):
                print(f"▶ 再生 {n + 1}/{args.loop}（速度 x{args.speed}）")
                t0 = time.perf_counter()
                while True:
                    tau = (time.perf_counter() - t0) * args.speed
                    if tau > t[-1]:
                        break
                    target = [np.interp(tau, t, P[:, j]) for j in range(len(MOTORS))]
                    send(robot, target)
                    time.sleep(1.0 / args.fps)
                if n < args.loop - 1:  # ループの継ぎ目で飛ばないよう開始位置に戻す
                    last = P[-1]
                    for k in range(1, steps + 1):
                        send(robot, last + (P[0] - last) * k / steps)
                        time.sleep(1.0 / args.fps)
        except KeyboardInterrupt:
            print("\n■ 中断しました")
    finally:
        # 中断・エラー時もトルクが入ったままにならないよう、必ずここを通す
        input("再生終了。アームを手で支えてから Enter（トルクがOFFになります）")
        robot.disconnect()


def main() -> None:
    p = argparse.ArgumentParser(description="SO-101 ダイレクトティーチング")
    sub = p.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--port", required=True, help="例: COM3（lerobot-find-port で確認）")
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

    args = p.parse_args()
    record(args) if args.cmd == "record" else play(args)


if __name__ == "__main__":
    main()
