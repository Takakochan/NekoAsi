"""
動きのパターンを、まとめて流れ作業で記録する（パターンごとに複数テイク）

使い方:
  python record_session.py --port /dev/ttyACM0 --id my_follower
  python record_session.py --port /dev/ttyACM0 --id my_follower --patterns slow_crawl dart_stop flutter

recordings/<パターン>_01.json, _02.json, ... に保存する。
すでにあるファイルは飛ばすので、途中でやめても同じコマンドで続きから再開できる。
"""
import argparse
from pathlib import Path

from direct_teach import connect, record_frames, save_recording

PATTERNS = {
    "slow_crawl": "床の上をゆっくり這わせる（ネズミ・虫）",
    "dart_stop": "床の上で素早く動いて急に止まる、を繰り返す",
    "flick": "手首で素早く跳ねさせる（跳ねる虫）",
    "flutter": "空中で小刻みに揺らし、ときどき床に降ろす（鳥）",
    "peek": "物陰からチラッと出して、すぐ引っ込める",
    "retreat": "猫の方から離れていく（猫に向かって近づけない）",
}


def main() -> None:
    p = argparse.ArgumentParser(description="パターンをまとめて記録する")
    p.add_argument("--port", required=True, help="例: /dev/ttyACM0（lerobot-find-port で確認）")
    p.add_argument("--id", required=True, help="キャリブレーション時の id")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--patterns", nargs="+", default=list(PATTERNS), help="記録するパターン名")
    p.add_argument("--takes", type=int, default=3, help="1パターンあたりのテイク数")
    p.add_argument("--duration", type=float, default=0, help="1テイクの最大秒数。0なら Ctrl+C まで")
    p.add_argument("--out-dir", default="recordings")
    p.add_argument("--overwrite", action="store_true", help="すでにあるファイルも撮り直す")
    args = p.parse_args()

    robot = connect(args.port, args.id)
    try:
        robot.bus.disable_torque()
        print("トルクOFFにしました。アームを手で動かせます。")
        print("毎回、同じホーム姿勢から始めて、ホーム姿勢で終わってください。\n")

        for pattern in args.patterns:
            skip_pattern = False
            for take in range(1, args.takes + 1):
                out = Path(args.out_dir) / f"{pattern}_{take:02d}.json"
                if out.exists() and not args.overwrite:
                    print(f"（{out} はあるので飛ばします）")
                    continue

                while True:  # 撮り直しのためのループ
                    print(f"\n=== {pattern}  テイク {take}/{args.takes} ===")
                    print(f"    {PATTERNS.get(pattern, '')}")
                    ans = input("ホーム姿勢に置いて Enter → 3秒後に開始（s=このパターンを飛ばす / q=終了）: ")
                    ans = ans.strip().lower()
                    if ans == "q":
                        return
                    if ans == "s":
                        skip_pattern = True
                        break

                    frames = record_frames(robot, args.fps, args.duration)
                    save_recording(frames, args.fps, out, args.id)
                    if input("Enter=次へ / r=撮り直し: ").strip().lower() != "r":
                        break
                if skip_pattern:
                    break
        print("\nセッション終了")
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
