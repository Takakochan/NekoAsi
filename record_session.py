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
    "slow_crawl": {"how": "床の上をゆっくり這わせる（ネズミ・虫）"},
    "dart_stop": {"how": "床の上で素早く動いて急に止まる、を繰り返す"},
    "flick": {"how": "手首で素早く跳ねさせる（跳ねる虫）"},
    "flutter": {"how": "空中で小刻みに揺らし、ときどき床に降ろす（鳥）"},
    "peek": {"how": "物陰からチラッと出して、すぐ引っ込める"},
    "retreat": {"how": "猫の方から離れていく（猫に向かって近づけない）"},
    "air_hover": {
        "height": "高い（Vino の手が届きにくい高さ）",
        "speed": "ゆっくり",
        "how": "その場でふわふわ小さく上下・左右に揺れる。止まらず、ずっと漂っている感じ",
        "duration": "8〜15秒",
    },
    "air_zigzag": {
        "height": "高い",
        "speed": "速め",
        "how": "左右に小刻みに方向転換を繰り返す（ジグザグ）。切り返しをはっきりさせる",
        "duration": "5〜10秒",
    },
    "air_swoop": {
        "height": "高い",
        "speed": "速い",
        "how": "大きな弧を描いて一方からもう一方へ振る。羽ばたくような小刻みさは無し、ダイナミックに",
        "duration": "5〜10秒",
    },
    "air_peekaboo": {
        "height": "中〜高い（低くしすぎない）",
        "speed": "ゆっくり＋一瞬速い",
        "how": "一度上に隠す（視界の外や死角）→一瞬だけ下げて見せる→またすぐ上げる、を2〜3回繰り返す。"
        "下げるのは一瞬だけで、Vino の届く高さに留めない",
        "duration": "10〜15秒",
    },
    "fly_quick": {
        "height": "空中（高め）",
        "speed": "速い",
        "how": "flutter より速く、短い距離を素早くダートするように左右・上下に移動。"
        "小刻みな羽ばたきではなく、素早い移動そのもの",
        "duration": "5〜8秒",
    },
    "low_dart": {
        "height": "低い（床）",
        "speed": "速い",
        "how": "床スレスレを素早く走らせてピタッと止める。dart_stop の低い版。1〜2テイクのみでOK",
        "duration": "5〜8秒",
    },
}


def print_pattern_detail(pattern: str) -> None:
    """記録前に、動きの詳細（高さ・速さ・動かし方・長さの目安）をターミナルに表示する"""
    info = PATTERNS.get(pattern)
    if not info:
        return
    line = " / ".join(
        f"{label}: {info[key]}"
        for label, key in (("高さ", "height"), ("速さ", "speed"))
        if info.get(key)
    )
    if line:
        print(f"    {line}")
    if info.get("how"):
        print(f"    動き: {info['how']}")
    if info.get("duration"):
        print(f"    長さの目安: {info['duration']}")


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
                    print_pattern_detail(pattern)
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
