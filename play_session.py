"""
記録した動きを、トルクを入れたまま順番に再生する（Vino の前で何パターンも試す用）

使い方:
  python play_session.py --port /dev/ttyACM0 --id my_follower recordings/*_0*.json
  python play_session.py --port /dev/ttyACM0 --id my_follower recordings/*_0*.json --speed 1.5 --shuffle --log

1ファイルごとに「Enter=再生 / s=飛ばす / q=終了」と聞く。
再生中の Ctrl+C はそのファイルだけ止めて次へ進む。終了するときは必ずアームを支えてからトルクを切る。
"""
import argparse
import random

from direct_teach import connect, hold_current_pose, load_motion, play_frames, release


def main() -> None:
    p = argparse.ArgumentParser(description="記録した動きを順番に再生する")
    p.add_argument("files", nargs="+", help="再生する記録ファイル（例: recordings/*_0*.json）")
    p.add_argument("--port", required=True, help="例: /dev/ttyACM0（lerobot-find-port で確認）")
    p.add_argument("--id", required=True, help="キャリブレーション時の id")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--speed", type=float, default=1.0, help="1.5 なら1.5倍速")
    p.add_argument("--loop", type=int, default=1, help="1ファイルあたりの繰り返し回数")
    p.add_argument("--approach-sec", type=float, default=3.0, help="次の開始位置までの移動時間")
    p.add_argument("--shuffle", action="store_true", help="順番をランダムにする（同じ順番だと猫が慣れるため）")
    p.add_argument("--log", action="store_true", help="1回ごとに Vino の反応を reactions.csv に記録する")
    args = p.parse_args()

    files = list(args.files)
    if args.shuffle:
        random.shuffle(files)

    robot = connect(args.port, args.id)
    try:
        print("アームを手で支えてください（トルクを入れます）")
        input("準備ができたら Enter")
        hold_current_pose(robot)
        print("トルクONのまま、最後まで順番に再生します\n")

        for i, f in enumerate(files, 1):
            ans = input(f"[{i}/{len(files)}] {f}  Enter=再生 / s=飛ばす / q=終了: ").strip().lower()
            if ans == "q":
                break
            if ans == "s":
                continue

            t, P = load_motion(f)
            try:
                play_frames(robot, t, P, args.speed, args.loop, args.fps, args.approach_sec)
            except KeyboardInterrupt:
                print("\n■ このファイルを中断しました")
            if args.log:
                from reactions import ask_and_log

                ask_and_log(f, args.speed)
        print("\nセッション終了")
    except KeyboardInterrupt:
        print("\n■ セッションを中断しました")
    finally:
        # どこで止めても、アームを支えてもらってからトルクを切る
        release(robot)


if __name__ == "__main__":
    main()
