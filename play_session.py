"""
記録した動きを、トルクを入れたまま順番に再生する（Vino の前で何パターンも試す用）

使い方:
  python play_session.py --port /dev/ttyACM0 --id my_follower recordings/*_0*.json
  python play_session.py --port /dev/ttyACM0 --id my_follower recordings/*_0*.json --speed 1.5 --shuffle --log
  カメラで録画もする: ... --camera 0

1ファイルごとに「Enter=再生 / s=飛ばす / q=終了」と聞く。
再生中の Ctrl+C はそのファイルだけ止めて次へ進む。終了するときは必ずアームを支えてからトルクを切る。

sessions/<日時>_events.csv に、どのファイルをいつ再生したかを残す（time.time() の秒）。
approach_start は開始位置への移動を始めた時刻、play_start〜play_end が実際に記録済みの
動きを再生していた区間。
--camera を付けると sessions/<日時>.mp4 と <日時>_frames.csv も作る。時計が同じなので、
play_start〜play_end の区間だけを動画から正確に切り出せる。
"""
import argparse
import csv
import random
import time
from datetime import datetime
from pathlib import Path

from direct_teach import connect, hold_current_pose, load_motion, move_to, play_pattern, release
from reactions import pattern_name

# approach_start は開始位置への移動を始めた時刻。play_start〜play_end が実際に再生していた区間
# （カメラの frames.csv も time.time() なので、この区間だけ動画から切り出せる）
EVENT_FIELDS = ["file", "pattern", "speed", "loop", "approach_sec", "approach_start", "play_start", "play_end", "interrupted"]


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
    p.add_argument("--camera", help="録画するカメラ番号（例: 0）。省略すると録画しない")
    p.add_argument("--cam-width", type=int, default=640)
    p.add_argument("--cam-height", type=int, default=480)
    p.add_argument("--cam-fps", type=int, default=30)
    p.add_argument("--cam-exposure", type=int, default=None, help="手動露光時間（例: 220）。auto_exposure だと fps が20くらいに落ちるカメラ用")
    p.add_argument("--cam-gain", type=int, default=None, help="手動ゲイン（0〜63）。暗いときに明るさを足す")
    p.add_argument(
        "--cam-fourcc",
        default="MJPG",
        help="MJPG（既定。フレームが2〜7割壊れることがあるが使える）/ YUYV（このカメラでは OpenCV の不具合で映像が壊れる。非推奨）",
    )
    p.add_argument(
        "--cam-rotate", choices=["cw", "ccw", "180"], default=None,
        help="Iriun はスマホの向きに合わせて回転してくれないので、ソフト側で直す。縦持ちの向きで cw/ccw が変わるので試し撮りで確認すること",
    )
    p.add_argument("--session-dir", default="sessions")
    args = p.parse_args()

    files = list(args.files)
    if args.shuffle:
        random.shuffle(files)

    session = Path(args.session_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    session.parent.mkdir(parents=True, exist_ok=True)

    recorder = None
    if args.camera is not None:
        # トルクを入れる前に、カメラが使えるか確かめておく
        from camera import CameraRecorder, parse_source

        recorder = CameraRecorder(
            parse_source(args.camera),
            session.with_suffix(".mp4"),
            args.cam_width,
            args.cam_height,
            args.cam_fps,
            args.cam_exposure,
            args.cam_gain,
            args.cam_fourcc,
            rotate=args.cam_rotate,
        )

    robot = connect(args.port, args.id)
    events = session.with_name(session.name + "_events.csv").open("w", newline="")
    ev = csv.DictWriter(events, fieldnames=EVENT_FIELDS)
    ev.writeheader()
    try:
        print("アームを手で支えてください（トルクを入れます）")
        input("準備ができたら Enter")
        hold_current_pose(robot)
        if recorder:
            recorder.start()
        print("トルクONのまま、最後まで順番に再生します\n")

        for i, f in enumerate(files, 1):
            ans = input(f"[{i}/{len(files)}] {f}  Enter=再生 / s=飛ばす / q=終了: ").strip().lower()
            if ans == "q":
                break
            if ans == "s":
                continue

            t, P = load_motion(f)
            interrupted = False
            play_start = play_end = None
            approach_start = time.time()
            try:
                print("開始位置へ移動中...")
                move_to(robot, P[0], args.approach_sec, args.fps)
                play_start = time.time()
                play_pattern(robot, t, P, args.speed, args.loop, args.fps, args.approach_sec)
                play_end = time.time()
            except KeyboardInterrupt:
                interrupted = True
                play_end = time.time()
                print("\n■ このファイルを中断しました")
            ev.writerow({
                "file": Path(f).name, "pattern": pattern_name(f), "speed": args.speed, "loop": args.loop, "approach_sec": args.approach_sec,
                "approach_start": f"{approach_start:.4f}",
                "play_start": f"{play_start:.4f}" if play_start else "",
                "play_end": f"{play_end:.4f}" if play_end else "",
                "interrupted": interrupted,
            })
            events.flush()
            if args.log:
                from reactions import ask_and_log

                ask_and_log(f, args.speed)
        print("\nセッション終了")
    except KeyboardInterrupt:
        print("\n■ セッションを中断しました")
    finally:
        # どこで止めても、アームを支えてもらってからトルクを切る
        try:
            release(robot)
        finally:
            events.close()
            if recorder:
                recorder.stop()
            print(f"ログ: {events.name}")


if __name__ == "__main__":
    main()
