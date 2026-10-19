"""
記録した動きを、止まらずに連続で再生する（Vino の前で何パターンも試す用、連続再生版）

play_session.py との違い：1ファイルごとの「Enter=再生」待ちが無く、ホーム姿勢で終わる
記録を次々に繋げて再生する。各パターンの切れ目でイヤホンにビープ音を鳴らす。
記録はどれもホーム姿勢で始まり終わる前提だが、テイクごとに微妙なズレがあるので、
次のパターンへは --transition-sec（既定0.5秒）でゆっくり移動してから繋げる
（ズレをそのままサーボの最大速度でスナップさせると、Vino を驚かせる速い動きになるため）。

反応は direct_teach.py の --log と同じ 0〜3 をターミナルに打ち込む（Enter で確定、
次のパターンはそのまま裏で再生され続けるので、打ち込んでいる間もアームは止まらない）。
詳しいメモはイヤホンのマイクに話しかけて、声でまとめて録音しておく（セッション後に聞き返す）。

使い方:
  python play_session_continuous.py --port /dev/ttyACM0 --id my_follower recordings/*_0*.json \
    --speed 1.5 --shuffle --camera 0

Ctrl+C は（play_session.py と違い）その場でセッション全体を緊急停止する
（Vino が羽を咥えた場合など、すぐ次のパターンに進んでは危ないため）。

sessions/<日時>_events.csv にどのファイルをいつ再生したかを、sessions/<日時>.wav に
声の録音を残す。<日時>.mp4・frames.csv は --camera を付けたときだけ（play_session.py と同じ）。
打ち込んだ 0〜3 は、直前に再生が終わっていたファイルに割り当てて recordings/reactions.csv に
記録する（声のどのあたりのメモに対応するかも note 列に残す）。
"""
import argparse
import csv
import random
import threading
import time
from datetime import datetime
from pathlib import Path

from audio import BeepPlayer, VoiceRecorder, ensure_headset_profile, find_node_name
from direct_teach import connect, hold_current_pose, load_motion, move_to, play_pattern, release
from reactions import REACTIONS, log_reaction, pattern_name

EVENT_FIELDS = ["file", "pattern", "speed", "loop", "approach_sec", "approach_start", "play_start", "play_end", "interrupted"]


class ScoreInput:
    """バックグラウンドで 0〜3 の入力を受け取る（input() は別スレッドなのでアームの動きを止めない）"""

    def __init__(self):
        self.entries: list[tuple[float, str]] = []
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def _loop(self) -> None:
        while True:
            try:
                ans = input()
            except EOFError:
                return
            if ans.strip():
                self.entries.append((time.time(), ans.strip()))


def match_scores(entries: list[tuple[float, str]], event_rows: list[dict], voice_start: float | None) -> None:
    """入力された (時刻, 文字列) を、その時刻の直前に再生が終わっていたファイルに割り当てて記録する"""
    done = [e for e in event_rows if e["play_end"]]
    for t, raw in entries:
        if raw not in REACTIONS:
            print(f"  ⚠️ 無視した入力: {raw!r}（0〜3で入力してください）")
            continue
        target = None
        for e in done:
            if float(e["play_end"]) <= t:
                target = e
            else:
                break
        if target is None:
            print(f"  ⚠️ 入力 {raw!r} に対応するファイルが見つかりませんでした")
            continue
        note = f"声:{t - voice_start:.1f}s" if voice_start else ""
        log_reaction(target["file"], target["speed"], int(raw), note)


def main() -> None:
    p = argparse.ArgumentParser(description="記録した動きを止まらずに連続再生する")
    p.add_argument("files", nargs="+", help="再生する記録ファイル（例: recordings/*_0*.json）")
    p.add_argument("--port", required=True, help="例: /dev/ttyACM0（lerobot-find-port で確認）")
    p.add_argument("--id", required=True, help="キャリブレーション時の id")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--speed", type=float, default=1.0, help="1.5 なら1.5倍速")
    p.add_argument("--loop", type=int, default=1, help="1ファイルあたりの繰り返し回数")
    p.add_argument("--approach-sec", type=float, default=3.0, help="最初のファイルだけ、開始位置までの移動時間")
    p.add_argument("--transition-sec", type=float, default=0.5, help="2本目以降、次のパターンの開始位置へゆっくり移動する時間")
    p.add_argument("--shuffle", action="store_true", help="順番をランダムにする（同じ順番だと猫が慣れるため）")
    p.add_argument("--camera", help="録画するカメラ番号（例: 0）。省略すると録画しない")
    p.add_argument("--cam-width", type=int, default=640)
    p.add_argument("--cam-height", type=int, default=480)
    p.add_argument("--cam-fps", type=int, default=30)
    p.add_argument("--cam-exposure", type=int, default=None)
    p.add_argument("--cam-gain", type=int, default=None)
    p.add_argument("--cam-fourcc", default="MJPG")
    p.add_argument(
        "--cam-rotate", choices=["cw", "ccw", "180"], default=None,
        help="Iriun はスマホの向きに合わせて回転してくれないので、ソフト側で直す。縦持ちの向きで cw/ccw が変わるので試し撮りで確認すること",
    )
    p.add_argument("--earphone-name", default="Philips SN503", help="ビープを鳴らすBluetoothイヤホンの名前（部分一致）")
    p.add_argument("--mic-name", default=None, help="声を録音するマイクの名前（部分一致）。省略するとイヤホンのマイクを使う")
    p.add_argument("--beep-freq", type=float, default=880.0)
    p.add_argument("--beep-duration", type=float, default=0.6)
    p.add_argument("--beep-volume", type=float, default=0.2)
    p.add_argument("--session-dir", default="sessions")
    args = p.parse_args()

    files = list(args.files)
    if args.shuffle:
        random.shuffle(files)

    # アームに触る前に、音とカメラが使えるか確かめておく
    ensure_headset_profile(args.earphone_name)
    sink_name = find_node_name(args.earphone_name, "Audio/Sink")
    mic_name = find_node_name(args.mic_name or args.earphone_name, "Audio/Source")

    session = Path(args.session_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    session.parent.mkdir(parents=True, exist_ok=True)

    recorder = None
    if args.camera is not None:
        from camera import CameraRecorder, parse_source

        recorder = CameraRecorder(
            parse_source(args.camera), session.with_suffix(".mp4"),
            args.cam_width, args.cam_height, args.cam_fps, args.cam_exposure, args.cam_gain, args.cam_fourcc,
            rotate=args.cam_rotate,
        )

    robot = connect(args.port, args.id)

    # pw-cat を実際に起動するのはアーム接続に成功したあと（失敗時にプロセスを残さないため）
    beep = BeepPlayer(sink_name, args.beep_freq, args.beep_duration, args.beep_volume)
    voice = VoiceRecorder(mic_name, session.with_suffix(".wav"))

    events = session.with_name(session.name + "_events.csv").open("w", newline="")
    ev = csv.DictWriter(events, fieldnames=EVENT_FIELDS)
    ev.writeheader()

    event_rows: list[dict] = []
    score_input = None
    try:
        print("アームを手で支えてください（トルクを入れます）")
        input("準備ができたら Enter")
        hold_current_pose(robot)
        if recorder:
            recorder.start()
        beep.start()
        voice.start()
        score_input = ScoreInput()
        score_input.start()
        print("連続再生を開始します（Ctrl+C で緊急停止）")
        print("反応は 0〜3 を入力して Enter（0=無視 1=見た 2=構えた 3=飛びかかった）。詳しい内容は声で話してください\n")

        first = True
        for i, f in enumerate(files, 1):
            t, P = load_motion(f)
            approach_start = time.time()
            play_start = play_end = None
            interrupted = False
            # 1本目はどこからでも来られるので approach-sec（長め）、2本目以降はホーム姿勢の
            # 微妙なズレを直すだけなので transition-sec（短め）でゆっくり移動してからつなげる
            approach_sec = args.approach_sec if first else args.transition_sec
            try:
                print("開始位置へ移動中..." if first else "次のパターンへ移動中...")
                move_to(robot, P[0], approach_sec, args.fps)
                first = False
                play_start = time.time()
                play_pattern(robot, t, P, args.speed, args.loop, args.fps, approach_sec)
                play_end = time.time()
            except KeyboardInterrupt:
                play_end = time.time()
                interrupted = True

            row = {
                "file": Path(f).name, "pattern": pattern_name(f), "speed": args.speed, "loop": args.loop,
                "approach_sec": approach_sec,
                "approach_start": f"{approach_start:.4f}",
                "play_start": f"{play_start:.4f}" if play_start else "",
                "play_end": f"{play_end:.4f}" if play_end else "",
                "interrupted": interrupted,
            }
            ev.writerow(row)
            events.flush()
            event_rows.append(row)

            if interrupted:
                print(f"\n■ {f} を中断しました。セッションを終了します")
                break

            beep.beep()
            print(f"[{i}/{len(files)}] {f} 完了")
        else:
            print("\n連続再生、終了")
    except KeyboardInterrupt:
        print("\n■ セッションを中断しました")
    finally:
        try:
            release(robot)
        finally:
            events.close()
            if recorder:
                recorder.stop()
            beep.stop()
            voice.stop()
            if score_input is not None and score_input.entries:
                print("\n入力されたスコアを割り当てています...")
                match_scores(score_input.entries, event_rows, voice.start_time)
            print(f"ログ: {events.name}")
            print(f"声の録音: {voice.out_path}")


if __name__ == "__main__":
    main()
