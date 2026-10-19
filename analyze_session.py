"""
録画したセッション映像から、Vino の反応指標を出す（Phase 4：猫の状態をカメラで認識する）

使い方:
  python analyze_session.py --session sessions/20261018_112710
  python analyze_session.py --session sessions/20261018_112710 --only air_hover_01.json  # 動作確認用

sessions/<session>_events.csv の play_start〜play_end 区間ごとに、動画を1回だけ順番に読みながら
YOLO（COCO の cat クラス）で猫の位置を検出する。中心位置が動く速さが --pounce-speed を超えた瞬間を
「飛びかかり」の候補とする、最初のシンプルなルール。閾値は実データを見ながら --pounce-speed で調整する。

出力:
  recordings/video_metrics.csv       1テイク1行で追記（file, pattern, speed, 検出率, 飛びかかり回数・時刻）
  recordings/video_traces/<file>.csv 1テイクごとの猫の中心位置の軌跡（閾値調整・グラフ化用）

羽の位置はまだ色で追跡できていない（羽が黒っぽく、背景・猫の毛色と被るため）ので、
「羽の近くにいた時間」はこのスクリプトではまだ出さない。
"""
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from camera import apply_rotation

CAT_CLASS = 15  # COCO の cat クラス
METRICS_FIELDS = ["file", "pattern", "speed", "n_frames", "detected_ratio", "pounce_count", "time_to_pounce"]


def load_events(events_path: Path, only: set[str] | None) -> list[dict]:
    with events_path.open() as f:
        rows = [r for r in csv.DictReader(f) if r["interrupted"] != "True" and r["play_start"] and r["play_end"]]
    if only:
        rows = [r for r in rows if r["file"] in only]
    return sorted(rows, key=lambda r: float(r["play_start"]))


def frame_range(frame_times: np.ndarray, start: float, end: float) -> tuple[int, int]:
    """時刻 [start, end] に対応するフレーム番号の範囲を返す"""
    i0 = int(np.searchsorted(frame_times, start))
    i1 = int(np.searchsorted(frame_times, end))
    return i0, min(i1, len(frame_times) - 1)


def detect_cat(model: YOLO, frame, conf: float):
    """一番信頼度の高い猫の矩形を (中心x, 中心y) で返す。見つからなければ None"""
    r = model.predict(frame, classes=[CAT_CLASS], conf=conf, verbose=False)[0]
    if len(r.boxes) == 0:
        return None
    box = max(r.boxes, key=lambda b: float(b.conf[0]))
    x1, y1, x2, y2 = box.xyxy[0].tolist()
    return (x1 + x2) / 2, (y1 + y2) / 2


def resample_trace(trace: list[tuple[float, float, float]], dt: float = 1 / 15, max_gap: float = 0.3):
    """trace（検出できたフレームだけ、間隔は不揃い）を一定間隔 dt に補間する。
    検出が max_gap 秒より途切れた区間は（猫が隠れた・検出が外れたとみなし）補間しない。
    フレームごとの検出のブレで瞬間速度が暴れるのを抑えるため"""
    if len(trace) < 2:
        return []
    t = np.array([p[0] for p in trace])
    x = np.array([p[1] for p in trace])
    y = np.array([p[2] for p in trace])
    grid = np.arange(t[0], t[-1], dt)
    idx = np.clip(np.searchsorted(t, grid), 1, len(t) - 1)
    out = []
    for gi, ti in zip(idx, grid):
        t0, t1 = t[gi - 1], t[gi]
        if t1 - t0 > max_gap:
            continue
        frac = 0.0 if t1 == t0 else (ti - t0) / (t1 - t0)
        out.append((ti, x[gi - 1] + frac * (x[gi] - x[gi - 1]), y[gi - 1] + frac * (y[gi] - y[gi - 1])))
    return out


def find_pounces(trace: list[tuple[float, float, float]], speed_threshold: float, min_gap: float) -> list[float]:
    """trace = [(時刻, 中心x, 中心y), ...]（検出できたフレームだけ、間隔は不揃い）。
    一定間隔に補間してから、中心の移動速度が speed_threshold(px/秒) を超えた瞬間を
    「飛びかかり」の候補とする。min_gap 秒以内に連続するものは1回にまとめる"""
    resampled = resample_trace(trace)
    pounces = []
    last_t = None
    for (t0, x0, y0), (t1, x1, y1) in zip(resampled, resampled[1:]):
        dt = t1 - t0
        if dt <= 0:
            continue
        speed = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5 / dt
        if speed >= speed_threshold and (last_t is None or t1 - last_t >= min_gap):
            pounces.append(t1)
            last_t = t1
    return pounces


def analyze_session(session: Path, model: YOLO, conf: float, speed_threshold: float, min_gap: float,
                     out_path: Path, trace_dir: Path, only: set[str] | None, rotate: str | None = None) -> None:
    events = load_events(session.with_name(session.name + "_events.csv"), only)
    if not events:
        print(f"{session}: 有効なイベントが無い")
        return
    frame_rows = list(csv.DictReader(session.with_name(session.name + "_frames.csv").open()))
    frame_times = np.array([float(r["time"]) for r in frame_rows])

    ranges = [(frame_range(frame_times, float(e["play_start"]), float(e["play_end"])), e) for e in events]

    cap = cv2.VideoCapture(str(session.with_suffix(".mp4")))
    trace_dir.mkdir(parents=True, exist_ok=True)
    new_file = not out_path.exists()
    out = out_path.open("a", newline="")
    writer = csv.DictWriter(out, fieldnames=METRICS_FIELDS)
    if new_file:
        writer.writeheader()

    frame_idx = -1
    for (i0, i1), ev in ranges:
        while frame_idx < i0 - 1:  # イベント開始まで読み飛ばす（YOLOは掛けない）
            if not cap.grab():
                break
            frame_idx += 1

        trace = []
        n_frames = 0
        while frame_idx < i1:
            ok, frame = cap.read()
            frame_idx += 1
            if not ok:
                break
            n_frames += 1
            frame = apply_rotation(frame, rotate)
            det = detect_cat(model, frame, conf)
            if det is not None:
                trace.append((frame_times[frame_idx], det[0], det[1]))

        pounces = find_pounces(trace, speed_threshold, min_gap)
        detected_ratio = len(trace) / n_frames if n_frames else 0.0

        writer.writerow({
            "file": ev["file"], "pattern": ev["pattern"], "speed": ev["speed"],
            "n_frames": n_frames, "detected_ratio": f"{detected_ratio:.2f}",
            "pounce_count": len(pounces),
            "time_to_pounce": f"{pounces[0] - float(ev['play_start']):.2f}" if pounces else "",
        })
        out.flush()

        trace_path = trace_dir / f"{Path(ev['file']).stem}.csv"
        with trace_path.open("w", newline="") as tf:
            tw = csv.writer(tf)
            tw.writerow(["time", "cx", "cy"])
            tw.writerows(trace)

        print(f"{ev['file']}: {n_frames}フレーム中{len(trace)}フレームで検出、飛びかかり候補{len(pounces)}回")

    out.close()
    cap.release()


def main() -> None:
    p = argparse.ArgumentParser(description="セッション映像から Vino の反応指標を出す")
    p.add_argument("--session", required=True, help="例: sessions/20261018_112710（拡張子なし）")
    p.add_argument("--model", default="yolo11n.pt")
    p.add_argument("--conf", type=float, default=0.25, help="YOLO の信頼度のしきい値")
    p.add_argument("--pounce-speed", type=float, default=1200.0, help="飛びかかり判定の速さの閾値(px/秒)")
    p.add_argument("--min-gap", type=float, default=0.5, help="飛びかかりを1回にまとめる間隔(秒)")
    p.add_argument("--out", default="recordings/video_metrics.csv")
    p.add_argument("--trace-dir", default="recordings/video_traces")
    p.add_argument("--only", nargs="+", help="このファイル名だけ処理する（動作確認用）")
    p.add_argument(
        "--rotate", choices=["cw", "ccw", "180"], default=None,
        help="録画時の回転設定が間違っていた映像を、解析時に補正する（録画のやり直しは不要）",
    )
    args = p.parse_args()

    model = YOLO(args.model)
    analyze_session(
        Path(args.session), model, args.conf, args.pounce_speed, args.min_gap,
        Path(args.out), Path(args.trace_dir), set(args.only) if args.only else None,
        rotate=args.rotate,
    )


if __name__ == "__main__":
    main()
