"""
Vino の反応メモ（recordings/reactions.csv に1行ずつ足していく）

使い方:
  記録: python reactions.py add recordings/flick_01.json --speed 1.5
  集計: python reactions.py summary

再生と一緒に記録するなら: python direct_teach.py play ... --log
"""
import argparse
import csv
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

CSV_PATH = Path(__file__).parent / "recordings" / "reactions.csv"
FIELDS = ["datetime", "file", "pattern", "speed", "reaction", "score", "note"]
# score は反応の強さ。Phase 5 でパターンを比べるときに使う
REACTIONS = {"0": "無視", "1": "見た", "2": "構えた", "3": "飛びかかった"}


def pattern_name(file: str) -> str:
    """flick_02.json → flick"""
    return re.sub(r"_\d+$", "", Path(file).stem)


def log_reaction(file: str, speed: float, score: int, note: str = "") -> None:
    new = not CSV_PATH.exists()
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow({
            "datetime": datetime.now().isoformat(timespec="seconds"),
            "file": Path(file).name,
            "pattern": pattern_name(file),
            "speed": speed,
            "reaction": REACTIONS[str(score)],
            "score": score,
            "note": note,
        })
    print(f"📝 {CSV_PATH} に記録しました")


def ask_and_log(file: str, speed: float) -> None:
    choices = " / ".join(f"{k}={v}" for k, v in REACTIONS.items())
    while True:
        ans = input(f"Vino の反応は？（{choices} / Enter=記録しない）: ").strip()
        if ans == "":
            print("記録しませんでした")
            return
        if ans in REACTIONS:
            break
    note = input("メモ（任意）: ").strip()
    log_reaction(file, speed, int(ans), note)


def summary() -> None:
    if not CSV_PATH.exists():
        print("まだ記録がありません")
        return
    scores = defaultdict(list)
    with CSV_PATH.open() as f:
        for row in csv.DictReader(f):
            scores[row["pattern"]].append(int(row["score"]))
    print(f"{'pattern':14s} {'回数':>4s} {'平均':>5s}  " + "  ".join(REACTIONS.values()))
    for pat, s in sorted(scores.items(), key=lambda kv: -sum(kv[1]) / len(kv[1])):
        counts = "  ".join(f"{s.count(int(k)):>{len(v)}d}" for k, v in REACTIONS.items())
        print(f"{pat:14s} {len(s):>4d} {sum(s) / len(s):>5.2f}  {counts}")


def main() -> None:
    p = argparse.ArgumentParser(description="Vino の反応メモ")
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("add")
    a.add_argument("file", help="再生した記録ファイル")
    a.add_argument("--speed", type=float, default=1.0)
    sub.add_parser("summary")
    args = p.parse_args()
    if args.cmd == "add":
        ask_and_log(args.file, args.speed)
    else:
        summary()


if __name__ == "__main__":
    main()
