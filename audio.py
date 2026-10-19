"""
連続再生セッション用の音声まわり：イヤホンにビープ音を鳴らし、マイクの声を録音する

PipeWire（`pw-cat`）をパイプで繋いで使う。sounddevice（PortAudio/ALSA経由）だと
Bluetoothイヤホンには届かなかったため（素のALSAハードウェアしか見えない）。

PipeWireのノード番号は他の音（Chromeなど）が動くたびに変わって不安定なので、
`pw-dump` で毎回「説明にこの文字列を含むノード」を名前で探してから使う。

Bluetoothイヤホンは、しばらく音を送っていないと次の音の頭が欠けることがあるため、
BeepPlayer はセッション開始から終了まで無音を流し続け、ビープの瞬間だけ音を差し込む。
"""
import json
import subprocess
import threading
import time
import wave
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000


def ensure_headset_profile(description_contains: str) -> None:
    """Bluetoothイヤホンが Headset Head Unit (HSP/HFP) プロファイルになっていなければ切り替える。
    繋ぎ直すたびに音質優先の A2DP（スピーカーのみ、マイク無し）に戻ってしまうことがあるため、
    マイクを探す前に毎回呼んでおく"""
    out = subprocess.run(["pw-dump"], capture_output=True, text=True, check=True).stdout
    for obj in json.loads(out):
        props = (obj.get("info") or {}).get("props") or {}
        if props.get("device.api") != "bluez5" or description_contains not in (props.get("device.description") or ""):
            continue
        if (props.get("bluez5.profile") or "").startswith("headset-head-unit"):
            return  # もう切り替わっている
        profiles = (obj.get("info") or {}).get("params", {}).get("EnumProfile", [])
        headset = [p for p in profiles if p["name"].startswith("headset-head-unit")]
        if not headset:
            return
        best = max(headset, key=lambda p: p["priority"])  # 優先度が一番高いコーデックを選ぶ
        subprocess.run(["wpctl", "set-profile", str(obj["id"]), str(best["index"])], check=True)
        time.sleep(1.0)  # 切り替わるまで少し待つ
        return


def find_node_name(description_contains: str, media_class: str) -> str:
    """pw-dump から、説明に description_contains を含む media_class のノード名を探す"""
    out = subprocess.run(["pw-dump"], capture_output=True, text=True, check=True).stdout
    for obj in json.loads(out):
        props = (obj.get("info") or {}).get("props") or {}
        if media_class == props.get("media.class") and description_contains in (props.get("node.description") or ""):
            return props["node.name"]
    raise RuntimeError(f"「{description_contains}」の{media_class}が見つかりません。接続を確認してください")


def make_beep(freq: float = 880.0, duration: float = 0.6, volume: float = 0.2) -> bytes:
    """16bit PCM モノラルのビープ音を作る"""
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
    tone = (volume * np.sin(2 * np.pi * freq * t) * 32767).astype("<i2")
    return tone.tobytes()


class BeepPlayer:
    """pw-cat の再生プロセスを開いたままにして無音を流し続け、beep() で音を差し込む。
    実際にプロセスを起動するのは start() のときだけ（アーム接続前に作るだけで
    pw-cat を起動してしまうと、あとで接続に失敗したときにプロセスが残ってしまうため）"""

    def __init__(self, target_name: str, freq: float = 880.0, duration: float = 0.6, volume: float = 0.2):
        self.target_name = target_name
        self._tone = make_beep(freq, duration, volume)
        self.proc = None
        self._stop = False
        self._thread = None

    def _fill_silence(self) -> None:
        silence = b"\x00\x00" * (SAMPLE_RATE // 20)  # 50ms 分
        while not self._stop:
            try:
                self.proc.stdin.write(silence)
                self.proc.stdin.flush()
            except (BrokenPipeError, ValueError):
                break
            time.sleep(0.04)

    def start(self) -> None:
        self.proc = subprocess.Popen(
            ["pw-cat", "--playback", "--target", self.target_name, "--format", "s16",
             "--rate", str(SAMPLE_RATE), "--channels", "1", "-"],
            stdin=subprocess.PIPE,
        )
        self._thread = threading.Thread(target=self._fill_silence, daemon=True)
        self._thread.start()

    def beep(self) -> None:
        try:
            self.proc.stdin.write(self._tone)
            self.proc.stdin.flush()
        except (BrokenPipeError, ValueError):
            print("⚠️ ビープ音を送れませんでした（イヤホンの接続を確認してください）")

    def stop(self) -> None:
        self._stop = True
        if self._thread is not None:
            self._thread.join(timeout=1)
        if self.proc is None:
            return
        try:
            self.proc.stdin.close()
        except Exception:
            pass
        self.proc.terminate()
        self.proc.wait(timeout=2)


class VoiceRecorder:
    """pw-cat の録音プロセスを開始から終了まで繋ぎっぱなしにして、1本の wav に保存する"""

    def __init__(self, target_name: str, out_path: Path):
        self.target_name = target_name
        self.out_path = Path(out_path)
        self.proc = None
        self._chunks: list[bytes] = []
        self._thread = None
        self.start_time = None

    def start(self) -> None:
        self.start_time = time.time()
        self.proc = subprocess.Popen(
            ["pw-cat", "--record", "--target", self.target_name, "--format", "s16",
             "--rate", str(SAMPLE_RATE), "--channels", "1", "-"],
            stdout=subprocess.PIPE,
        )
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def _read_loop(self) -> None:
        while True:
            chunk = self.proc.stdout.read(4096)
            if not chunk:
                break
            self._chunks.append(chunk)

    def stop(self) -> None:
        if self.proc is None:
            return
        self.proc.terminate()
        self.proc.wait(timeout=2)
        self._thread.join(timeout=2)
        data = b"".join(self._chunks)
        with wave.open(str(self.out_path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SAMPLE_RATE)
            w.writeframes(data)
