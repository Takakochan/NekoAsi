# NekoAsi

NekoAsi is a personal robotics project: an [SO-101](https://github.com/TheRobotStudio/SO-ARM100) robot arm plays with a cat teaser (猫じゃらし) to interact with Vino, a cat I'm looking after. The long-term goal is to have the arm react to the cat's behavior instead of just replaying fixed motions.

This is also a hands-on way to learn ML engineering: recording real motion data, syncing it with video, running off-the-shelf perception models (YOLO) on the result, and building the pipeline needed to go from "robot moves a toy" to "robot responds to the cat."

## How it works

1. A motion is taught by hand (direct teaching) and recorded as a trajectory.
2. The arm replays that trajectory while a camera records the cat's reaction.
3. Playback events and video frames are logged on the same clock, so any segment of video can be matched back to the motion that was playing.
4. [YOLO11n](https://github.com/ultralytics/ultralytics) detects the cat in each frame to build a trajectory of the cat's position, as a first step toward measuring reactions automatically instead of scoring them by eye.

## Repo layout

| File | Purpose |
| --- | --- |
| `direct_teach.py` | Shared helpers (connect to the arm, record/load/play a motion) used by the other scripts |
| `record_session.py` / `play_session.py` / `play_session_continuous.py` | Record a motion by hand; play back one or many motions |
| `camera.py` | Camera capture (recording + live preview), including rotation correction for phone-as-webcam setups |
| `audio.py` | Audio playback/recording helpers for the live sessions |
| `json_to_lerobot.py` | Convert recorded motions + synced video into [LeRobot](https://github.com/huggingface/lerobot) dataset format and upload to Hugging Face |
| `analyze_session.py` | Run YOLO on a recorded session to track the cat and flag candidate "pounce" moments |
| `ml_demo.py` | Minimal end-to-end ML walkthrough on a single recorded episode |
| `reactions.py` | Manual reaction scoring helpers |
| `recordings/` | Recorded motion trajectories (JSON) and analysis output (metrics, traces) |
| `calibration/` | Arm calibration data |

Scripts share the connection/record/playback helpers in `direct_teach.py` rather than duplicating them.

## Data

Raw camera/audio recordings are not stored in this repository, since they show the inside of my home. They're published separately on Hugging Face:

- [`NekoAsi/nekoasi-motions`](https://huggingface.co/datasets/NekoAsi/nekoasi-motions) — recorded motion trajectories (no video)
- [`NekoAsi/nekoasi-sessions`](https://huggingface.co/datasets/NekoAsi/nekoasi-sessions) — synced video/audio sessions

## Safety note

`record_session.py`, `play_session.py`, `play_session_continuous.py`, and `lerobot-calibrate` all toggle the arm's torque. When torque is off, the arm can drop under its own weight, so a person needs to be physically supporting it while these run. Scripts that don't move the arm (analysis, camera checks, etc.) are safe to run on their own.
