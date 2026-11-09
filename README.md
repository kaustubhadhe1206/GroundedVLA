# GroundedVLA

Does giving a Vision-Language-Action model explicit, language-grounded 3D positions of the objects named in an instruction make it more reliable at pick-and-place, especially under position and camera shifts?

CSCI 645 (Robotic Manipulation, USC), advisor Prof. Daniel Seita. Team: Harsh Sharma, Kaustubh Adhe.

We compare three SmolVLA variants on LIBERO, all trained by behavior cloning on the same human demonstrations:

| Variant | Inputs |
|---|---|
| V1 | RGB (agentview + wrist), instruction, robot state |
| V2 | V1 + target/destination mask image |
| V3 | V2 + grounding tokens (target/destination 3D positions in the robot frame) |

Full plan: [docs/GroundedVLA_Implementation_Plan.md](docs/GroundedVLA_Implementation_Plan.md).

## Quick start: watch a model in the simulator (laptop)

```powershell
.\run replay         --task T1 --episodes 3    # expert demos (must always succeed)
.\run smolvla_libero --task T2 --episodes 5    # SmolVLA already fine-tuned on LIBERO by Hugging Face
.\run smolvla_base   --task T1                 # pretrained SmolVLA, zero-shot (plumbing test, ~0% success)
.\run smolvla_ckpt   --checkpoint D:\groundedvla\checkpoints\<run>\<step> --task T3   # our checkpoints
.\run --help
```

A MuJoCo window opens and plays the episodes one after another. It reopens once per episode, because LIBERO rebuilds the scene on reset. Closing the window stops the run. Viewer keys: `[` / `]` cycle cameras, `Esc` gives a free camera, and you can drag with the mouse to orbit.

| Option | Meaning |
|---|---|
| `--task` | `T1`..`T4`, or any LIBERO task as `<suite>:<id>`, e.g. `libero_spatial:4` |
| `--episodes`, `--start` | how many episodes, and the first initial-state / demo index |
| `--mode naive --horizon N` | execute N actions of each predicted chunk, then re-predict |
| `--mode temporal_ensemble` | ACT-style temporal ensemble (predict every step) |
| `--viewer none --save-video` | headless; mp4 per episode (agentview \| wrist) |
| `--fast` | don't slow the viewer to real time (20 Hz) |

Every episode is also logged as one JSON line in `<outputs>/run_sim/<model>/<task>.jsonl`.

OpenVLA-7B needs about 16 GB of GPU memory and its own environment, so run it on Colab with `notebooks/01b_openvla_pipeline_check.ipynb`.

## Tasks

| Key | Suite | Instruction | Max steps |
|---|---|---|---|
| T1 | libero_spatial | pick up the black bowl between the plate and the ramekin and place it on the plate | 220 |
| T2 | libero_spatial | pick up the black bowl next to the cookie box and place it on the plate | 220 |
| T3 | libero_goal | put the bowl on the stove | 300 |
| T4 (optional) | libero_goal | put the bowl on the plate | 300 |

## Observation and action format

Our wrapper (`groundedvla/envs/libero_wrapper.py`) produces exactly the Hugging Face LIBERO format, so `HuggingFaceVLA/smolvla_libero` runs unchanged:

| Key | Shape | Content |
|---|---|---|
| `observation.images.image` | 256×256×3 uint8 | agentview RGB, rotated 180° (upright) |
| `observation.images.image2` | 256×256×3 uint8 | wrist RGB, rotated 180° |
| `observation.state` | 8 float32 | eef position (3, m, world frame), eef axis-angle (3, robosuite `quat2axisangle`), gripper finger qpos (2) |
| `task` | str | the benchmark instruction (not the BDDL `:language` text) |
| `action` | 7 float32 in [-1, 1] | eef delta position (3), delta axis-angle (3), gripper (-1 open, +1 close) |

Each episode starts with 10 no-op settle steps (except expert replay). The controller is OSC_POSE at 20 Hz.

## Local setup (Windows)

Everything lives on `D:\groundedvla` (plan Section 2.5). The code in this repo is the only thing on C.

1. Run `. D:\groundedvla\activate.ps1` at the start of each terminal. It sets every cache to D: and activates `D:\groundedvla\envs\gvla`.
2. Install, if the env is new:
   ```powershell
   pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu126
   pip install -r requirements.txt
   pip install -e . --no-deps
   python scripts/patch_windows.py      # robosuite mujoco.dll + GPU-rendering macro (Windows only)
   python scripts/download_all.py --what model assets data
   ```
3. In VS Code, select `D:\groundedvla\envs\gvla\python.exe` as the interpreter (Ctrl+Shift+P, then "Python: Select Interpreter").
4. Check it: `pytest tests -q`, `python scripts/smoke_test_env.py --task T1`, `.\run replay --task T1`.

`run.bat` sets the same cache variables itself, so `.\run ...` works from any terminal.

## Colab workflow

1. Push to GitHub (public repo).
2. Open a notebook in VS Code with the Colab extension and set `REPO_URL` in its first cell.
3. The notebook mounts Drive, clones the repo, installs `requirements.txt`, and writes everything persistent to `MyDrive/groundedvla/`:

| Notebook | Purpose | GPU |
|---|---|---|
| `00_colab_setup.ipynb` | install, download demos to Drive, A1 render check, A2 expert replay | T4 is enough |
| `01_pipeline_check.ipynb` | A3: `smolvla_base` and `smolvla_libero` rollouts with videos | A100 (T4 is slower) |
| `01b_openvla_pipeline_check.ipynb` | A3: OpenVLA-7B rollouts, **separate runtime** | A100 |
| `02_build_dataset.ipynb` | A5: masks, 3D grounding and LeRobot dataset; A6 and A7 checks | T4 is enough |

Datasets and weights are copied from Drive to `/content/local` before use (Drive is slow for many small reads). Checkpoints and results are written back to Drive.

## Repository layout

See plan Section 4. Main entry points:

- `scripts/run_sim.py`: run any model, with a viewer or headless
- `groundedvla/eval/run_eval.py`: the evaluation harness (resumable JSONL)
- `groundedvla/eval/report.py`: summary tables with 95% confidence intervals
- `groundedvla/data/replay_demos.py`, `make_split_manifest.py`, `convert_to_lerobot.py`, `check_dataset.py`: the dataset build, run in that order
- `groundedvla/grounding/check_grounding.py`: grounding accuracy and camera invariance checks
- `groundedvla/envs/compat.py`: rendering backend plus the robosuite/NumPy 2 segmentation fix (imported automatically)

## Acknowledgments

LIBERO (Liu et al.), robosuite, MuJoCo, LeRobot and SmolVLA (Hugging Face), `HuggingFaceVLA/smolvla_libero`, OpenVLA (Kim et al.).
