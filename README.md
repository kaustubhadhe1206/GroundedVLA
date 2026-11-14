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

## Interactive demo (laptop, browser)

```powershell
.\demo smolvla_libero --task T1
.\demo smolvla_ckpt --checkpoint D:\groundedvla\checkpoints\v1_rgb\step_020000 --horizon 10 --task T3
# two models side by side on the identical scene (e.g. V1 vs V3 with the camera rotated)
.\demo smolvla_ckpt --checkpoint <v1 step dir> --label V1 --compare smolvla_ckpt --compare-checkpoint <v3 step dir> --compare-label V3
```

Then open http://127.0.0.1:8765. The left side has:
- the scene picker (T1–T7, initial state)
- **test conditions**: camera yaw, height and field of view, and a target shift in cm, all applied on reset
- an instruction box with one-click suggestions (the trained instruction and its paraphrases)
- status and progress, per-model results, the scene goal, and the two images the model sees

The right side shows the live simulator with any scene camera (agentview, frontview, sideview, ...). When comparing, both models run from the identical scene, one after the other, and play back side by side with their outcome.

- The robot only moves after you press **Run** (Ctrl+Enter).
- A later prompt continues from the current state until you press **Reset**.
- **Stop** ends the current instruction.

**Free-form prompts.** Policies can only perform skills they were trained on. With **Map my words to a trained skill** (on by default), the prompt is parsed into an object and a destination, then matched against this scene's skills. The kitchen scene has T3, T4, T6 and T7; a spatial scene has only its own task.
- If both the object and the destination agree with a skill, the policy gets that skill's trained instruction, and "goal satisfied" checks that skill's goal. For example, "stick the bowl into the top drawer" in the kitchen is understood as T6.
- Otherwise the page says no trained skill fits, the prompt goes to the policy as typed, and there is no goal check. For example, "put the bowl on top of the cabinet" fails because there is no bowl-on-cabinet skill.
- With the box unticked, the prompt is sent word for word and the scene's own goal is checked.

Speed: a policy call takes about 0.8 s on the laptop's RTX 3050, and `smolvla_libero` re-plans every step, so live motion runs at about 1 step/s. With **Smooth playback** (on by default) the run is computed first, with a live preview and a progress bar, then played at the real 20 Hz. **Replay** plays the last run again, which is handy for screen recording. Untick the box, or pass `--live`, to watch each step as it is computed. Our checkpoints with N = 10 plan 10 steps per call, so they are about 10× faster.

## Tasks

| Key | Suite | Instruction | Max steps |
|---|---|---|---|
| T1 | libero_spatial | pick up the black bowl between the plate and the ramekin and place it on the plate | 220 |
| T2 | libero_spatial | pick up the black bowl next to the cookie box and place it on the plate | 220 |
| T3 | libero_goal | put the bowl on the stove | 300 |
| T4 | libero_goal | put the bowl on the plate | 300 |
| T5 | libero_spatial | pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate | 220 |
| T6 | libero_goal | open the top drawer and put the bowl inside | 300 |
| T7 | libero_goal | put the wine bottle on top of the cabinet | 300 |

T3, T4, T6 and T7 share one kitchen scene: same objects, different goals, so only the instruction says which skill to do. T5–T7 (added Nov 13) get a lighter evaluation: in-distribution and camera yaw ±20°, 25 rollouts each (`light_eval` in `configs/eval_ood.yaml`).

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
| `03_train.ipynb` | B1/B3: throughput test, training with auto-resume, checkpoint selection, horizon tuning, first test; `CONFIG` picks the variant, `SEED` the training seed | A100 |
| `04_eval.ipynb` | B4: every condition in `configs/eval_ood.yaml` for one run, then the report and paired comparisons across runs | A100 |

Training runs (each with the same data, budget and evaluation):

| Run | Config | Purpose |
|---|---|---|
| `v1_rgb`, `v1_rgb_s1` | `v1_rgb.yaml` (seeds 0, 1) | RGB baseline |
| `v1_aug` | `v1_aug.yaml` | RGB + camera/color augmentation: does plain augmentation buy the same robustness as 3D? |
| `v2_mask` | `v2_mask.yaml` | + target/destination mask (object identity) |
| `v3_grounded`, `v3_grounded_s1` | `v3_grounded.yaml` (seeds 0, 1) | + 3D grounding tokens |

Datasets and weights are copied from Drive to `/content/local` before use (Drive is slow for many small reads). Checkpoints and results are written back to Drive.

## Repository layout

See plan Section 4. Main entry points:

- `scripts/run_sim.py`: run any model, with a viewer or headless
- `groundedvla/train/train.py`: fine-tuning for V1/V2/V3 (checkpoints every 2k steps, auto-resume)
- `groundedvla/eval/run_eval.py`: the evaluation harness (resumable JSONL; `--split val` for checkpoint selection)
- `groundedvla/eval/report.py`: summary tables with 95% confidence intervals
- `groundedvla/data/replay_demos.py`, `make_split_manifest.py`, `convert_to_lerobot.py`, `check_dataset.py`: the dataset build, run in that order
- `groundedvla/grounding/check_grounding.py`: grounding accuracy and camera invariance checks
- `groundedvla/grounding/perception.py`, `eval_perception.py`: masks predicted from the instruction (Grounding DINO + spatial relation + SAM) instead of the simulator's, and their accuracy
- `groundedvla/eval/report.py --compare A B`: paired success difference on matched episodes with a bootstrap 95% CI
- `groundedvla/envs/compat.py`: rendering backend plus the robosuite/NumPy 2 segmentation fix (imported automatically)

## Acknowledgments

LIBERO (Liu et al.), robosuite, MuJoCo, LeRobot and SmolVLA (Hugging Face), `HuggingFaceVLA/smolvla_libero`, OpenVLA (Kim et al.).
