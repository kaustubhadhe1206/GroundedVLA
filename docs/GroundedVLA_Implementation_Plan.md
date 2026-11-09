# GroundedVLA: Five-Week Implementation Plan

CSCI 645: Robotic Manipulation
Advisor: Professor Daniel Seita
Team: Harsh Sharma, Kaustubh Adhe
Duration: five weeks (Nov 1 to Dec 3, 2026). Final presentation: **Dec 3, 2026**.

---

## 1. What the Project Is About

GroundedVLA tests whether giving a Vision-Language-Action (VLA) model explicit 3D information about the objects named in an instruction makes a robot more reliable at language-conditioned pick-and-place, especially when the scene or camera differs from training.

A VLA is the robot's control policy. Every step, it takes camera images, a language instruction (for example, "pick up the black bowl between the plate and the ramekin and place it on the plate"), and the robot's state, and outputs the next arm and gripper movements. An RGB-only VLA has to infer where objects are from visual cues alone. We will compute the target and destination 3D positions in the robot's frame from simulated depth and segmentation, feed them to the model through a small adapter, and compare against the same model without that information.

The robot is trained with imitation learning (behavior cloning): it learns to copy expert demonstrations from the LIBERO benchmark. No reinforcement learning is used.

### Imitation Learning Setup

- **Teacher:** human teleoperation. LIBERO ships 50 demonstrations per task, recorded by people driving the robot with a 3D mouse. There is no PPO teacher and no scripted IK expert.
- **Fixed offline dataset:** we can't query the teacher during training, so DAgger-style corrections are not possible.
- **Learning method:** behavior cloning with SmolVLA's action expert, which generates action chunks with **flow matching**. This is a close relative of Diffusion Policy: both start from noise and refine it into actions. Flow matching learns a direct path from noise to actions, so it needs about 10 refinement steps instead of 50–100.
- **Action chunking (two execution modes, compared in Steps B1 and B4):**
  - **Naive chunking:** predict a 50-action chunk, execute the first N actions, then predict again (N tuned in Step B1).
  - **Temporal ensemble (ACT-style):** predict a chunk every step and execute an exponentially weighted average of all chunks that cover the current step (weights w_i = exp(−m·i), with i = 0 the oldest chunk and m = 0.01, as in ACT). Evaluation-time only, so no retraining is needed.

### Research Question

Does explicit, language-grounded metric 3D information improve manipulation success and spatial generalization beyond what an RGB-only VLA learns implicitly?

### Hypothesis

Grounding helps most when object positions and camera viewpoints differ from training. Positions in the robot frame stay the same when the camera moves, while image appearance changes. In familiar scenes, the RGB baseline may perform about as well.

### The Three Policy Variants

The proposal compares two policies. We add a middle variant as a control, so we can tell whether any gain comes from knowing which object matters or from knowing where it is in metric 3D.

- **V1, RGB baseline:** SmolVLA with RGB images, instruction, and robot state.
- **V2, RGB + mask (control):** V1 inputs plus an image showing the target and destination masks.
- **V3, RGB + mask + 3D (grounded policy):** V2 inputs plus a grounding adapter that adds the target and destination 3D positions as extra input tokens.

Depth is never fed to the model as an image. It is only used to compute the 3D coordinates in V3.

Reading the results:
- V2 vs V1: does explicit target identification help?
- V3 vs V2: does metric 3D help beyond identification? This is the core research question.
- V3 vs V1: total effect of grounding, matching the proposal's two-policy comparison.

All three use the same tasks, demonstrations, splits, training budget, and evaluation conditions.

### Definition of Done

- V1, V2, and V3 trained and evaluated on the same tasks, demonstrations, and initial states.
- In-distribution and out-of-distribution results for every proposal metric, plus collisions and failure categories.
- Reproducible code, split manifest, experiment logs, final report, presentation, and demo video.

### Scope Rules

- One simulator, one robot, one action space, and 3 tasks (a 4th only if time allows).
- Simulator-provided masks and depth only; no learned segmentation model.
- Small MLP for 3D features; no point-cloud networks.
- Parameter-efficient training only; the SmolVLA backbone stays frozen.
- Architecture freezes at the end of Week 3. Results freeze at the end of Week 4. No new features in Week 5.
- If a milestone slips, cut tasks or evaluation conditions before cutting rollouts per condition. Never cut V1, V3, matched evaluation, or the proposal metrics.

---

## 2. Technical Setup

### 2.1 Robot and Simulator

- **Robot:** Franka Panda, a fixed tabletop 7-DoF arm with a parallel-jaw gripper (LIBERO default).
- **Simulator:** LIBERO, built on robosuite, which runs on MuJoCo.
- **Controller:** robosuite operational-space controller (OSC_POSE) at 20 Hz.
- **Action space (7-D):** end-effector delta position (3), delta rotation in axis-angle (3), gripper open/close (1).
- **Robot state (8-D):** end-effector position (3), end-effector orientation in axis-angle (3), gripper finger joint positions (2).
- **Cameras:**
  - `agentview`: fixed third-person camera
  - `robot0_eye_in_hand`: wrist camera
  - Both render at 256 x 256 and provide RGB, depth, and instance segmentation.

### 2.2 Model: SmolVLA

- Loaded through Hugging Face LeRobot from the `lerobot/smolvla_base` checkpoint (about 450M parameters).
- Structure:
  - Vision-language backbone (SmolVLM2-based) that encodes images and the instruction into tokens
  - Robot state projected into a single token
  - Action expert that predicts a chunk of future actions with flow matching
- Default chunk size: 50 actions. Execution mode is naive chunking (first N actions) or temporal ensemble, see Section 1 and Step A4.
- Default training setup in LeRobot: vision encoder frozen and only the action expert trained. Confirmed in LeRobot 0.4.4: `freeze_vision_encoder: true`, `train_expert_only: true`, `train_state_proj: true`.
- **`smolvla_base` does not match LIBERO out of the box.** It was pretrained on a different robot (SO-100) and expects 3 cameras (`observation.images.camera1..3`), a 6-D state, and a 6-D action. Fine-tuning replaces these with our 2 cameras, 8-D state, and 7-D action. For the zero-shot test in Step A3 we feed 2 cameras, truncate the state to 6-D, and map the 6-D normalized output onto the 6 arm dimensions, with the gripper held open. That test only checks the plumbing.

### 2.2.1 Pipeline-Check Models (Step A3, never fine-tuned by us)

| Name in `run_sim` | Checkpoint | Size | Purpose |
|---|---|---|---|
| `smolvla_base` | `lerobot/smolvla_base` | ~0.45B | Plumbing test, expect ~0% success |
| `smolvla_libero` | `HuggingFaceVLA/smolvla_libero` (already fine-tuned on LIBERO by Hugging Face) | ~0.45B | Validates our SmolVLA observation/action mapping: should succeed often |
| `openvla_7b` | `openvla/openvla-7b-finetuned-libero-{spatial,goal}` | ~7.5B | Validates the simulator, wrapper, and success detection independently of LeRobot |

`HuggingFaceVLA/smolvla_libero` uses the input keys `observation.images.image`, `observation.images.image2` (wrist), and an 8-D `observation.state`, plus a 7-D action. **Our wrapper and datasets use exactly these keys**, so that checkpoint runs on our pipeline unchanged.

### 2.3 Libraries

- Python 3.10
- PyTorch with CUDA, bf16 mixed precision
- LeRobot (SmolVLA, dataset format, training loop)
- Hugging Face Transformers, PEFT (LoRA), Accelerate
- LIBERO, robosuite (version pinned by LIBERO), MuJoCo
- NumPy, SciPy, OpenCV, h5py
- Weights & Biases for logging
- Open3D (optional, point cloud visualization during debugging)
- imageio + ffmpeg for rollout videos

Pin every version in `requirements.txt` once Part A works, and never upgrade mid-project. Versions verified on the local machine (Oct 30): Python 3.10, torch 2.8.0 (cu126), lerobot 0.4.4, transformers 4.57.6, hf-libero 0.1.4, robosuite 1.4.0, mujoco 3.3.2, numpy 2.2.6.

**Known compatibility issues:**
- **robosuite 1.4.0 + NumPy 2:** instance segmentation crashes (`OverflowError` in `read_pixels`). `groundedvla/envs/compat.py` patches this at import time, both locally and on Colab. Never edit site-packages by hand for this.
- **OpenVLA needs its own environment:** it requires `transformers==4.40.1`, `timm==0.9.10`, and `tokenizers==0.19.1`, which conflict with LeRobot. Locally that means a separate conda env, `D:\groundedvla\envs\gvla-openvla`. On Colab it means a separate runtime (`notebooks/01b_openvla_pipeline_check.ipynb`). Our code imports LeRobot and Transformers lazily, so each policy only needs its own libraries.

### 2.4 Compute

| Machine | GPU | Used for |
|---|---|---|
| Colab A100 (paid, ~$50 budget) **or** USC CARC (A100/A40) | 40–80 GB | All training (B1, B3) and main evaluation (B4, B5). bf16. |
| Colab T4 (free) | 16 GB | Part A debugging, short tests. No bf16 (use fp16/fp32). OpenVLA-7B needs 4-bit or 8-bit loading here. |
| Local laptop | RTX 3050, 4 GB | `run_sim` viewer, demo replay, SmolVLA rollouts, unit tests. **Cannot run OpenVLA-7B.** |

- **Rendering:** headless on Colab/CARC with `MUJOCO_GL=egl` and `PYOPENGL_PLATFORM=egl`. Locally, `MUJOCO_GL=glfw`.
- **CARC (optional):** jobs run as Slurm batch scripts (`scripts/slurm/`), so the laptop doesn't need to stay on. Check GPU queue wait times and job time limits, and confirm the course or advisor allocation first.

### 2.6 Colab + GitHub + Google Drive Workflow

1. Push code to the **public** GitHub repo (no token needed).
2. Each notebook's first cells: mount Google Drive, `git clone` (or `git pull`) the repo into `/content/GroundedVLA`, `pip install -e .`, and **log the commit hash** for the run.
3. **Google Drive (USC, ~80 GB free)** stores everything persistent:
   ```
   /content/drive/MyDrive/groundedvla/
     models/        # smolvla_base, smolvla_libero, openvla-7b-finetuned-libero-*, our fine-tuned checkpoints
     datasets/      # LIBERO raw demos, LeRobot datasets (one .tar per dataset)
     checkpoints/   # training runs
     outputs/       # eval JSONL, videos, plots, results tables/sheets
   ```
4. **Never train or evaluate straight off Drive.** Reading thousands of small files from Drive is very slow. At session start, copy each dataset archive and the model weights to local disk (`/content/local/...`), extract them there, and read from local disk. Write checkpoints and results back to Drive.
5. Paths come from `configs/paths/colab.yaml` (Colab) or `configs/paths/local.yaml` (laptop). Code never hard-codes a path.
6. Save the notebook after each run so its outputs are visible in VS Code.

**Storage budget on Drive (fits easily in ~80 GB):**

| Item | Size |
|---|---|
| OpenVLA-7B LIBERO checkpoints (spatial + goal) | ~15 GB each, ~30 GB total (download only what Step A3 needs) |
| smolvla_base, smolvla_libero | ~1 GB each |
| Our fine-tuned checkpoints (V1–V3, latest + best only) | ~1–2 GB each, ~6–10 GB |
| LIBERO raw demos (T1–T4) | ~3 GB |
| Replayed LeRobot datasets (RGB + mask as video, 16-D grounding vector) | a few GB |
| Videos and results | a few GB |

**Storage rule (Step A5):** do not store raw depth or segmentation for every step. Compute the 16-D grounding vector and the V2 mask image during the replay, and store only those plus RGB, state, and actions. Keep full depth and segmentation for a small debug subset (5 demos per task) for `check_grounding.py`.

### 2.7 Checkpointing and Resuming (training and evaluation)

Colab sessions end without warning (idle timeouts, disconnects, session caps), even on paid A100s. Everything must resume.

- **Training:** every 2,000 steps, save model, optimizer, LR scheduler, AMP scaler, RNG states, global step, W&B run ID, and commit hash to `checkpoints/<run_name>/step_XXXXXX/`.
  - Write to a temporary folder first, then rename it, so a disconnect mid-save can't corrupt the latest checkpoint.
  - Keep only the 2 most recent checkpoints plus the selection checkpoints (10k, 15k, 20k) and the best one, to save Drive space.
  - On start, the notebook automatically resumes from the newest complete checkpoint and continues the same W&B run.
- **Evaluation:** append one JSON line per finished episode to `outputs/eval/<run>/<condition>.jsonl`. A restarted evaluation skips episodes that are already logged, so no rollout is lost or repeated.
- **Datasets (A5):** replay writes one shard per task and demo, so a crashed replay restarts where it stopped.

### 2.5 Storage Rules: D Drive Only (Do Not Touch C Drive)

**Rule:** everything for this project lives under `D:\groundedvla`. Nothing is installed, downloaded, cached, or written to `C:\`, including the SmolVLA model, datasets, Python environments, package caches, logs, and temp files.

On Colab, none of this applies, because Colab runs in the cloud and uses Google Drive. This section covers any work you do on your own computer (debugging, replay, small tests).

**Folder layout**

Exception: the git repo itself (code only, a few MB) lives in the VS Code workspace (`...\2-Robot_Manipulation_Final_Project\GroundedVLA`). Everything heavy lives on D.

```
D:\groundedvla\
  envs\           # Python environments (gvla; gvla-openvla only if OpenVLA is ever run locally)
  models\         # SmolVLA checkpoint
  datasets\       # LIBERO demos, replayed data, LeRobot datasets
  checkpoints\    # training outputs
  outputs\        # eval logs, videos, plots
  cache\          # Hugging Face, pip, conda, torch, wandb, temp
```

**Step 1: Install tools on D**
- Install Miniconda to `D:\miniconda3` (the default location is on C).
- Create the environment on D: `conda create --prefix D:\groundedvla\envs\gvla python=3.10`
- Always install packages with the cache on D: `pip install --cache-dir D:\groundedvla\cache\pip <package>`

**Step 2: Set cache locations before installing or downloading anything**

Run these once in PowerShell, then restart the terminal:

```
setx HF_HOME "D:\groundedvla\cache\huggingface"
setx HF_LEROBOT_HOME "D:\groundedvla\cache\lerobot"
setx TORCH_HOME "D:\groundedvla\cache\torch"
setx PIP_CACHE_DIR "D:\groundedvla\cache\pip"
setx CONDA_PKGS_DIRS "D:\groundedvla\cache\conda_pkgs"
setx WANDB_DIR "D:\groundedvla\outputs\wandb"
setx WANDB_CACHE_DIR "D:\groundedvla\cache\wandb"
setx WANDB_CONFIG_DIR "D:\groundedvla\cache\wandb_config"
setx LIBERO_CONFIG_PATH "D:\groundedvla\cache\libero_config"
```

Temp files default to C. Redirect them per session by running this at the start of each terminal (put it in `D:\groundedvla\activate.ps1`):

```
$env:TEMP = "D:\groundedvla\cache\tmp"
$env:TMP  = "D:\groundedvla\cache\tmp"
```

Create the `tmp` folder first. Confirm the exact `HF_LEROBOT_HOME` variable name in your installed LeRobot version.

**Step 3: Download the SmolVLA model to D**

```
huggingface-cli download lerobot/smolvla_base --local-dir D:\groundedvla\models\smolvla_base
```

Load it from that path in code instead of the repo name, so it never re-downloads to a default cache.

**Step 4: Download LIBERO data to D**
- After installing LIBERO, open the config file in `D:\groundedvla\cache\libero_config\config.yaml` and set the dataset path to `D:\groundedvla\datasets\libero`.
- Then run LIBERO's dataset download script. Replayed demos with depth and segmentation (Step A5) and the converted LeRobot dataset also go in `D:\groundedvla\datasets`.

**Step 5: Point all outputs to D**
- Training: set the output directory to `D:\groundedvla\checkpoints\<run_name>` and the dataset root to `D:\groundedvla\datasets\<name>`.
- Evaluation videos, logs, and plots: write to `D:\groundedvla\outputs`.
- In code, read every path from the config files, never hard-code a path or leave a default.

**Things that quietly use C drive**
- **WSL2:** its virtual disk is stored on C by default. If you use WSL2 (needed for headless MuJoCo rendering, since EGL does not work natively on Windows), move the distro to D before installing anything: `wsl --export Ubuntu D:\wsl\ubuntu.tar`, then `wsl --unregister Ubuntu`, then `wsl --import Ubuntu D:\wsl\ubuntu D:\wsl\ubuntu.tar`. Keep the project inside that distro.
- **Docker Desktop:** also stores images on C by default. Move its disk image location to D in settings, or avoid Docker.
- **Python installed from the Microsoft Store or python.org:** installs to C. Use the Miniconda on D instead.
- **VS Code extensions, Jupyter checkpoints, and pip's default cache:** all default to C. Set the cache variables above (`run.bat` and `activate.ps1` do this) and select `D:\groundedvla\envs\gvla\python.exe` as the VS Code interpreter.

**Check that it worked**
- Before starting, note free space on C with `Get-PSDrive C`.
- After installing the environment, downloading SmolVLA, and downloading LIBERO, run it again. Free space on C should be unchanged (within a few MB of logs).
- If it dropped by hundreds of MB or more, search for the culprit with a disk usage tool such as WinDirStat or TreeSize Free, then fix that variable.

---


## 3. Tasks and Data

### 3.1 Selected Tasks

Three tasks, same robot and action space, trained together as one multi-task policy:

- **T1 (LIBERO-Spatial):** "pick up the black bowl between the plate and the ramekin and place it on the plate"
- **T2 (LIBERO-Spatial):** "pick up the black bowl next to the cookie box and place it on the plate"
- **T3 (LIBERO-Goal):** "put the bowl on the stove"
- **T4 (optional, LIBERO-Goal):** "put the bowl on the plate". Same scene as T3 with a different destination, so it tests destination grounding directly.

Why these:
- T1 and T2 need the spatial phrase to pick the right bowl out of two identical ones, which tests target grounding.
- Every LIBERO-Spatial task uses the plate as the destination, so T3 (and T4) add destination variety.
- Max episode length: 220 steps for T1 and T2, 300 for T3 and T4.

Copy the exact instruction strings from the BDDL files in Week 1 and confirm each task's goal is a single `On` (or `In`) predicate.

Each LIBERO task includes 50 expert demonstrations and a fixed set of 50 initial scene states used for evaluation.

### 3.2 Extracting Target and Destination

Each task's BDDL file defines its goal as a predicate such as `(On akita_black_bowl_1 plate_1)`.

- First argument = target object
- Second argument = destination

We parse this directly instead of parsing the instruction text. The policy itself still only sees the instruction; the mask and grounding inputs use the BDDL goal as the language-to-object mapping, consistent with the proposal's use of simulator-provided information.

Important consequence: V2 and V3 are told which object matters by the simulator, not by understanding the instruction. That is exactly why V2 is the right control for V3: both get the same object identity, so V3 vs V2 isolates the value of metric 3D. State this clearly in the report.

### 3.3 Splits and Split Manifest

- **Training demos:** 45 of the 50 demos per task.
- **Validation demos:** the other 5 per task, for validation loss.
- **Checkpoint selection:** 10 closed-loop rollouts per task on validation initial states, sampled with a separate seed from the test states.
- **In-distribution test:** LIBERO's 50 fixed initial states per task.
- **Out-of-distribution test:** defined in Section 6. Never used for training or checkpoint selection.

Save everything in `data/split_manifest.json`: demo IDs per split, validation and test seeds, OOD condition settings, and dropped replays. All three variants read the same manifest.

### 3.4 Fine-Tuning Dataset

All three variants are fine-tuned on the **same dataset**: LIBERO's human demonstrations for T1–T3 (T4 if used), 50 per task. They are re-rendered in Step A5 at 256 × 256 with depth and segmentation, failed replays and no-op steps are removed, and the result is converted to LeRobot format.

V1 ignores the extra fields, V2 adds the mask image, and V3 adds the mask image and the grounding vector. We do **not** train on Hugging Face's pre-converted `HuggingFaceVLA/libero` dataset. It has no depth or segmentation, and the variants would then train on different data.

---

## 4. Repository Structure

```
GroundedVLA/                  # git repo root (public GitHub)
  README.md                   # setup, run commands, obs/action dimensions
  pyproject.toml              # `pip install -e .`
  requirements.txt            # pinned main env (LeRobot, LIBERO, ...)
  requirements-openvla.txt    # pinned OpenVLA env (separate, see 2.3)
  run.bat                     # local shortcut: `.\run <model> [options]`
  docs/
    GroundedVLA_Implementation_Plan.md
  configs/
    paths/local.yaml          # D:\groundedvla paths
    paths/colab.yaml          # Google Drive + /content/local paths
    models.yaml               # registry of model names used by run_sim / eval
    v1_rgb.yaml
    v2_mask.yaml
    v3_grounded.yaml
    eval_ood.yaml
  scripts/
    run_sim.py                # run any model in LIBERO, live MuJoCo viewer or saved video
    patch_windows.py          # one-time local robosuite fixes (Windows only)
    download_all.py           # models + LIBERO data to configured paths
    slurm/                    # CARC batch scripts (optional)
  notebooks/
    00_colab_setup.ipynb      # Drive mount, clone, install, A1 checks
    01_pipeline_check.ipynb   # A3: smolvla_base / smolvla_libero rollouts + videos
    01b_openvla_pipeline_check.ipynb  # A3: OpenVLA-7B in its own runtime
    02_build_dataset.ipynb    # A5 dataset build, A6/A7 checks
    03_train.ipynb            # B1/B3 training with auto-resume (added in Part B)
    04_eval.ipynb             # B4/B5 evaluation (added in Part B)
  groundedvla/                # python package
    tasks.py                  # T1-T4 specs
    utils/paths.py            # loads configs/paths/*.yaml
    policies/                 # one class per model behind a common reset()/act() interface
      smolvla.py              # smolvla_base, smolvla_libero, our V1-V3 checkpoints
      openvla.py              # OpenVLA-7B LIBERO checkpoints
      replay.py               # expert demo replay (A2 check)
      chunking.py             # naive chunking + temporal ensemble
    viz/viewer.py             # MuJoCo passive viewer + video writer
    envs/
      compat.py               # robosuite/NumPy 2 patch, MUJOCO_GL setup
      libero_wrapper.py       # obs/action mapping, image flip, camera perturbation
      camera_utils.py         # intrinsics, extrinsics, depth conversion
    data/
      replay_demos.py         # re-render demos with depth + segmentation
      convert_to_lerobot.py   # write LeRobot dataset format
      make_split_manifest.py  # train/val demos per task
      check_dataset.py        # random frames + real SmolVLA loss on our data
      bddl_parser.py          # extract target and destination from goal
      split_manifest.json
    grounding/
      masks.py                # target/destination mask image for V2 and V3
      grounding.py            # mask -> point cloud -> robot-frame centroid
      check_grounding.py      # compare against ground truth
    models/
      grounding_adapter.py    # Fourier features + MLP -> tokens
      grounded_smolvla.py     # SmolVLA subclass that injects grounding tokens
    train/
      train.py
    eval/
      rollout.py
      metrics.py
      monitor.py              # per-step signals: lifts, displacement, joint limits, final contacts
      collisions.py           # contact-based collision counting
      failure_labels.py       # automatic failure categories
      run_eval.py
      report.py               # result tables, plots, confidence intervals
    ood/
      make_ood_states.py      # shifted placements, layout swaps, distractors
      paraphrases.json        # unseen instruction paraphrases
```

---

## 5. Part A: Testing the Pretrained SmolVLA and Building the Pipeline

Goal: every component works end to end with the pretrained model before any fine-tuning. No training happens in Part A.

Part A steps A1 to A4 happen in Week 1 (Nov 1–7). A5 and A6 run in Week 2 alongside baseline training, and A7 in Week 3 (see Section 8). Fine-tuning still only starts once the steps it depends on are done (see Part B).

**Local check tool (`scripts/run_sim.py`, shortcut `run.bat`).** Runs any registered model in LIBERO episode after episode, either in a live MuJoCo viewer window (laptop) or headless with saved videos (Colab/CARC):

```
.\run replay          --task T1 --episodes 3     # expert demo replay (A2 check)
.\run smolvla_base    --task T1 --episodes 3     # zero-shot plumbing test (A3)
.\run smolvla_libero  --task T2 --episodes 5     # already fine-tuned SmolVLA (A3)
.\run openvla_7b      --task T1 --viewer none --save-video   # Colab A100 only (A3)
.\run smolvla_ckpt    --checkpoint <path> --task T3          # our V1-V3 checkpoints (Part B)
```

### Step A1: Environment Setup

Tasks:
- If working on your own computer, complete Section 2.5 first (D drive folders, environment variables, SmolVLA download). Do not install anything before that.
- Install LIBERO, robosuite, MuJoCo, LeRobot, Weights & Biases, and FFmpeg on Colab.
- Confirm GPU type and memory, Drive storage, and session time limits.
- Set EGL environment variables before importing MuJoCo.
- Load T1, reset it, and step it with zero actions.
- Save frames from both cameras.
- Measure training throughput (steps per second at batch size 32) with a short dummy run, and use it to check the training budget in Step B1.

Notes:
- LIBERO images come out upside down. Flip RGB, depth, and segmentation by 180 degrees consistently in `libero_wrapper.py`.
- Import `groundedvla.envs.compat` before creating any LIBERO env. It applies the NumPy 2 segmentation fix and sets `MUJOCO_GL` (EGL on Linux, GLFW on Windows).

Done when:
- Both camera frames save correctly and appear upright.
- RGB, depth, and instance segmentation all render (segmentation needs the compat patch).
- One full episode runs headless without crashing.
- Installation steps are written in the README.

### Step A2: Observation and Action Mapping

Tasks:
- Write `libero_wrapper.py` to convert LIBERO observations into SmolVLA inputs:
  - `observation.images.image` ← agentview RGB
  - `observation.images.image2` ← wrist RGB (Hugging Face LIBERO naming, so `HuggingFaceVLA/smolvla_libero` runs unchanged)
  - `observation.state` ← 8-D robot state (eef pos 3, eef axis-angle 3, gripper finger qpos 2; same as LeRobot's `LiberoProcessorStep`)
  - `task` ← instruction string
- Map SmolVLA's output chunk to 7-D LIBERO actions.
- Check gripper sign convention: LIBERO uses -1 for open and +1 for close. Verify against a demo.
- Run 10 no-op steps at the start of each episode so objects settle, matching standard LIBERO evaluation practice.
- Document every observation and action dimension in the README.

Done when:
- Replaying a demo's recorded actions through the wrapper completes the task (`.\run replay`), and the replay video is saved.
- Unit test confirms input shapes and value ranges match what SmolVLA expects.

### Step A3: Pretrained Model Rollouts (Pipeline Check)

Three models, none fine-tuned by us (Section 2.2.1). Together they separate "our pipeline is broken" from "the model is weak":

1. **`smolvla_base` (plumbing test):** 10 rollouts per task. Expect near-zero success, since it was pretrained on a different robot, camera setup, and action space.
2. **`smolvla_libero` (`HuggingFaceVLA/smolvla_libero`):** 10 rollouts per task. It is already fine-tuned on LIBERO, so high success here confirms our image flip, state format, key names, normalization, and gripper sign.
3. **`openvla_7b` (`openvla-7b-finetuned-libero-spatial` for T1/T2, `-goal` for T3/T4):** 10 rollouts per task on a Colab A100, in its own runtime. It confirms the simulator, initial states, and success detection without using LeRobot at all.

Use the episode lengths from Section 3.1 and save videos for every rollout.

Verified locally (Oct 30):
- `smolvla_libero` must run with horizon 1 (its config has `n_action_steps: 1`). At N = 10 it went 0/3 on T1: the arm went straight to the plate without grasping the bowl.
- At N = 1 our pipeline matched LeRobot's official `lerobot-eval` episode for episode on T1 (initial states 0 and 1: success, then fail in both). This confirms our observation/action mapping is equivalent to LeRobot's.
- `smolvla_base` moves the arm (not frozen or saturated) with 0% success, as expected.
- The checkpoint is only moderate on T1 (1/2 in both pipelines). Use 10 episodes per task on the A100 before drawing conclusions.

Done when:
- Rollouts complete without errors for all three models.
- `smolvla_base`: robot moves are within valid ranges, not saturated or frozen, and the arm visibly responds to the model.
- `smolvla_libero` and `openvla_7b` succeed on most rollouts of T1–T3 (roughly their published LIBERO success). If they don't, fix the wrapper before anything else; a broken pipeline would make every later number meaningless.

### Step A4: Evaluation Harness

One harness and logger used by every variant.

Tasks:
- Write `rollout.py`, `metrics.py`, `collisions.py`, and `failure_labels.py` with the exact definitions below.
- Support a config option for execution mode: `naive` with horizon N (number of chunk actions executed before re-predicting; test N = 5, 10, and 25 in Step B1), or `temporal_ensemble` (predict every step, exponentially weighted average of overlapping chunks, m = 0.01).
- Log per-episode results as one JSON line per episode (resumable, Section 2.7) and to Weights & Biases.

Fields recorded per episode:
- **Identity:** task, instruction, policy variant, checkpoint, random seed, evaluation condition
- **Success:** task success, correct object, grasp success, correct destination
- **Accuracy:** final object-to-target position error
- **Safety:** number of unintended collisions
- **Efficiency:** episode steps, completion time, inference latency
- **Diagnosis:** failure category and rollout video path

Metric definitions:
- **Task success:** BDDL goal predicate is true before the episode time limit.
- **Correct object selection:** the first object lifted is the target. "Lifted" means the object is in contact with both gripper fingers and its height is at least 3 cm above its starting height.
- **Grasp success:** the target object is lifted at any point in the episode.
- **Correct-destination placement:** at episode end, the target satisfies the goal predicate with the correct destination.
- **Final position error:** horizontal distance (cm) between the target's final center and the destination's center.
- **Unintended collisions:** number of new contacts (contact pairs not present in the previous control step, normal force above 1 N) between any robot part and any object other than the target, plus contacts between arm links (not fingers) and the table. Report mean per episode and the percentage of episodes with at least one.
- **Episode steps and completion time:** control steps until success or timeout, and simulated time (steps divided by 20 Hz).
- **Inference latency:** time per policy call, measured with `torch.cuda.synchronize()` before and after. Grounding computation time is measured separately. Report median and 95th percentile.

Failure categories (each failed episode gets the first label that matches, in this order):
1. **Wrong object:** the first lifted object is not the target.
2. **Collision:** the robot moved a non-target object by more than 5 cm, or knocked the target over before grasping it.
3. **Failed grasp:** the target was never lifted.
4. **Dropped object:** the target was lifted, then ended on the table outside the destination.
5. **Wrong destination:** the target ended resting on or in an object other than the destination.
6. **Motion failure:** the end effector moved less than 1 cm over the last 50 steps before timeout, or the arm hit a joint limit.
7. **Timeout:** any other failure.

Check the automatic labels by watching 10 random failure videos per variant.

Statistics:
- Report success rates with 95% Wilson confidence intervals.
- Report position error as mean and standard deviation.

Done when:
- Running the harness on 10 expert demo replays reports 100% success, no failure labels, and correct values for every metric.
- The path from task reset to observation, action, completed rollout, and saved metrics works end to end.

### Step A5: Demo Replay with Depth and Segmentation

Tasks:
- LIBERO's original demos store RGB only. For each demo:
  1. Reset to the demo's initial state.
  2. Replay its recorded actions.
  3. Save at every step: agentview and wrist RGB, robot state, action, task ID, episode ID, the target and destination object IDs, the V2 mask image, and the 16-D grounding vector (computed on the fly from depth and segmentation).
  4. Save full depth and instance segmentation only for a debug subset (5 demos per task), per the storage rule in Section 2.6.
- Save the mapping from segmentation instance IDs to BDDL object names (for example `akita_black_bowl_1`) with the dataset. Masks and grounding depend on it, and robosuite's instance IDs aren't the same as MuJoCo body or geom IDs.
- Convert raw MuJoCo depth (normalized values) to metric depth in meters with robosuite's depth conversion utility.
- Remove no-op steps where all position and rotation actions are near zero and the gripper command doesn't change.
- Discard any replay that doesn't succeed (small physics differences can cause drift) and log how many were dropped.
- Write the train and validation split into `split_manifest.json` (Section 3.3).
- Convert the results to LeRobot dataset format.

Done when:
- At least 90% of demos per task replay successfully.
- Dataset loads in LeRobot and a random batch visualizes correctly.
- Image preprocessing, tokenization, state format, action normalization, and chunk dimensions are checked against SmolVLA's expectations.

### Step A6: Masks and 3D Grounding Module

**Mask image (for V2 and V3), `masks.py`:**
- Build a 256 x 256 x 3 image from the agentview segmentation: red channel = target mask, green channel = destination mask, blue = 0.
- Fed to SmolVLA as a third image input, `observation.images.mask`.

**3D grounding (for V3), `grounding.py`.** For each timestep:

1. Get target and destination object IDs from the BDDL goal (Section 3.2).
2. Select the target's pixels from the agentview instance segmentation.
3. Back-project those pixels to 3D camera coordinates using metric depth and the camera intrinsic matrix.
4. Transform the points to world coordinates with the camera extrinsic matrix, then to the robot base frame using the base pose.
   - **Frame conventions:** MuJoCo cameras look along −z with +y up (OpenGL). robosuite's `get_camera_extrinsic_matrix` already converts to the OpenCV convention (+z forward, +y down), which is what our pinhole back-projection assumes. Raw obs arrays have row 0 at the bottom, so back-project on `raw[::-1]` (vertical flip only), checked by projecting ground-truth positions onto the masks. Policy images use the 180 degree flip `raw[::-1, ::-1]`, which is that view mirrored left-right.
5. Take the midpoint of the 5th and 95th percentile of the points on each axis as the centroid (a robust bounding-box center). Changed Nov 1 from the plain median: the camera sees one side of an object, which pulled the median about 2.4 cm off a bowl's center, while the box center stays within about 0.5 cm.
6. Repeat for the destination.
7. If the target has fewer than 20 visible pixels (occluded or inside the gripper), reuse the last valid centroid and set a visibility flag to 0.

Grounding vector (16 values):
- Target centroid xyz (3)
- Destination centroid xyz (3)
- End-effector position xyz (3)
- Target minus end-effector (3)
- Destination minus target (3)
- Target visibility flag (1)

The first six values are the core target and destination coordinates. The rest are cheap extras that make relative motion easier to learn.

Normalization:
- Subtract the workspace center and divide by the workspace half-extent so every coordinate falls roughly in [-1, 1]. Compute these values once from the training data and save them with the checkpoint.

Validation (`check_grounding.py`):
- Compare centroids against the simulator's ground-truth object positions across all training data. Ground truth is the center of the object's collision geometry, not its body origin (a bowl's origin is at its bottom, about 3 cm below its center).
- Project ground-truth positions back into the image and overlay masks, centroids, and coordinate axes on frames to catch frame-convention bugs.

Done when:
- Median centroid error is under 2 cm for visible objects.
- 95th percentile error is under 4 cm. (Some offset is expected because the visible surface center differs from the object's body center.)
- Mask images line up with the objects in the RGB frames.

### Step A7: Camera Perturbation Support

Tasks:
- Add a function in `libero_wrapper.py` to change agentview camera position, orientation, and field of view at reset.
- Make sure the mask and grounding modules read the updated extrinsics and intrinsics after each change.

Done when:
- Under a 20 degree yaw rotation, centroid error stays under 2 cm (confirms grounding is camera-invariant).
- Images visibly change while robot-frame centroids do not.

---

## 6. Out-of-Distribution Evaluation Conditions

All conditions apply only at evaluation. Training always uses the original LIBERO demos. Initial states for every condition are generated once by `make_ood_states.py` with fixed seeds and saved in the split manifest, so all three variants face identical scenes.

### 6.1 Position OOD

Generated by editing the object placement regions in each task's BDDL file:

- **Small shift:** every object's region shifted by 5 cm in a random horizontal direction.
- **Large shift:** regions shifted by 10 to 15 cm.

Any initial state where objects overlap or fall off the table is resampled.

### 6.2 Scene OOD

- **Layout swap (T1, T2):** distractor and target positions swapped while keeping the instruction true (for example, the bowl "next to the cookie box" is still the correct one, but now in a different location).
- **Extra distractor:** one additional object from LIBERO's existing asset library placed in the scene, away from the target and destination.

### 6.3 Viewpoint OOD

Applied to the agentview camera only:

- **Yaw rotation around the workspace:** +10, +20, and -20 degrees
- **Height change:** +10 cm
- **Field of view change:** +10%

### 6.4 Language OOD

- 3 hand-written paraphrases per task with the same meaning, for example "grab the black bowl sitting between the plate and the ramekin and put it onto the plate."
- Write and save them in `paraphrases.json` before any evaluation, and never use them in training.
- Caveat: V2 and V3 get the target from the BDDL goal, so they are partly protected from paraphrases (Section 3.2). Report this condition separately, and read it mainly as "does the model still follow the instruction when worded differently," not as evidence for metric 3D.

### 6.5 Rollout Budget

- **Tier 1, 50 rollouts per task:** in-distribution, large shift, yaw ±20 degrees.
- **Tier 2, 25 rollouts per task:** every other condition.
- If time runs short, drop Tier 2 conditions or T4 before reducing Tier 1 rollouts.

### 6.6 Generalization Gap

For each variant and condition:

Degradation = in-distribution success rate minus out-of-distribution success rate.

---

## 7. Part B: Fine-Tuning

Step B1 starts once A1 to A5 pass their "done when" checks. Steps B2 and B3 start once A6 and A7 also pass.

**Where training runs:** `notebooks/02_train.ipynb` on a Colab A100, opened through the Colab VS Code extension. The notebook clones the repo from GitHub, copies the dataset and weights from Drive to local disk, trains with auto-resume (Section 2.7), and writes checkpoints back to Drive. On CARC the same `groundedvla/train/train.py` runs from a Slurm script. The notebook is only a thin launcher, so both paths run identical code.

### Step B1: Fine-Tune the RGB Baseline (V1)

Starting hyperparameters:
- Starting checkpoint: `lerobot/smolvla_base`
- Trainable parameters: action expert and state projection (LeRobot defaults). Backbone frozen.
- Optimizer: AdamW, learning rate 1e-4, weight decay 1e-10, cosine decay with 1,000 warmup steps
- Batch size: 32 (raise to 64 if memory allows on the A100)
- Training steps: 20,000 for all selected tasks together (adjust using the throughput measured in Step A1)
- Checkpoints evaluated for selection: 10k, 15k, and 20k steps (Section 3.3)
- Precision: bf16
- Chunk size: 50
- Image size: SmolVLA default preprocessing from the 256 x 256 inputs

Track:
- Training and validation loss, learning rate, GPU memory, and training time.

Then:
- Tune execution horizon N (5, 10, 25) on the validation rollouts and fix the best N for every experiment afterward.
- Compare naive chunking (best N) against temporal ensemble on the same validation rollouts. Use whichever is better for all later experiments, and report both (this ties the project to the Action Chunking material from class).
- Run 20 to 30 in-distribution test rollouts per task, save success and failure videos, and build the first results table with the most common failure categories.

Done when:
- V1 completes full tasks using only its own predicted actions, with in-distribution success of at least 60% averaged over the tasks. Below that, debug data and action mapping before Step B3.
- Training curves and rollout videos look reasonable.

Fallback:
- If fine-tuning isn't working by Wednesday of Week 2 (Nov 11), start all three variants from `HuggingFaceVLA/smolvla_libero` (already fine-tuned on LIBERO, verified in Step A3) instead of `smolvla_base`, and narrow the work to lightweight adaptation and evaluation. All variants must use the same starting checkpoint.

### Step B2: Build the Mask Input and Grounding Adapter

**V2 mask input:**
- Add `observation.images.mask` (Step A6) as a third camera input in the SmolVLA config. No other model code changes.

**V3 grounding adapter (`grounding_adapter.py`):**
1. Fourier features: each of the 15 coordinate values is mapped to sin and cos at 8 frequencies (2^0 through 2^7 times π). Gives 15 × 16 = 240 values.
2. Concatenate the 240 Fourier values, the 15 raw values, and the visibility flag: 256 inputs.
3. MLP: 256 → 512 → 512 → (K × hidden size), with GELU activations and LayerNorm.
4. Reshape into K = 2 tokens, each the size of SmolVLA's token embedding.
5. Initialize the final layer to zero, so at the start of training the adapter contributes nothing and the model behaves like V2. This prevents early training instability.

Injection (`grounded_smolvla.py`):
- Subclass the SmolVLA policy and override the function that builds the prefix tokens (image, language, and state tokens).
- Append the 2 grounding tokens immediately after the state token.
- Give the grounding tokens the same attention treatment as the state token so the action expert can attend to them.
- Check the method names in the installed LeRobot version, since internal function names may differ.

Fallbacks if token injection is unstable or too slow to build (decide by the end of Week 3, then freeze):
1. Append the normalized 16-D vector to `observation.state`, growing the state from 8 to 24 values. Only a config change, no model code.
2. Add the projected grounding embedding to every input token of the action expert, so it reaches the action prediction directly.

Done when:
- V2 trains and runs through the harness with the mask image visible in logged inputs.
- With adapter output forced to zero, V3 produces identical actions to V2 on the same inputs.
- One training batch runs forward and backward with gradients reaching the adapter.

### Step B3: Fine-Tune V2 and V3

- Exact same hyperparameters, data, manifest, steps, seeds, checkpoint selection, and execution horizon as Step B1.
- Trainable parameters: everything trained in B1, plus the grounding adapter for V3.
- Adapter learning rate: 1e-4 (same as the rest). If training is unstable, lower it to 5e-5.

Done when:
- V2 and V3 complete autonomous rollouts through the same harness as V1.
- V3 in-distribution success is at least as high as V1 (if it's lower, check grounding normalization and injection first).

### Step B4: Main Evaluation

For each variant (V1, V2, V3), every task, and every condition in Section 6, using the Tier 1 and Tier 2 rollout counts:

Fair comparison rules:
- Same initial states, seeds, episode limits, and success definition for every variant.
- Same harness and logger.
- Same execution mode (naive N or temporal ensemble, fixed in Step B1).

Seeds:
- One training seed per variant. If time allows, train a second seed for V1 and V3.

Outputs:
- Success rate with 95% confidence intervals for every condition
- Every metric from Step A4, including collisions and failure category counts
- Degradation from in-distribution to each OOD condition
- Latency for each variant, with grounding time shown separately
- Representative success and failure videos from identical test conditions

Questions the analysis must answer:
- Does explicit 3D grounding improve overall task success?
- Does it mainly help with position or viewpoint generalization?
- Is the target mask alone enough to get most of the improvement (V2 vs V3)?
- Does 3D grounding reduce placement error or specific failure categories?
- What latency or implementation cost does grounding add?
- Under which conditions does the RGB baseline do equally well or better?

A valid conclusion does not require V3 to win everywhere. The report should say where 3D helps, where it doesn't, and what limits come from using simulator-provided perception.

### Step B5: Extra Ablations

The V1/V2/V3 comparison is the main ablation. These extras reuse the trained V3 checkpoint, so they need no new training. Run them on in-distribution, large shift, and yaw ±20 degrees only:

1. **Noisy grounding:** add Gaussian noise (standard deviation 1, 3, and 5 cm) to centroids at evaluation time. Shows how accurate grounding needs to be.
2. **Ground-truth positions:** replace computed centroids with simulator ground truth. Upper bound on what perfect grounding gives.

Stretch, only if Week 4 has spare time:
3. **No language conditioning:** replace the target and destination vector with centroids of all objects in a fixed order, so the model gets 3D positions but isn't told which object matters. Needs its own training run.

---

## 8. Timeline (5 Weeks)

Revised Oct 30: the project starts Nov 1 and is presented on **Dec 3**.

- **Week 1 (Nov 1–7): Infrastructure.** Steps A1 to A4. Colab setup notebook, `run_sim` tool, expert replay video, pipeline check with `smolvla_base`, `smolvla_libero`, and OpenVLA-7B, harness and logger.
  - Exit: task reset to observation, action, completed rollout, and saved metrics works end to end, and `smolvla_libero` / OpenVLA succeed on T1–T3.
- **Week 2 (Nov 8–14): RGB baseline.** Step A5 (dataset and split manifest), then B1 (V1 training, horizon tuning, naive vs temporal ensemble, first results). In parallel, Member 2 builds and validates A6.
  - Exit: V1 completes full tasks on its own. Fallback checkpoint decision by Wednesday, Nov 11.
- **Week 3 (Nov 15–21): Grounding.** A7, B2, B3 (train V2 and V3). Generate all OOD initial states and paraphrases (Section 6).
  - Exit: V2 and V3 run through the same harness. **Architecture frozen (Nov 21).**
- **Week 4 (Nov 22–28): Experiments.** B4 main evaluation, B5 extra ablations, failure labels, videos. Thanksgiving is Nov 26: queue long evaluation runs before it.
  - Exit: all results and failure labels complete. **Results frozen (Nov 28).**
- **Week 5 (Nov 29–Dec 3): Analysis and deliverables.** Check for missing runs, corrupted logs, inconsistent episode limits, wrong success detection, and test data leaking into training. Rerun only what's needed to confirm unexpected results. Write report, presentation, demo video, and clean up the repo.
  - Exit: everything in Section 10 complete. **Present Dec 3.**

---

## 9. Work Split

Both members review each other's code before merging.

**Member 1: RGB baseline and mask control**
- Steps A2, A3, A5
- V1 training and horizon tuning (B1)
- V2 mask input and training (B2, B3)
- Language OOD paraphrases (6.4) and latency measurements

**Member 2: Depth, 3D grounding, and spatial evaluation**
- Steps A6, A7
- Grounding adapter and V3 training (B2, B3)
- Position, scene, and viewpoint OOD conditions (6.1 to 6.3)
- Extra ablations (B5)

**Shared**
- Steps A1 and A4 (setup, harness, logger)
- Main evaluation runs (B4), failure labeling, rollout debugging
- Analysis, report, presentation, and demo video

---

## 10. Deliverables

- **Report:** introduction, related work, method, setup, results, ablations, failure analysis, limitations, and conclusion
- **Presentation:** problem, the three variants, experiments, results, limitations, and conclusion
- **Demo video:** the environment, all three variants on the same scene, an OOD example (large shift or 20 degree camera rotation), a failure case, and the final comparison
- **Code repository:** README with setup, training, and evaluation commands, configs, diagrams, pinned requirements, Colab notebook, and acknowledgments of reused datasets, code, and pretrained models
- **Artifacts:** final checkpoints for V1, V2, V3, logs, results tables, plots (success by condition, degradation, failure categories, latency, naive chunking vs temporal ensemble), and the split manifest
- **Pipeline-check results:** `smolvla_base`, `smolvla_libero`, and OpenVLA-7B success on T1–T3 (Step A3), showing the pipeline is correct

### Final Checklist

Use during the internal review before submission.

- ☐ LIBERO setup is reproducible from the README.
- ☐ All selected tasks use the same robot and action space.
- ☐ V1, V2, and V3 use the same demonstrations and split manifest.
- ☐ Masks and depth are documented as simulator-provided, and the oracle target identity is stated as a limitation.
- ☐ 3D coordinates are verified against simulator ground truth.
- ☐ The 3D module is lightweight and its injection point is documented.
- ☐ V1, V2, and V3 are all evaluated.
- ☐ Matched seeds and identical initial states are used for every comparison.
- ☐ All proposal metrics, collisions, and failure categories are reported.
- ☐ Failure cases are shown, not only successful videos.
- ☐ In-distribution and OOD results are clearly separated.
- ☐ Inference latency and the cost of grounding are discussed.
- ☐ The demo video includes both successes and failures.

---

## 11. Risks and Mitigations

- **Colab disconnects or A100 unavailable:** checkpoint and resume per Section 2.7, fall back to USC CARC, and keep T1 and T2 as the minimum viable scope.
- **Slow training or evaluation from Google Drive:** always copy datasets and weights to the Colab local disk first (Section 2.6).
- **Drive fills up:** keep only the latest and selection checkpoints per run, don't store per-step depth and segmentation, and download OpenVLA checkpoints only for the suites in use.
- **Library conflicts:** OpenVLA lives in its own environment or runtime (Section 2.3). The robosuite/NumPy 2 segmentation bug is patched in `groundedvla/envs/compat.py`.
- **Local GPU too small (RTX 3050, 4 GB):** use the laptop only for viewing, replay, and SmolVLA checks. Everything else runs on the A100 or CARC.
- **EGL rendering fails on Colab:** solve in Week 1 before anything else; fall back to OSMesa (slower) if needed.
- **Demo replays drift and fail:** drop failed replays and report the count; if more than 10% fail in a task, check that initial states are restored exactly.
- **Camera frame and image flip bugs:** catch with the ground-truth overlay check in Step A6.
- **Baseline underperforms:** check action normalization, gripper sign, and execution horizon before blaming the model. If still stuck by Wednesday of Week 2, use the fallback in Step B1.
- **Token injection unstable or hard to build:** switch to a fallback in Step B2 before the Week 3 architecture freeze.
- **V2 shows no change from V1:** the frozen vision encoder may not read the mask image well. Report it as found; optionally try overlaying a tinted mask on the agentview image instead, applied before the Week 3 freeze.
- **V3 ignores the adapter:** check adapter gradient norms; if near zero, try a higher adapter learning rate or more grounding tokens (K = 4).
- **V3 matches V1 everywhere:** still a valid result. V2 and the extra ablations explain why, and noisy and ground-truth grounding show where the limits are.
- **Evaluation takes too long:** measure time per episode in Week 2, then use the Tier 2 cuts in Section 6.5.
- **C drive fills up:** follow Section 2.5, set cache variables before any install, and compare free space on C before and after each major download.
- **Time overrun:** drop T4, then the stretch ablation, then Tier 2 conditions. Keep V1, V3, matched evaluation, and the proposal metrics.
