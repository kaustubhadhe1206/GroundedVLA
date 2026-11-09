"""Checkpoint bookkeeping in groundedvla/train/train.py (plan Section 2.7)."""
from groundedvla.train.train import TRAINING_STATE, latest_resumable, prune_checkpoints, split_episodes


def make_ckpts(run_dir, steps):
    for s in steps:
        d = run_dir / f"step_{s:06d}"
        d.mkdir(parents=True)
        (d / "model.safetensors").write_text("w")
        (d / TRAINING_STATE).write_text("s")


def test_prune_keeps_latest_selection_and_best(tmp_path):
    make_ckpts(tmp_path, range(2000, 20001, 2000))
    prune_checkpoints(tmp_path, keep_steps={10000, 15000, 20000}, best_step=6000)
    left = {int(p.name.split("_")[1]): (p / TRAINING_STATE).exists() for p in tmp_path.glob("step_*")}
    # newest two keep their optimizer state; selection/best steps keep weights only
    assert left == {6000: False, 10000: False, 18000: True, 20000: True}
    assert all((p / "model.safetensors").exists() for p in tmp_path.glob("step_*"))


def test_resume_ignores_weights_only_and_temporary(tmp_path):
    make_ckpts(tmp_path, [2000, 4000])
    (tmp_path / "step_004000" / TRAINING_STATE).unlink()          # weights-only checkpoint
    (tmp_path / ".tmp_step_006000").mkdir()                        # interrupted save
    assert latest_resumable(tmp_path).name == "step_002000"


def test_split_episodes_are_disjoint(tmp_path):
    (tmp_path / "splits.json").write_text('{"T1": {"train": [0, 1], "val": [2]}, "T2": {"train": [3], "val": [4]}}')
    assert split_episodes(tmp_path, ["T1", "T2"]) == ([0, 1, 3], [2, 4])
