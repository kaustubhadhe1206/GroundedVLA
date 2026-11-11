"""How good are instruction-predicted masks compared to the simulator's? (improvement 5)

Runs perception.py on frames of the debug demos (the ones stored with depth and segmentation,
A5) and compares with the simulator's ground truth:
  - correct object: the predicted target mask overlaps the true target more than any other object
    (the T1/T2 test: two identical black bowls, only the instruction tells them apart)
  - mask IoU for target and destination
  - robot-frame center error of the predicted masks vs. the true object centers (as in check_grounding)
  - time per frame
and saves overlays to <outputs>/perception_checks/<task>.png.

Frames are taken before the grasp (first quarter of each demo), where target identity matters and
the target isn't hidden in the gripper.

Usage:
    python -m groundedvla.grounding.eval_perception --tasks T1 T2 T3 T4
"""
import argparse
import json

import h5py
import numpy as np
from PIL import Image, ImageDraw

from groundedvla.envs import camera_utils as cu
from groundedvla.envs.libero_wrapper import LiberoWrapper
from groundedvla.grounding.grounding import mask_centroid
from groundedvla.grounding.perception import InstructionPerception
from groundedvla.tasks import TASKS
from groundedvla.utils.paths import load_paths


def to_cv(img: np.ndarray) -> np.ndarray:
    """Policy orientation (raw rotated 180) -> OpenCV order (raw flipped vertically)."""
    return np.ascontiguousarray(img[:, ::-1])


def iou(a: np.ndarray, b: np.ndarray) -> float:
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else 0.0


def camera_model(key: str):
    """agentview intrinsics and camera->robot-base transform (fixed camera in the replayed demos)."""
    w = LiberoWrapper(TASKS[key], with_depth_seg=False)
    w.reset()
    K = cu.intrinsics(w.sim, "agentview", 256, 256)
    T = cu.world_to_base(w.sim) @ cu.extrinsics(w.sim, "agentview")
    w.close()
    return K, T


def overlay(rgb, res, gt_target) -> Image.Image:
    img = rgb.astype(np.float32).copy()
    for m, color in ((res.target_mask, (255, 60, 60)), (res.dest_mask, (60, 220, 60))):
        img[m] = 0.5 * img[m] + 0.5 * np.array(color)
    img[gt_target & ~res.target_mask] = 0.6 * img[gt_target & ~res.target_mask] + 0.4 * np.array((255, 255, 0))
    out = Image.fromarray(img.astype(np.uint8)).resize((512, 512), Image.NEAREST)
    d = ImageDraw.Draw(out)
    for i, (b, lab, s) in enumerate(zip(res.boxes, res.labels, res.scores)):
        color = (255, 60, 60) if i == res.target_idx else (60, 220, 60) if i == res.dest_idx else (180, 180, 180)
        d.rectangle([v * 2 for v in b], outline=color, width=2)
        d.text((b[0] * 2 + 3, b[1] * 2 + 2), f"{lab} {s:.2f}", fill=color)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=["T1", "T2", "T3", "T4"])
    ap.add_argument("--frames-per-demo", type=int, default=3)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    paths = load_paths()
    out_dir = paths["outputs"] / "perception_checks"
    out_dir.mkdir(parents=True, exist_ok=True)
    # All simulator work first: creating a LIBERO env (OpenGL context) after the vision models are on
    # the GPU crashed the process silently on Windows.
    cameras = {key: camera_model(key) for key in args.tasks}
    perception = InstructionPerception(paths, device=args.device)
    summary = {}

    for key in args.tasks:
        K, T = cameras[key]
        rows, tiles = [], []
        with h5py.File(paths["replayed"] / f"{key}.hdf5", "r") as f:
            instruction = f.attrs["instruction"]
            tid, did = int(f.attrs["target_id"]), int(f.attrs["dest_id"])
            demos = [n for n in f if "agentview_seg" in f[n] and f[n].attrs.get("complete", False)]
            for name in sorted(demos, key=lambda s: int(s.split("_")[-1])):
                d = f[name]
                n = d["agentview_rgb"].shape[0]
                for t in np.linspace(0, n // 4, args.frames_per_demo).astype(int):   # before the grasp
                    rgb, depth, seg = d["agentview_rgb"][t], d["agentview_depth"][t], d["agentview_seg"][t]
                    depth_cv = to_cv(depth)

                    def centers_fn(boxes):
                        cs = []
                        for x0, y0, x1, y1 in boxes:
                            m = np.zeros(seg.shape, bool)
                            # inner half of the box: mostly object, little background
                            cx, cy, hw, hh = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) / 4, (y1 - y0) / 4
                            m[int(cy - hh):int(cy + hh) + 1, int(cx - hw):int(cx + hw) + 1] = True
                            c = mask_centroid(to_cv(m), depth_cv, K, T)
                            cs.append(c if c is not None else np.full(3, np.nan))
                        return np.array(cs)

                    res = perception(rgb, instruction, centers_fn)
                    gt_t, gt_d = seg == tid, seg == did
                    ids = [i for i in np.unique(seg) if i not in (0, tid)]
                    other_best = max((iou(res.target_mask, seg == i) for i in ids), default=0.0)
                    t_iou, d_iou = iou(res.target_mask, gt_t), iou(res.dest_mask, gt_d)
                    pc_t = mask_centroid(to_cv(res.target_mask), depth_cv, K, T)
                    pc_d = mask_centroid(to_cv(res.dest_mask), depth_cv, K, T)
                    err = lambda p, g: float(np.linalg.norm(p - g) * 100) if p is not None else float("nan")
                    rows.append({"demo": name, "t": int(t), "correct_object": bool(t_iou > 0.3 and t_iou > other_best),
                                 "target_iou": t_iou, "dest_iou": d_iou,
                                 "target_err_cm": err(pc_t, d["gt_target"][t]), "dest_err_cm": err(pc_d, d["gt_dest"][t]),
                                 "ms": sum(res.ms.values())})
                    if len(tiles) < 6:
                        tiles.append(overlay(rgb, res, gt_t))
        nanmed = lambda k: float(np.nanmedian([r[k] for r in rows]))
        s = {"instruction": instruction, "frames": len(rows),
             "correct_object": float(np.mean([r["correct_object"] for r in rows])),
             "target_iou_median": nanmed("target_iou"), "dest_iou_median": nanmed("dest_iou"),
             "target_err_cm_median": nanmed("target_err_cm"), "dest_err_cm_median": nanmed("dest_err_cm"),
             "ms_median": nanmed("ms")}
        summary[key] = s
        print(f"{key}: correct object {s['correct_object']:.0%} over {s['frames']} frames | IoU target "
              f"{s['target_iou_median']:.2f} dest {s['dest_iou_median']:.2f} | center error target "
              f"{s['target_err_cm_median']:.1f} cm dest {s['dest_err_cm_median']:.1f} cm | {s['ms_median']:.0f} ms/frame")
        if tiles:
            grid = Image.new("RGB", (512 * 3, 512 * ((len(tiles) + 2) // 3)))
            for i, tile in enumerate(tiles):
                grid.paste(tile, ((i % 3) * 512, (i // 3) * 512))
            grid.save(out_dir / f"{key}.png")
        (out_dir / f"{key}_frames.json").write_text(json.dumps(rows, indent=1))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
