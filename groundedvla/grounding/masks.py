"""Target/destination mask image for V2 and V3 (Step A6).

Red = target mask, green = destination mask, blue = 0. Fed as `observation.images.mask`.

Segmentation IDs: robosuite's "instance" segmentation labels each pixel with
`1 + index of the instance in env.model.instances_to_ids` (0 = background / unlabeled). The
instances are the scene objects and fixtures by BDDL name (akita_black_bowl_1, plate_1,
flat_stove_1, ...) plus the robot parts (MountedPanda0, RethinkMount0, PandaGripper0). These are
not MuJoCo body or geom IDs; `instance_id_map` is saved with the dataset (Step A5).
"""
import numpy as np


def instance_id_map(env) -> dict[str, int]:
    """{instance name: segmentation id} for the robosuite env (`LiberoWrapper.env.env`)."""
    return {name: i + 1 for i, name in enumerate(env.model.instances_to_ids)}


def instance_ids(env, names: list[str]) -> list[int]:
    mapping = instance_id_map(env)
    for n in names:
        if n not in mapping:
            raise KeyError(f"{n} not in instances {list(mapping)}")
    return [mapping[n] for n in names]


def binary_mask(seg: np.ndarray, inst_id: int) -> np.ndarray:
    seg = seg.squeeze(-1) if seg.ndim == 3 else seg
    return seg == inst_id


def mask_image(seg: np.ndarray, target_id: int, dest_id: int) -> np.ndarray:
    """(H, W[,1]) instance segmentation -> (H, W, 3) uint8 mask image."""
    t, d = binary_mask(seg, target_id), binary_mask(seg, dest_id)
    img = np.zeros((*t.shape, 3), dtype=np.uint8)
    img[..., 0] = t * 255
    img[..., 1] = d * 255
    return img
