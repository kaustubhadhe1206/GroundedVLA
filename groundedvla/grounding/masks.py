"""Target/destination mask image for V2 and V3 (Step A6).

Red = target mask, green = destination mask, blue = 0. Fed as `observation.images.mask`.
"""
import numpy as np


def instance_ids(env, names: list[str]) -> list[int]:
    """Segmentation IDs for object instances. TODO(A6): confirm id mapping in LIBERO/robosuite."""
    mapping = env.model.instances_to_ids  # name -> {"geom": [...]} / id
    ids = []
    for n in names:
        if n not in mapping:
            raise KeyError(f"{n} not in instances {list(mapping)}")
        ids.append(list(mapping).index(n))
    return ids


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
