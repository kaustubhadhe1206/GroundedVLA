"""Environment fixes that must run before any LIBERO env is created (plan Step A1, Section 2.3).

Import this module first (libero_wrapper does it for you):
1. Pick the MuJoCo GL backend: EGL on headless Linux (Colab/CARC), GLFW on Windows.
2. Patch robosuite 1.4.0's segmentation readout, which overflows under NumPy 2
   (`uint8 * 2**8` -> OverflowError). Patched in memory, never in site-packages, so the same
   code works locally and on a fresh Colab runtime.
"""
import os
import platform


def setup_rendering() -> None:
    system = platform.system()
    if system == "Linux":
        os.environ.setdefault("MUJOCO_GL", "egl")
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    elif system == "Windows":
        os.environ.setdefault("MUJOCO_GL", "glfw")


def patch_robosuite_segmentation() -> None:
    import mujoco
    import numpy as np
    from robosuite.utils import binding_utils as bu

    if getattr(bu.MjRenderContext.read_pixels, "_gvla_patched", False):
        return

    def read_pixels(self, width, height, depth=False, segmentation=False):
        viewport = mujoco.MjrRect(0, 0, width, height)
        rgb_img = np.empty((height, width, 3), dtype=np.uint8)
        depth_img = np.empty((height, width), dtype=np.float32) if depth else None
        mujoco.mjr_readPixels(rgb=rgb_img, depth=depth_img, viewport=viewport, con=self.con)

        ret_img = rgb_img
        if segmentation:
            rgb = rgb_img.astype(np.int32)  # the only change: avoid uint8 overflow
            seg_img = rgb[:, :, 0] + rgb[:, :, 1] * (2**8) + rgb[:, :, 2] * (2**16)
            seg_img[seg_img >= (self.scn.ngeom + 1)] = 0
            seg_ids = np.full((self.scn.ngeom + 1, 2), fill_value=-1, dtype=np.int32)
            for i in range(self.scn.ngeom):
                geom = self.scn.geoms[i]
                if geom.segid != -1:
                    seg_ids[geom.segid + 1, 0] = geom.objtype
                    seg_ids[geom.segid + 1, 1] = geom.objid
            ret_img = seg_ids[seg_img]

        if depth:
            return ret_img, depth_img
        return ret_img

    read_pixels._gvla_patched = True
    bu.MjRenderContext.read_pixels = read_pixels


setup_rendering()
patch_robosuite_segmentation()
