"""Make robosuite 1.4.0 work on native Windows (local debugging only; not needed on Colab).

1. robosuite loads `robosuite/utils/mujoco.dll` on Windows -> copy it from the mujoco package.
2. robosuite.macros.MUJOCO_GPU_RENDERING=True forces MUJOCO_GL=egl, which is invalid on
   Windows -> make it False on Windows (GLFW/WGL is used instead).
Idempotent: safe to run again after reinstalling robosuite.
"""
import importlib.util
import platform
import shutil
from pathlib import Path


def pkg_dir(name: str) -> Path:
    return Path(importlib.util.find_spec(name).origin).parent


def main():
    if platform.system() != "Windows":
        print("not Windows, nothing to do")
        return
    rs, mj = pkg_dir("robosuite"), pkg_dir("mujoco")

    dll = rs / "utils" / "mujoco.dll"
    if not dll.exists():
        shutil.copy2(mj / "mujoco.dll", dll)
        print(f"copied mujoco.dll -> {dll}")

    macros = rs / "macros.py"
    text = macros.read_text()
    old, new = "MUJOCO_GPU_RENDERING = True", 'MUJOCO_GPU_RENDERING = __import__("platform").system() != "Windows"'
    if old in text:
        macros.write_text(text.replace(old, new))
        print(f"patched {macros}")

    private = rs / "macros_private.py"
    if not private.exists():
        shutil.copy2(macros, private)  # silences the "no private macro file" warning
    print("ok")


if __name__ == "__main__":
    main()
