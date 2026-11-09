#!/bin/bash
# Install a pinned requirements file on Colab, run from the repo root by every notebook:
#   bash scripts/colab_install.sh requirements.txt          (SmolVLA / LeRobot runtime)
#   bash scripts/colab_install.sh requirements-openvla.txt  (OpenVLA runtime)
#
# hf-libero and robomimic depend on egl_probe / hf-egl-probe, which fail to build on Colab and are
# never imported by our code (robomimic only uses them in robomimic.envs; the local env runs
# without them too). So those two are installed with --no-deps, plus the dependencies they
# actually use, pinned to the local env's versions.
set -euo pipefail
REQ="${1:-requirements.txt}"
NO_DEPS='^(hf-libero|robomimic)=='

grep -v -E "$NO_DEPS" "$REQ" > /tmp/gvla_requirements.txt
pip install -q -r /tmp/gvla_requirements.txt
pip install -q --no-deps $(grep -E "$NO_DEPS" "$REQ" | tr '\n' ' ')
pip install -q thop==0.1.1.post2209072238 tensorboardX==2.6.5 termcolor==3.3.0 psutil tqdm matplotlib tensorboard
pip install -q -e . --no-deps
python -c "import libero.libero, robosuite, mujoco; print('LIBERO stack OK, mujoco', mujoco.__version__)"
