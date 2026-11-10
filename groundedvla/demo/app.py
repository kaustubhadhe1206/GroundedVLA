"""Interactive demo: type an instruction in the browser, watch the robot act in LIBERO.

    .\\demo smolvla_libero                                   # HF's LIBERO SmolVLA (horizon 1, slow on a laptop)
    .\\demo smolvla_ckpt --checkpoint D:/groundedvla/checkpoints/v1_rgb/step_020000 --horizon 10
    then open http://127.0.0.1:8765

Left: scene, prompt, status and what the model sees. Right: live simulator view (any scene camera).
The robot only moves after you submit a prompt; a later prompt continues from the current state
until you reset the scene.

Speed: a policy call takes ~0.9 s on the laptop's RTX 3050 (measured; physics 19 ms, rendering 2 ms),
and smolvla_libero must re-plan every step, so live motion runs at ~1 step/s. Smooth playback (the
default) computes the run first, with a live preview, then plays it at the real 20 Hz; Replay
plays the last run again. `--live` starts in live mode instead.

The prompt is fed to the policy verbatim in place of the task instruction. The policy only knows
the tasks it was trained on, so prompts outside them show its limits. "Goal satisfied" is the
scene's own BDDL goal (e.g. bowl on plate), whatever the prompt says.

Threads: MuJoCo's GL context and the policy live in the main thread, which runs the commands the
page sends. The HTTP server only reads the latest rendered JPEGs, so it never touches the sim.
Bound to 127.0.0.1, so it is reachable from this machine only.
"""
import argparse
import io
import json
import os
import queue
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
from PIL import Image

import groundedvla.envs.compat  # noqa: F401  (must precede LIBERO / robosuite imports)
from groundedvla.envs.libero_wrapper import LiberoWrapper
from groundedvla.policies.chunking import MODES, ChunkExecutor
from groundedvla.policies.registry import build_policy, default_horizon, load_registry
from groundedvla.tasks import CONTROL_HZ, TASKS, resolve_task
from groundedvla.utils.paths import REPO_ROOT, load_paths

INDEX_HTML = Path(__file__).with_name("index.html")
PARAPHRASES = REPO_ROOT / "groundedvla" / "ood" / "paraphrases.json"
DISPLAY_CAMERAS = ("agentview", "frontview", "sideview", "birdview", "galleryview")


def jpeg(img: np.ndarray, quality: int = 85) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


class Frames:
    """Latest JPEG per stream, with a condition so the MJPEG handlers wait for new frames."""

    def __init__(self):
        self.cond = threading.Condition()
        self.data: dict[str, tuple[int, bytes]] = {}

    def put(self, name: str, img: np.ndarray) -> bytes:
        b = jpeg(img)
        self.put_bytes(name, b)
        return b

    def put_bytes(self, name: str, b: bytes) -> None:
        with self.cond:
            n = self.data.get(name, (0, b""))[0] + 1
            self.data[name] = (n, b)
            self.cond.notify_all()

    def wait(self, name: str, last: int, timeout: float = 1.0) -> tuple[int, bytes] | None:
        with self.cond:
            self.cond.wait_for(lambda: self.data.get(name, (0, b""))[0] != last, timeout)
            return self.data.get(name)


class Demo:
    def __init__(self, args, paths):
        self.args, self.paths = args, paths
        self.frames = Frames()
        self.commands: queue.Queue = queue.Queue()
        self.lock = threading.Lock()
        self.state = {"status": "loading", "message": "loading the model...", "model": args.model,
                      "mode": args.mode, "horizon": None, "task": None, "instruction": "", "goal": "",
                      "init": 0, "prompt": "", "step": 0, "max_steps": 0, "goal_satisfied": False,
                      "latency_ms": None, "camera": args.camera, "cameras": [], "tasks": list(TASKS),
                      "suggestions": [], "history": [], "playback": args.playback, "has_recording": False}
        self.env: LiberoWrapper | None = None
        self.obs = None
        self.policy = None
        self.recording: list[tuple[bytes, bytes]] = []   # JPEG (main, inputs) per step of the last run

    # ------------------------------------------------------------------ state for the page
    def update(self, **kw) -> None:
        with self.lock:
            self.state.update(kw)

    def snapshot(self) -> dict:
        with self.lock:
            return json.loads(json.dumps(self.state))

    def log(self, text: str) -> None:
        with self.lock:
            self.state["history"] = (self.state["history"] + [f"{time.strftime('%H:%M:%S')}  {text}"])[-12:]
        print(f"[demo] {text}", flush=True)

    # ------------------------------------------------------------------ rendering
    def render(self) -> tuple[bytes, bytes | None]:
        """Show the current state; returns the JPEGs so a run can be recorded for playback."""
        cam = self.state["camera"]
        w, h = self.args.display_size
        raw = self.env.sim.render(camera_name=cam, width=w, height=h)
        main = self.frames.put("main", np.ascontiguousarray(raw[::-1]))   # true camera view, upright
        inputs = None
        if self.obs is not None:
            inputs = self.frames.put("inputs", np.concatenate(
                [self.obs["observation.images.image"], self.obs["observation.images.image2"]], 1))
        return main, inputs

    def play(self, frames: list[tuple[bytes, bytes | None]], label: str) -> bool:
        """Play recorded frames at the real control rate (20 Hz). Returns False if stopped."""
        self.update(status="playing", message=f"{label}: real-time playback")
        for main, inputs in frames:
            t0 = time.perf_counter()
            if self.stop_requested():
                return False
            self.frames.put_bytes("main", main)
            if inputs is not None:
                self.frames.put_bytes("inputs", inputs)
            time.sleep(max(0.0, 1.0 / CONTROL_HZ - (time.perf_counter() - t0)))
        return True

    # ------------------------------------------------------------------ commands (main thread)
    def load_scene(self, task_key: str, init: int) -> None:
        spec = resolve_task(task_key)
        self.update(status="loading", message=f"building scene {task_key}...")
        if self.env is None or self.env.task.key != spec.key:
            if self.env is not None:
                self.env.close()
            self.env = LiberoWrapper(spec, with_depth_seg=False)
            self.init_states = self.env.init_states()
        init = int(init) % len(self.init_states)
        self.obs = self.env.reset(self.init_states[init], settle=True)
        names = {self.env.sim.model.camera_id2name(i) for i in range(self.env.sim.model.ncam)}
        cams = [c for c in DISPLAY_CAMERAS if c in names]
        g = self.env.goal
        self.update(status="idle", message="type an instruction and press Run", task=spec.key, init=init,
                    instruction=self.env.instruction, goal=f"{g.target} {g.predicate.lower()} {g.destination_object}",
                    step=0, max_steps=self.args.max_steps or spec.max_steps, goal_satisfied=self.env.eval_goal(),
                    cameras=cams, camera=self.state["camera"] if self.state["camera"] in cams else cams[0],
                    suggestions=self.suggestions(spec.key, self.env.instruction), prompt="")
        self.log(f"scene {spec.key}, initial state {init}")
        self.render()

    @staticmethod
    def suggestions(task_key: str, instruction: str) -> list[dict]:
        out = [{"text": instruction, "kind": "trained"}]
        try:
            para = json.loads(PARAPHRASES.read_text()).get(task_key, [])
        except (OSError, ValueError):
            para = []
        out += [{"text": p, "kind": "paraphrase"} for p in para if p and p != "TODO"]
        return out

    def run_prompt(self, prompt: str) -> None:
        prompt = " ".join(prompt.split())
        if not prompt:
            return
        max_steps, playback = self.state["max_steps"], self.state["playback"]
        self.update(status="running", prompt=prompt, step=0, message="thinking...")
        self.log(f'prompt: "{prompt}"')
        self.policy.reset()
        self.executor.reset()
        latencies = []
        frames = [self.render()]

        def predict(o):
            t0 = time.perf_counter()
            chunk = self.policy.predict_chunk(o)
            latencies.append((time.perf_counter() - t0) * 1e3)
            self.update(latency_ms=round(latencies[-1]))
            return chunk

        outcome, step = "time limit reached", 0
        while step < max_steps:
            if self.stop_requested():
                outcome = "stopped"
                break
            step += 1
            t0 = time.perf_counter()
            self.obs["task"] = prompt                     # the typed prompt replaces the task instruction
            action, _ = self.executor.act(self.obs, predict)
            self.obs, _, _, info = self.env.step(self.env.action_from_policy(action))
            satisfied = bool(info["success"])
            msg = (f"computing step {step} / {max_steps}, smooth playback starts when done" if playback
                   else "acting live")
            self.update(step=step, goal_satisfied=satisfied, message=msg)
            frames.append(self.render())
            if satisfied:
                outcome = "goal satisfied"
                break
            if self.args.realtime and not playback:
                time.sleep(max(0.0, 1.0 / CONTROL_HZ - (time.perf_counter() - t0)))
        lat = f", median policy call {np.median(latencies):.0f} ms" if latencies else ""
        self.log(f"{outcome} after {step} steps{lat}")
        self.recording = frames
        self.update(has_recording=True)
        if playback and outcome != "stopped":
            self.play(frames, outcome)
        self.update(status="idle", message=f"{outcome} after {step} steps. Type another prompt, Replay, or Reset.")

    def replay(self) -> None:
        if self.recording:
            done = self.play(self.recording, "replay")
            self.update(status="idle", message="replay finished" if done else "replay stopped")

    def stop_requested(self) -> bool:
        """Drain the queue while running: 'stop' ends the episode, other commands wait their turn."""
        pending, stop = [], False
        while True:
            try:
                cmd = self.commands.get_nowait()
            except queue.Empty:
                break
            if cmd["cmd"] == "stop":
                stop = True
            elif cmd["cmd"] == "camera":
                self.set_camera(cmd.get("name", ""))
            else:
                pending.append(cmd)
        for cmd in pending:
            self.commands.put(cmd)
        return stop

    def set_camera(self, name: str) -> None:
        if name in self.state["cameras"]:
            self.update(camera=name)
            if self.env is not None:
                self.render()

    def handle(self, cmd: dict) -> None:
        kind = cmd.get("cmd")
        if kind == "reset":
            self.load_scene(cmd.get("task", self.state["task"]), cmd.get("init", self.state["init"]))
        elif kind == "run":
            self.run_prompt(str(cmd.get("prompt", ""))[:500])
        elif kind == "camera":
            self.set_camera(cmd.get("name", ""))
        elif kind == "replay":
            self.replay()
        elif kind == "playback":
            self.update(playback=bool(cmd.get("on", True)))

    def main_loop(self) -> None:
        a = self.args
        task = resolve_task(a.task)
        self.policy = build_policy(a.model, task, self.paths, checkpoint=a.checkpoint, device=a.device)
        horizon = a.horizon or default_horizon(a.model)
        self.executor = ChunkExecutor(a.mode, horizon, a.te_m)
        self.update(horizon=horizon if a.mode == "naive" else None)
        self.log(f"loaded {a.model} ({self.policy.checkpoint})")
        self.load_scene(task.key, a.init)
        while True:
            try:
                cmd = self.commands.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                self.handle(cmd)
            except Exception as e:  # keep the demo alive; show the error on the page
                self.update(status="idle", message=f"error: {e}")
                self.log(f"error: {e!r}")


def make_handler(demo: Demo):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # quiet
            pass

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            try:
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass  # the browser went away mid-response

        def do_GET(self):
            path = self.path.split("?")[0]
            if path == "/":
                self._send(200, INDEX_HTML.read_bytes(), "text/html; charset=utf-8")
            elif path == "/state":
                self._send(200, json.dumps(demo.snapshot()).encode(), "application/json")
            elif path in ("/stream/main", "/stream/inputs"):
                self.stream(path.rsplit("/", 1)[1])
            else:
                self._send(404, b"not found", "text/plain")

        def stream(self, name: str) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            last = 0
            try:
                while True:
                    item = demo.frames.wait(name, last)
                    if item is None or item[0] == last:
                        continue
                    last, data = item
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(data)}\r\n\r\n".encode() + data + b"\r\n")
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass

        def do_POST(self):
            if self.path != "/cmd":
                return self._send(404, b"not found", "text/plain")
            try:
                n = min(int(self.headers.get("Content-Length", 0)), 10_000)
                cmd = json.loads(self.rfile.read(n) or b"{}")
                if cmd.get("cmd") not in ("reset", "run", "stop", "camera", "replay", "playback"):
                    raise ValueError("unknown command")
            except (ValueError, json.JSONDecodeError) as e:
                return self._send(400, json.dumps({"error": str(e)}).encode(), "application/json")
            demo.commands.put(cmd)
            self._send(200, b'{"ok": true}', "application/json")

    return Handler


def parse_args():
    smolvla = [k for k, v in load_registry().items() if v["type"] == "smolvla"]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", nargs="?", default="smolvla_libero", choices=smolvla)
    ap.add_argument("--checkpoint", default=None, help="model dir (required for smolvla_ckpt)")
    ap.add_argument("--task", default="T1", help="initial scene: T1..T4 or <suite>:<task_id>")
    ap.add_argument("--init", type=int, default=0, help="initial-state index")
    ap.add_argument("--mode", choices=MODES, default="naive")
    ap.add_argument("--horizon", type=int, default=None, help="naive: actions per chunk (default: models.yaml)")
    ap.add_argument("--te-m", type=float, default=0.01)
    ap.add_argument("--max-steps", type=int, default=None, help="per prompt (default: the task's limit)")
    ap.add_argument("--camera", default="agentview", help="initial display camera")
    ap.add_argument("--display-size", type=int, nargs=2, default=(640, 480), metavar=("W", "H"))
    ap.add_argument("--no-realtime", dest="realtime", action="store_false", help="live mode: don't cap at 20 Hz")
    ap.add_argument("--live", dest="playback", action="store_false",
                    help="start in live mode (show each step as computed) instead of smooth playback")
    ap.add_argument("--device", default=None)
    ap.add_argument("--port", type=int, default=8765)
    return ap.parse_args()


class Server(ThreadingHTTPServer):
    # On Windows SO_REUSEADDR lets a second server bind the same port silently, so a stale demo
    # would keep answering. Without it, starting the demo twice fails loudly instead.
    allow_reuse_address = os.name != "nt"
    daemon_threads = True


def main():
    args = parse_args()
    try:
        server = Server(("127.0.0.1", args.port), None)
    except OSError:
        sys.exit(f"[demo] port {args.port} is in use: is the demo already running? Close it or pass --port.")
    demo = Demo(args, load_paths())
    server.RequestHandlerClass = make_handler(demo)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"[demo] open http://127.0.0.1:{args.port}  (Ctrl+C to quit)", flush=True)
    try:
        demo.main_loop()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        if demo.env is not None:
            demo.env.close()
        sys.exit(0)


if __name__ == "__main__":
    main()
