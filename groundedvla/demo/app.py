"""Interactive demo: type an instruction in the browser, watch the robot act in LIBERO.

    .\\demo smolvla_libero                                   # HF's LIBERO SmolVLA (horizon 1, slow on a laptop)
    .\\demo smolvla_ckpt --checkpoint D:/groundedvla/checkpoints/v1_rgb/step_020000 --horizon 10
    # side by side: two models on the identical scene, e.g. V1 vs V3 under a rotated camera
    .\\demo smolvla_ckpt --checkpoint <v1 dir> --label V1 --compare smolvla_ckpt --compare-checkpoint <v3 dir> --compare-label V3
    then open http://127.0.0.1:8765

Left: scene, test conditions (camera rotation / height / zoom, target shift), prompt, status, what the
model sees. Right: live simulator view (any scene camera), or both models side by side.
The robot only moves after you submit a prompt. With one model, a later prompt continues from the
current state until you reset; when comparing, every run starts both models from the same scene.

Speed: a policy call takes ~0.8 s on the laptop's RTX 3050 (measured; physics 19 ms, rendering 2 ms),
and smolvla_libero must re-plan every step, so live motion runs at ~1 step/s. Smooth playback (the
default) computes the run first, with a live preview, then plays it at the real 20 Hz; Replay
plays the last run again. `--live` starts in live mode instead (single model only).

The prompt is fed to the policy verbatim in place of the task instruction. The policy only knows
the tasks it was trained on, so prompts outside them show its limits. "Goal satisfied" is the
scene's own BDDL goal (e.g. bowl on plate), whatever the prompt says.

Threads: MuJoCo's GL context and the policies live in the main thread, which runs the commands the
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
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import groundedvla.envs.compat  # noqa: F401  (must precede LIBERO / robosuite imports)
from groundedvla.envs.libero_wrapper import CameraPerturbation, LiberoWrapper
from groundedvla.policies.chunking import MODES, ChunkExecutor
from groundedvla.policies.registry import build_policy, default_horizon, load_registry
from groundedvla.tasks import CONTROL_HZ, NUM_SETTLE_STEPS, TASKS, resolve_task
from groundedvla.utils.paths import REPO_ROOT, load_paths

INDEX_HTML = Path(__file__).with_name("index.html")
PARAPHRASES = REPO_ROOT / "groundedvla" / "ood" / "paraphrases.json"
DISPLAY_CAMERAS = ("agentview", "frontview", "sideview", "birdview", "galleryview")
MAX_SHIFT_M = 0.15


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


@dataclass
class Agent:
    label: str
    policy: object
    executor: ChunkExecutor
    horizon: int | None


@dataclass
class Conditions:
    """Test conditions applied at reset: agentview camera change (A7) and a target shift (6.1)."""
    yaw: float = 0.0          # degrees, orbit around the workspace
    height: float = 0.0       # meters
    fov: float = 1.0          # field-of-view scale
    dx: float = 0.0           # target shift in meters, world/robot x (away from the robot)
    dy: float = 0.0           # world/robot y (to the robot's left)

    @classmethod
    def from_cmd(cls, cmd: dict, current: "Conditions") -> "Conditions":
        c = cls(**{k: float(cmd.get(k, getattr(current, k))) for k in ("yaw", "height", "fov", "dx", "dy")})
        c.yaw = float(np.clip(c.yaw, -45, 45))
        c.height = float(np.clip(c.height, -0.1, 0.3))
        c.fov = float(np.clip(c.fov, 0.7, 1.4))
        c.dx, c.dy = (float(np.clip(v, -MAX_SHIFT_M, MAX_SHIFT_M)) for v in (c.dx, c.dy))
        return c

    def camera(self) -> CameraPerturbation | None:
        if self.yaw or self.height or self.fov != 1.0:
            return CameraPerturbation(self.yaw, self.height, self.fov)
        return None

    @property
    def is_default(self) -> bool:
        return self.camera() is None and not (self.dx or self.dy)

    def describe(self) -> str:
        parts = []
        if self.yaw:
            parts.append(f"camera yaw {self.yaw:+g}°")
        if self.height:
            parts.append(f"camera height {self.height * 100:+g} cm")
        if self.fov != 1.0:
            parts.append(f"FOV ×{self.fov:g}")
        if self.dx or self.dy:
            parts.append(f"target shifted ({self.dx * 100:+g}, {self.dy * 100:+g}) cm")
        return ", ".join(parts) or "in-distribution (no change)"


@dataclass
class Recording:
    label: str
    frames: list = field(default_factory=list)    # (main jpeg, inputs jpeg, goal satisfied, step)
    outcome: str = ""
    steps: int = 0


class Demo:
    def __init__(self, args, paths):
        self.args, self.paths = args, paths
        self.frames = Frames()
        self.commands: queue.Queue = queue.Queue()
        self.lock = threading.Lock()
        self.cond = Conditions()
        self.state = {"status": "loading", "message": "loading the model...", "agents": [], "compare": False,
                      "mode": args.mode, "task": None, "instruction": "", "goal": "",
                      "init": 0, "prompt": "", "step": 0, "max_steps": 0, "goal_satisfied": False,
                      "latency_ms": None, "camera": args.camera, "cameras": [], "tasks": list(TASKS),
                      "suggestions": [], "history": [], "playback": args.playback, "has_recording": False,
                      "conditions": vars(self.cond).copy(), "conditions_text": self.cond.describe(),
                      "running_agent": None, "results": []}
        self.env: LiberoWrapper | None = None
        self.obs = None
        self.agents: list[Agent] = []
        self.recording: list[tuple[bytes, bytes | None]] = []   # what Replay plays

    # ------------------------------------------------------------------ state for the page
    def update(self, **kw) -> None:
        with self.lock:
            self.state.update(kw)

    def snapshot(self) -> dict:
        with self.lock:
            return json.loads(json.dumps(self.state))

    def log(self, text: str) -> None:
        with self.lock:
            self.state["history"] = (self.state["history"] + [f"{time.strftime('%H:%M:%S')}  {text}"])[-14:]
        print(f"[demo] {text}", flush=True)

    # ------------------------------------------------------------------ rendering
    def render_main(self) -> np.ndarray:
        w, h = self.args.display_size
        raw = self.env.sim.render(camera_name=self.state["camera"], width=w, height=h)
        return np.ascontiguousarray(raw[::-1])                     # true camera view, upright

    def render(self) -> tuple[bytes, bytes | None]:
        """Show the current state; returns the JPEGs so a run can be recorded for playback."""
        main = self.frames.put("main", self.render_main())
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

    # ------------------------------------------------------------------ scene
    def load_scene(self, task_key: str, init: int, quiet: bool = False) -> None:
        spec = resolve_task(task_key)
        self.update(status="loading", message=f"building scene {task_key}...")
        if self.env is None or self.env.task.key != spec.key:
            if self.env is not None:
                self.env.close()
            self.env = LiberoWrapper(spec, with_depth_seg=self.needs_depth_seg)
            self.init_states = self.env.init_states()
        init = int(init) % len(self.init_states)
        self.env.camera_perturbation = self.cond.camera()
        self.obs = self.env.reset(self.init_states[init], settle=True)
        if self.cond.dx or self.cond.dy:
            self.shift_target(self.cond.dx, self.cond.dy)
        names = {self.env.sim.model.camera_id2name(i) for i in range(self.env.sim.model.ncam)}
        cams = [c for c in DISPLAY_CAMERAS if c in names]
        g = self.env.goal
        self.update(status="idle", message="type an instruction and press Run", task=spec.key, init=init,
                    instruction=self.env.instruction, goal=f"{g.target} {g.predicate.lower()} {g.destination_object}",
                    step=0, max_steps=self.args.max_steps or spec.max_steps, goal_satisfied=self.env.eval_goal(),
                    cameras=cams, camera=self.state["camera"] if self.state["camera"] in cams else cams[0],
                    suggestions=self.suggestions(spec.key, self.env.instruction),
                    conditions=vars(self.cond).copy(), conditions_text=self.cond.describe())
        if not quiet:
            self.log(f"scene {spec.key}, initial state {init}, {self.cond.describe()}")
        self.render()

    def shift_target(self, dx: float, dy: float) -> None:
        """Move the target object horizontally (free joint), then let it settle."""
        joint = self.env.robosuite_env.objects_dict[self.env.goal.target].joints[0]
        sim = self.env.sim
        q = sim.data.get_joint_qpos(joint).copy()
        q[0] += dx
        q[1] += dy
        sim.data.set_joint_qpos(joint, q)
        sim.data.set_joint_qvel(joint, np.zeros(6))
        sim.forward()
        for _ in range(NUM_SETTLE_STEPS):
            self.obs, *_ = self.env.step(self.env.dummy_action())

    @property
    def needs_depth_seg(self) -> bool:
        return any(getattr(a.policy, "needs_depth_seg", False) for a in self.agents)

    @staticmethod
    def suggestions(task_key: str, instruction: str) -> list[dict]:
        out = [{"text": instruction, "kind": "trained"}]
        try:
            para = json.loads(PARAPHRASES.read_text()).get(task_key, [])
        except (OSError, ValueError):
            para = []
        out += [{"text": p, "kind": "paraphrase"} for p in para if p and p != "TODO"]
        return out

    # ------------------------------------------------------------------ runs
    def run_agent(self, agent: Agent, prompt: str, live: bool) -> Recording:
        """One episode of one agent from the current scene state, recording every frame."""
        max_steps = self.state["max_steps"]
        rec = Recording(agent.label)
        agent.policy.reset()
        agent.executor.reset()
        latencies = []

        def predict(o):
            t0 = time.perf_counter()
            chunk = agent.policy.predict_chunk(o)
            latencies.append((time.perf_counter() - t0) * 1e3)
            self.update(latency_ms=round(latencies[-1]))
            return chunk

        main, inputs = self.render()
        rec.frames.append((main, inputs, self.env.eval_goal(), 0))
        rec.outcome, step = "time limit reached", 0
        while step < max_steps:
            if self.stop_requested():
                rec.outcome = "stopped"
                break
            step += 1
            t0 = time.perf_counter()
            self.obs["task"] = prompt                     # the typed prompt replaces the task instruction
            action, _ = agent.executor.act(self.obs, predict)
            self.obs, _, _, info = self.env.step(self.env.action_from_policy(action))
            satisfied = bool(info["success"])
            msg = "acting live" if live else f"{agent.label}: computing step {step} / {max_steps}, playback follows"
            self.update(step=step, goal_satisfied=satisfied, message=msg)
            main, inputs = self.render()
            rec.frames.append((main, inputs, satisfied, step))
            if satisfied:
                rec.outcome = "goal satisfied"
                break
            if live and self.args.realtime:
                time.sleep(max(0.0, 1.0 / CONTROL_HZ - (time.perf_counter() - t0)))
        rec.steps = step
        lat = f", median policy call {np.median(latencies):.0f} ms" if latencies else ""
        self.log(f"{agent.label}: {rec.outcome} after {step} steps{lat}")
        return rec

    def run_prompt(self, prompt: str) -> None:
        prompt = " ".join(prompt.split())
        if not prompt:
            return
        self.update(status="running", prompt=prompt, step=0, message="thinking...", results=[])
        self.log(f'prompt: "{prompt}"' + ("" if self.cond.is_default else f" [{self.cond.describe()}]"))
        compare = len(self.agents) > 1
        live = not self.state["playback"] and not compare
        recs = []
        for agent in self.agents:
            if compare:                                 # identical starting scene for every model
                self.load_scene(self.state["task"], self.state["init"], quiet=True)
                self.update(status="running")
            self.update(running_agent=agent.label)
            recs.append(self.run_agent(agent, prompt, live))
            if recs[-1].outcome == "stopped":
                break
        self.update(running_agent=None,
                    results=[{"label": r.label, "outcome": r.outcome, "steps": r.steps} for r in recs])
        stopped = any(r.outcome == "stopped" for r in recs)
        frames = self.side_by_side(recs) if compare and not stopped else [(m, i) for m, i, _, _ in recs[0].frames]
        self.recording = frames
        self.update(has_recording=True)
        if not live and not stopped:
            self.play(frames, " vs ".join(f"{r.label}: {r.outcome}" for r in recs))
        summary = "; ".join(f"{r.label}: {r.outcome} after {r.steps} steps" for r in recs)
        self.update(status="idle", message=f"{summary}. Type another prompt, Replay, or Reset.")

    def side_by_side(self, recs: list[Recording]) -> list[tuple[bytes, bytes | None]]:
        """Pair the agents' frames step by step (the shorter run holds its last frame), with labels."""
        self.update(message="composing the side-by-side video...")
        font = ImageFont.load_default(size=18)
        n = max(len(r.frames) for r in recs)
        out = []
        for t in range(n):
            tiles = []
            for r in recs:
                main, _, satisfied, step = r.frames[min(t, len(r.frames) - 1)]
                img = Image.open(io.BytesIO(main)).convert("RGB")
                d = ImageDraw.Draw(img)
                done = t >= len(r.frames) - 1
                status = (("GOAL SATISFIED" if satisfied else r.outcome.upper()) if done else f"step {step}")
                color = (60, 220, 120) if satisfied else (255, 120, 120) if done else (255, 255, 255)
                d.rectangle([0, 0, img.width, 32], fill=(15, 17, 21))
                d.text((12, 6), r.label, fill=(255, 255, 255), font=font)
                d.text((img.width - 12, 6), status, fill=color, font=font, anchor="ra")
                tiles.append(np.asarray(img))
            gap = np.full((tiles[0].shape[0], 6, 3), 15, np.uint8)
            frame = np.concatenate([tiles[0]] + [x for tile in tiles[1:] for x in (gap, tile)], axis=1)
            out.append((jpeg(frame), recs[0].frames[min(t, len(recs[0].frames) - 1)][1]))
        return out

    def replay(self) -> None:
        if self.recording:
            done = self.play(self.recording, "replay")
            self.update(status="idle", message="replay finished" if done else "replay stopped")

    # ------------------------------------------------------------------ command loop
    def stop_requested(self) -> bool:
        """Drain the queue while busy: 'stop' ends the run, other commands wait their turn."""
        pending, stop = [], False
        while True:
            try:
                cmd = self.commands.get_nowait()
            except queue.Empty:
                break
            if cmd["cmd"] == "stop":
                stop = True
            elif cmd["cmd"] == "camera":
                self.set_camera(cmd.get("name", ""), render=False)
            else:
                pending.append(cmd)
        for cmd in pending:
            self.commands.put(cmd)
        return stop

    def set_camera(self, name: str, render: bool = True) -> None:
        if name in self.state["cameras"]:
            self.update(camera=name)
            if render and self.env is not None:
                self.render()

    def handle(self, cmd: dict) -> None:
        kind = cmd.get("cmd")
        if kind == "reset":
            self.cond = Conditions.from_cmd(cmd, self.cond)
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
        specs = [(a.model, a.checkpoint, a.horizon, a.label)]
        if a.compare:
            specs.append((a.compare, a.compare_checkpoint, a.compare_horizon, a.compare_label))
        for model, ckpt, horizon, label in specs:
            policy = build_policy(model, task, self.paths, checkpoint=ckpt, device=a.device)
            h = horizon or default_horizon(model)
            label = label or (Path(ckpt).parent.name if ckpt else model)
            self.agents.append(Agent(label, policy, ChunkExecutor(a.mode, h, a.te_m), h if a.mode == "naive" else None))
            self.log(f"loaded {label}: {model} ({policy.checkpoint}), "
                     + (f"N={h}" if a.mode == "naive" else "temporal ensemble"))
        self.update(agents=[{"label": ag.label, "horizon": ag.horizon} for ag in self.agents],
                    compare=len(self.agents) > 1)
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
        def log_message(self, *_):  # quiet
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
            try:
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                last = 0
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
    ap.add_argument("--horizon", type=int, default=None, help="naive: actions per chunk (default: models.yaml)")
    ap.add_argument("--label", default=None, help="name shown on the page (default: run or model name)")
    ap.add_argument("--compare", default=None, choices=smolvla, help="second model, run on the same scene")
    ap.add_argument("--compare-checkpoint", default=None)
    ap.add_argument("--compare-horizon", type=int, default=None)
    ap.add_argument("--compare-label", default=None)
    ap.add_argument("--task", default="T1", help="initial scene: T1..T4 or <suite>:<task_id>")
    ap.add_argument("--init", type=int, default=0, help="initial-state index")
    ap.add_argument("--mode", choices=MODES, default="naive")
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
