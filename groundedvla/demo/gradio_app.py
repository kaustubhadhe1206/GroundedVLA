"""The interactive demo with a Gradio page, for Colab (T4 is enough) or any machine without a browser.

Notebook (cell returns at once, prints a public *.gradio.live link):
    from groundedvla.demo.gradio_app import launch
    ui = launch("smolvla_libero --task T3")       # same arguments as `python -m groundedvla.demo.app`
    ui.close()                                     # stop and free the GPU (launching again also does this)
Terminal:
    python -m groundedvla.demo.gradio_app smolvla_libero --task T3 [--no-share]

Same engine as the browser demo (groundedvla/demo/app.py, its Demo class): one background thread owns
the MuJoCo context and the policies and runs the commands the page queues. A 0.1 s timer on the page
pulls the latest frames (as inline JPEGs) and the status, so the page never touches the simulator.
Needs gradio 5 (`pip install "gradio>=5.20,<6"`): gradio 6 needs a newer huggingface-hub than LeRobot.
"""
import base64
import html
import sys
import threading

import gradio as gr

from groundedvla.demo.app import Demo, parse_args
from groundedvla.tasks import TASKS
from groundedvla.utils.paths import load_paths

TICK_S = 0.1
_ACTIVE: "GradioDemo | None" = None

CSS = """
.gvla-img img {width: 100%; border-radius: 6px; display: block}
.gvla-status {line-height: 1.55; font-size: 14px}
.gvla-bar {background: #8884; height: 6px; border-radius: 3px; margin: 2px 0 4px}
.gvla-bar div {height: 6px; border-radius: 3px}
"""


def img_html(jpeg: bytes | None, placeholder: str = "") -> str:
    if not jpeg:
        return f"<div style='padding:40px;text-align:center;opacity:.6'>{placeholder}</div>"
    return f"<div class='gvla-img'><img src='data:image/jpeg;base64,{base64.b64encode(jpeg).decode()}'></div>"


def status_html(s: dict) -> str:
    e = html.escape
    color = {"running": "#d97706", "playing": "#2563eb", "error": "#dc2626",
             "loading": "#6b7280"}.get(s["status"], "#16a34a")
    agents = ", ".join(f'{a["label"]} (N={a["horizon"]})' if a.get("horizon") else a["label"]
                       for a in s.get("agents", [])) or "loading the model..."
    rows = [f'<b style="color:{color}">{e(s["status"].upper())}</b> {e(s["message"])}',
            f"Models: {e(agents)}",
            f'Scene {e(str(s.get("task")))}, initial state {s.get("init")}: {e(s.get("conditions_text", ""))}']
    if s.get("max_steps"):
        pct = 100 * s.get("step", 0) / s["max_steps"]
        rows.append(f'Step {s.get("step", 0)} / {s["max_steps"]}'
                    + (f' &nbsp; policy call {s["latency_ms"]} ms' if s.get("latency_ms") else "")
                    + f'<div class="gvla-bar"><div style="width:{pct:.0f}%;background:{color}"></div></div>')
    m = s.get("match")
    if m:
        rows.append(f'Understood as <b>{e(m["task"])}</b>: "{e(m["instruction"])}"' if m.get("task")
                    else f"No trained skill fits ({e(m['reason'])}); prompt sent as typed, no goal check")
    for r in s.get("results", []):
        ok = r["outcome"] == "goal satisfied"
        rows.append(f'<b style="color:{"#16a34a" if ok else "#dc2626"}">{e(r["label"])}: {e(r["outcome"])}</b>'
                    f' after {r["steps"]} steps')
    if s.get("goal_satisfied") and not s.get("results"):
        rows.append('<b style="color:#16a34a">GOAL SATISFIED</b>')
    skills = s.get("skills", [])
    if skills:
        rows.append("<small>Skills in this scene: " + "; ".join(
            f'{e(k["task"])} "{e(k["instruction"])}"' for k in skills) + "</small>")
    return "<div class='gvla-status'>" + "<br>".join(rows) + "</div>"


class GradioDemo:
    def __init__(self, argv: list[str]):
        self.args = parse_args(argv)
        self.demo = Demo(self.args, load_paths())
        self.thread = threading.Thread(target=self._run, daemon=True, name="demo-sim")
        self.thread.start()
        self.blocks = self._build()

    def _run(self):
        try:
            self.demo.main_loop()
        except Exception as e:           # e.g. the model failed to load: show it on the page
            self.demo.update(status="error", message=f"error: {e!r}")
            self.demo.log(f"error: {e!r}")

    def send(self, cmd: str, **kw) -> None:
        self.demo.commands.put({"cmd": cmd, **kw})

    # ------------------------------------------------------------------ page
    def _build(self) -> gr.Blocks:
        a = self.args
        with gr.Blocks(title="GroundedVLA demo", css=CSS, theme=gr.themes.Soft()) as page:
            seen = gr.State({"main": -1, "inputs": -1, "cameras": None, "suggestions": None, "history": None})
            gr.Markdown("### GroundedVLA: type an instruction, watch the robot act in LIBERO")
            with gr.Row():
                with gr.Column(scale=2, min_width=340):
                    with gr.Row():
                        task = gr.Dropdown(list(TASKS), value=a.task if a.task in TASKS else "T1", label="Scene",
                                           min_width=90)
                        init = gr.Number(a.init, precision=0, minimum=0, maximum=49, label="Initial state",
                                         min_width=90)
                        reset = gr.Button("Reset", min_width=80)
                    with gr.Accordion("Test conditions (applied on reset)", open=False):
                        yaw = gr.Slider(-45, 45, 0, step=5, label="Camera yaw (°)")
                        height = gr.Slider(-10, 30, 0, step=5, label="Camera height (cm)")
                        fov = gr.Slider(0.7, 1.4, 1.0, step=0.05, label="Field-of-view scale")
                        dx = gr.Slider(-15, 15, 0, step=1, label="Target shift x (cm, away from robot)")
                        dy = gr.Slider(-15, 15, 0, step=1, label="Target shift y (cm, robot's left)")
                        with gr.Row():
                            apply_btn = gr.Button("Apply & reset")
                            indist = gr.Button("In-distribution")
                    prompt = gr.Textbox(label="Instruction", lines=2, placeholder="e.g. put the bowl on the stove")
                    suggestions = gr.Dropdown([], label="Suggestions (pick one to fill the box)", value=None)
                    with gr.Row():
                        run = gr.Button("Run", variant="primary")
                        stop = gr.Button("Stop", variant="stop")
                        replay = gr.Button("Replay")
                    with gr.Row():
                        playback = gr.Checkbox(a.playback, label="Smooth playback")
                        interpret = gr.Checkbox(True, label="Map my words to a trained skill")
                    status = gr.HTML(status_html(self.demo.snapshot()))
                    gr.Markdown("**What the model sees** (agentview | wrist)")
                    inputs_img = gr.HTML(img_html(None))
                    with gr.Accordion("Log", open=False):
                        log = gr.Textbox(lines=10, max_lines=14, show_label=False, interactive=False)
                with gr.Column(scale=3):
                    camera = gr.Dropdown([a.camera], value=a.camera, label="View", interactive=True)
                    main_img = gr.HTML(img_html(None, "loading the model and the scene..."))

            cond = [task, init, yaw, height, fov, dx, dy]

            def do_reset(t, i, y, h, f, x, yy):
                self.send("reset", task=t, init=int(i or 0), yaw=y, height=h / 100, fov=f, dx=x / 100, dy=yy / 100)

            reset.click(do_reset, cond, None)
            apply_btn.click(do_reset, cond, None)

            def do_indist(t, i):
                do_reset(t, i, 0, 0, 1.0, 0, 0)
                return 0, 0, 1.0, 0, 0

            indist.click(do_indist, [task, init], [yaw, height, fov, dx, dy])
            run.click(lambda p: self.send("run", prompt=p), prompt, None)
            prompt.submit(lambda p: self.send("run", prompt=p), prompt, None)
            stop.click(lambda: self.send("stop"), None, None)
            replay.click(lambda: self.send("replay"), None, None)
            playback.input(lambda v: self.send("playback", on=v), playback, None)
            interpret.input(lambda v: self.send("interpret", on=v), interpret, None)
            camera.input(lambda v: self.send("camera", name=v), camera, None)
            suggestions.input(lambda v: v or gr.skip(), suggestions, prompt)

            def tick(sn):
                s = self.demo.snapshot()
                out = []
                for name, ph in (("main", "loading..."), ("inputs", "")):
                    item = self.demo.frames.data.get(name)
                    if item and item[0] != sn[name]:
                        sn[name] = item[0]
                        out.append(img_html(item[1], ph))
                    else:
                        out.append(gr.skip())
                out.append(status_html(s))
                cams = (tuple(s["cameras"]), s["camera"])
                if cams != sn["cameras"] and s["cameras"]:
                    sn["cameras"] = cams
                    out.append(gr.update(choices=list(cams[0]), value=cams[1]))
                else:
                    out.append(gr.skip())
                sug = tuple(x["text"] for x in s.get("suggestions", []))
                if sug != sn["suggestions"]:
                    sn["suggestions"] = sug
                    out.append(gr.update(choices=[(f'{x["kind"]}: {x["text"]}', x["text"])
                                                  for x in s["suggestions"]], value=None))
                else:
                    out.append(gr.skip())
                hist = "\n".join(s.get("history", []))
                if hist != sn["history"]:
                    sn["history"] = hist
                    out.append(hist)
                else:
                    out.append(gr.skip())
                return *out, sn

            timer = gr.Timer(TICK_S)
            timer.tick(tick, seen, [main_img, inputs_img, status, camera, suggestions, log, seen],
                       show_progress="hidden", concurrency_limit=None)
        return page

    # ------------------------------------------------------------------ lifecycle
    def launch(self, share: bool = True, port: int | None = None, block: bool = False):
        return self.blocks.launch(share=share, server_port=port, prevent_thread_lock=not block,
                                  show_error=True, quiet=False)

    def close(self) -> None:
        """Stop the page and the simulator thread, free the GPU and the MuJoCo context."""
        self.demo.alive = False
        self.send("stop")
        try:
            self.blocks.close()
        except Exception:
            pass
        self.thread.join(timeout=60)
        if self.demo.env is not None:
            self.demo.env.close()
            self.demo.env = None
        self.demo.agents.clear()
        import gc

        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[demo] closed")


def launch(argv: str | list[str] = "smolvla_libero --task T1", share: bool = True,
           port: int | None = None) -> GradioDemo:
    """Start the demo in the background (closing a previous one) and print its link; returns the handle."""
    global _ACTIVE
    if _ACTIVE is not None:
        _ACTIVE.close()
    _ACTIVE = GradioDemo(argv.split() if isinstance(argv, str) else list(argv))
    _ACTIVE.launch(share=share, port=port)
    return _ACTIVE


def main():
    argv = sys.argv[1:]
    share = "--no-share" not in argv
    argv = [x for x in argv if x != "--no-share"]
    ui = GradioDemo(argv)
    try:
        ui.launch(share=share, block=True)
    except KeyboardInterrupt:
        pass
    finally:
        ui.close()


if __name__ == "__main__":
    main()
