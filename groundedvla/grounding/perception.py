"""Predicted target/destination masks from the instruction (improvement 5: no simulator oracle).

V2/V3 normally take the target and destination from the BDDL goal and their masks from the
simulator's segmentation (plan Section 3.2), so they are told which object matters. This module
replaces that oracle with off-the-shelf perception that reads the instruction:

  1. parse_instruction: "pick up the black bowl between the plate and the ramekin and place it
     on the plate" -> target "black bowl", relation "between" [plate, ramekin], destination "plate"
  2. Grounding DINO (open-vocabulary detection) finds every noun in the sentence
  3. resolve_target: open-vocabulary detectors can't resolve "between"/"next to", and T1/T2 have
     two identical black bowls, so the relation is resolved geometrically on the candidates'
     3D positions (from depth), like a modular robotics pipeline would
  4. SAM turns the chosen target and destination boxes into masks
  5. mask_centroid (grounding.py) gives robot-frame centers from those masks + depth, exactly as
     for the simulator masks

Depth and camera calibration still come from the simulator (an RGB-D camera with known pose).
All images here are in the policy's orientation (upright, as stored in the dataset).

Usage: see groundedvla/grounding/eval_perception.py.
"""
import re
import time
from dataclasses import dataclass, field

import numpy as np

DETECTOR_ID = "IDEA-Research/grounding-dino-tiny"
SAM_ID = "facebook/sam-vit-base"
BOX_THRESHOLD = 0.25
TEXT_THRESHOLD = 0.2
NMS_IOU = 0.5

_CLAUSE = r"(?P<clause>(?:between|next to|on|in|from) .+?)"
_PICK = r"(?:pick up|grab|take|move)"
_PLACE = r"(?:place|put|set)"
_PATTERNS = [
    # LIBERO-Spatial style: "pick up the <target> <relation clause> and place it on the <destination>"
    re.compile(rf"^{_PICK} the (?P<target>.+?) {_CLAUSE} and {_PLACE} it (?:down )?(?:on|in) the (?P<dest>.+)$"),
    re.compile(rf"^{_PICK} the (?P<target>.+?) and {_PLACE} it (?:down )?(?:on|in) the (?P<dest>.+)$"),
    # "move the <target> <relation clause> onto the <destination>"
    re.compile(rf"^move the (?P<target>.+?) {_CLAUSE} on the (?P<dest>.+)$"),
    # LIBERO-Goal style: "put the <target> on the <destination>"
    re.compile(rf"^(?:{_PLACE}|move) the (?P<target>.+?) (?:down )?(?:on|in) the (?P<dest>.+)$"),
]
# paraphrase normalization (Section 6.4 wordings) before matching; order matters
_NORMALIZE = [
    (r"\b(?:sitting|located|that is|which is) ", ""),
    (r"\b(?:beside|near)\b", "next to"),
    (r"\bthe box of cookies\b", "the cookie box"),
    (r"\b(?:onto|on top of)\b", "on"),
    (r"\b(?:into|inside)\b", "in"),
]


@dataclass
class ParsedInstruction:
    target: str                    # "black bowl"
    destination: str               # "plate"
    relation: str | None = None    # "between" | "next to" | "on" | "in" | "center" | None
    anchors: list[str] = field(default_factory=list)   # nouns the relation refers to

    @property
    def nouns(self) -> list[str]:
        return list(dict.fromkeys([self.target, self.destination, *self.anchors]))


def parse_instruction(text: str) -> ParsedInstruction:
    s = " ".join(text.lower().strip().rstrip(".").split())
    for pat, rep in _NORMALIZE:
        s = re.sub(pat, rep, s)
    for pat in _PATTERNS:
        m = pat.match(s)
        if not m:
            continue
        p = ParsedInstruction(target=m["target"], destination=m["dest"])
        clause = m.groupdict().get("clause")
        if clause:
            p.relation, p.anchors = _parse_clause(clause)
        return p
    raise ValueError(f"can't parse instruction: {text!r}")


def _parse_clause(clause: str) -> tuple[str, list[str]]:
    if clause.startswith("between "):
        a, b = re.split(r" and ", clause[len("between "):], maxsplit=1)
        return "between", [_strip_article(a), _strip_article(b)]
    if clause.startswith("from ") and "center" in clause:
        return "center", []
    for rel in ("next to", "on", "in", "from"):
        if clause.startswith(rel + " "):
            noun = _strip_article(clause[len(rel) + 1:])
            # "the top drawer of the wooden cabinet" -> the object is the cabinet
            noun = noun.split(" of ")[-1] if " of " in noun else noun
            return rel, [_strip_article(noun)]
    raise ValueError(f"unknown relation clause: {clause!r}")


def _strip_article(s: str) -> str:
    return re.sub(r"^(the|a|an) ", "", s.strip())


# ----------------------------------------------------------------------------- geometry
def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def nms(boxes: np.ndarray, scores: np.ndarray, iou: float = NMS_IOU) -> list[int]:
    keep = []
    for i in np.argsort(-scores):
        if all(box_iou(boxes[i], boxes[j]) < iou for j in keep):
            keep.append(int(i))
    return keep


def matches(label: str, noun: str) -> bool:
    """Detector phrase vs. instruction noun: the noun's head word must appear in the label."""
    return noun.split()[-1] in label.split()


def resolve_target(parsed: ParsedInstruction, cand: list[int], centers: np.ndarray, scores: np.ndarray,
                   anchor_idx: dict[str, int | None]) -> int:
    """Pick the target among same-class candidates using the spatial relation.

    centers: (N, 3) robot-frame positions of all detections; anchor_idx: noun -> detection index.
    Distances are horizontal (x, y): "between"/"next to"/"on" are about table layout.
    """
    if len(cand) == 1 or parsed.relation is None:
        return max(cand, key=lambda i: scores[i])
    xy = centers[:, :2]
    anchors = [anchor_idx.get(a) for a in parsed.anchors]
    if parsed.relation == "center":
        mid = xy.mean(axis=0)
        return min(cand, key=lambda i: np.linalg.norm(xy[i] - mid))
    if any(a is None for a in anchors):          # anchor not detected: fall back to detector score
        return max(cand, key=lambda i: scores[i])
    if parsed.relation == "between":
        mid = (xy[anchors[0]] + xy[anchors[1]]) / 2
        return min(cand, key=lambda i: np.linalg.norm(xy[i] - mid))
    a = anchors[0]
    others = [i for i in cand if i != a]
    if parsed.relation == "on":                  # stacked on the anchor: close in xy and higher up
        above = [i for i in others if centers[i, 2] > centers[a, 2]] or others
        return min(above, key=lambda i: np.linalg.norm(xy[i] - xy[a]))
    return min(others, key=lambda i: np.linalg.norm(xy[i] - xy[a]))   # "next to", "in", "from"


# ----------------------------------------------------------------------------- models
@dataclass
class PerceptionResult:
    parsed: ParsedInstruction
    boxes: np.ndarray            # (N, 4) x0, y0, x1, y1 pixels
    labels: list[str]
    scores: np.ndarray
    target_idx: int | None
    dest_idx: int | None
    target_mask: np.ndarray      # (H, W) bool, all False if not found
    dest_mask: np.ndarray
    ms: dict                     # timing per stage


class InstructionPerception:
    """Grounding DINO + spatial resolution + SAM. One call per image."""

    def __init__(self, paths: dict, device: str | None = None, detector_id: str = DETECTOR_ID, sam_id: str = SAM_ID):
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor, SamModel, SamProcessor

        from groundedvla.policies.registry import resolve_weights

        # into <models>/<name> like every other model (no HF cache symlinks: they fail on Windows)
        det_path = str(resolve_weights(detector_id, detector_id.split("/")[-1], paths))
        sam_path = str(resolve_weights(sam_id, sam_id.split("/")[-1], paths))
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.det_proc = AutoProcessor.from_pretrained(det_path)
        self.det = AutoModelForZeroShotObjectDetection.from_pretrained(det_path).to(self.device).eval()
        self.sam_proc = SamProcessor.from_pretrained(sam_path)
        self.sam = SamModel.from_pretrained(sam_path).to(self.device).eval()

    def detect(self, rgb: np.ndarray, nouns: list[str]):
        from PIL import Image

        image = Image.fromarray(rgb)
        text = ". ".join(nouns) + "."
        inputs = self.det_proc(images=image, text=text, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            out = self.det(**inputs)
        res = self.det_proc.post_process_grounded_object_detection(
            out, inputs.input_ids, threshold=BOX_THRESHOLD, text_threshold=TEXT_THRESHOLD,
            target_sizes=[rgb.shape[:2]])[0]
        labels = res.get("text_labels", res.get("labels"))
        return res["boxes"].cpu().numpy(), res["scores"].cpu().numpy(), [str(l) for l in labels]

    def segment(self, rgb: np.ndarray, boxes: list[np.ndarray]) -> list[np.ndarray]:
        from PIL import Image

        if not boxes:
            return []
        inputs = self.sam_proc(Image.fromarray(rgb), input_boxes=[[b.tolist() for b in boxes]],
                               return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            out = self.sam(**inputs)
        masks = self.sam_proc.image_processor.post_process_masks(
            out.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu())[0]
        best = out.iou_scores[0].argmax(-1).cpu()                      # best of SAM's 3 masks per box
        return [masks[i, best[i]].numpy().astype(bool) for i in range(len(boxes))]

    def __call__(self, rgb: np.ndarray, instruction: str, centers_fn) -> PerceptionResult:
        """centers_fn(boxes) -> (N, 3) robot-frame centers of the boxes (uses depth + calibration)."""
        ms, t = {}, time.perf_counter()
        parsed = parse_instruction(instruction)
        boxes, scores, labels = self.detect(rgb, parsed.nouns)
        ms["detect"] = (time.perf_counter() - t) * 1e3
        keep = nms(boxes, scores) if len(boxes) else []
        boxes, scores, labels = boxes[keep], scores[keep], [labels[i] for i in keep]
        h, w = rgb.shape[:2]
        empty = np.zeros((h, w), bool)
        if not len(boxes):
            return PerceptionResult(parsed, boxes, labels, scores, None, None, empty, empty.copy(), ms)

        t = time.perf_counter()
        centers = centers_fn(boxes)
        best_of = {}
        for noun in parsed.nouns:
            idx = [i for i, l in enumerate(labels) if matches(l, noun)]
            best_of[noun] = max(idx, key=lambda i: scores[i]) if idx else None
        cand = [i for i, l in enumerate(labels) if matches(l, parsed.target)]
        target = resolve_target(parsed, cand, centers, scores, best_of) if cand else None
        dest_cand = [i for i, l in enumerate(labels) if matches(l, parsed.destination) and i != target]
        dest = max(dest_cand, key=lambda i: scores[i]) if dest_cand else None
        ms["resolve"] = (time.perf_counter() - t) * 1e3

        t = time.perf_counter()
        chosen = [i for i in (target, dest) if i is not None]
        masks = dict(zip(chosen, self.segment(rgb, [boxes[i] for i in chosen])))
        ms["segment"] = (time.perf_counter() - t) * 1e3
        return PerceptionResult(parsed, boxes, labels, scores, target, dest,
                                masks.get(target, empty), masks.get(dest, empty.copy()), ms)
