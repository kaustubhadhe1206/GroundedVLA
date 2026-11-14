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

DETECTOR_ID = "IDEA-Research/grounding-dino-base"   # tiny confused bowls and plates on these renders
SAM_ID = "facebook/sam-vit-base"
MIN_SCORE = 0.15          # a box is kept if it scores at least this for some noun
UNIQUE_SCORE = 0.2        # destination / anchors: their best box must score at least this
TARGET_SCORE = 0.15       # target candidates: boxes scoring at least this for the target noun
NMS_IOU = 0.5
MAX_BOX_FRACTION = 0.08   # largest real tabletop object (plate) is ~3% of the image; cabinet 17%, robot 13%
# What the detector is asked for, per instruction noun. Open-vocabulary detectors know common visual
# words better than LIBERO's names ("ramekin" landed on the robot arm, the ramekin itself was called a
# black bowl). The instruction noun is kept for the relation logic; only the detector prompt changes.
# Probed Nov 10 on base: "cookie box" scored the cabinet 0.58 and the real box 0.15; "snack box" found it
# in 3/3 demos (0.29-0.35).
DETECTOR_PROMPTS = {"ramekin": "small metal cup", "black bowl": "bowl", "cookie box": "snack box",
                    "wooden cabinet": "cabinet", "top drawer": "drawer"}
# Large fixtures may exceed MAX_BOX_FRACTION when the instruction names them (T5-T7: the cabinet).
FIXTURE_PROMPTS = ("cabinet", "drawer")
MAX_FIXTURE_FRACTION = 0.4
# Decoy phrases: things always in view that would otherwise be matched to a noun; their boxes are dropped.
DECOY_PROMPTS = ("robot arm", "cabinet", "stove")   # ignored when the instruction names them

_CLAUSE = r"(?P<clause>(?:between|next to|on|in|from) .+?)"
_PICK = r"(?:pick up|grab|take|move|lift|get|bring)"
_PLACE = r"(?:place|put|set|drop|stick|leave)"
_PATTERNS = [
    # LIBERO-Spatial style: "pick up the <target> <relation clause> and place it on the <destination>"
    re.compile(rf"^{_PICK} the (?P<target>.+?) {_CLAUSE} and {_PLACE} it (?:down )?(?:on|in) the (?P<dest>.+)$"),
    re.compile(rf"^{_PICK} the (?P<target>.+?) and {_PLACE} it (?:down )?(?:on|in) the (?P<dest>.+)$"),
    # "move the <target> <relation clause> onto the <destination>"
    re.compile(rf"^move the (?P<target>.+?) {_CLAUSE} on the (?P<dest>.+)$"),
    # LIBERO-Goal style: "put the <target> on the <destination>"
    re.compile(rf"^(?:{_PLACE}|move) the (?P<target>.+?) (?:down )?(?:on|in) the (?P<dest>.+)$"),
    # "open the <destination> and put the <target> inside (it)"
    re.compile(rf"^open the (?P<dest>.+?) and {_PLACE} the (?P<target>.+?) in(?: it)?$"),
]
# paraphrase normalization (Section 6.4 wordings) before matching; order matters
_NORMALIZE = [
    (r"\b(?:sitting|located|that is|which is) ", ""),
    (r"\bpull open\b", "open"),
    (r"\bout of\b", "in"),                          # "take the bowl out of the drawer" names where it is
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
    s = " ".join(re.sub(r"[^a-z ]", " ", text.lower()).split())
    for pat, rep in _NORMALIZE:
        s = re.sub(pat, rep, s)
    s = _add_articles(s)
    for pat in _PATTERNS:
        m = pat.match(s)
        if not m:
            continue
        # "the top drawer of the cabinet" -> "top drawer": the destination is the specific part
        p = ParsedInstruction(target=m["target"], destination=m["dest"].split(" of ")[0])
        clause = m.groupdict().get("clause")
        if not clause:   # "put the black bowl between the plate and the ramekin on the plate"
            split = re.match(r"^(?P<t>.+?) (?P<c>(?:between|next to|from) .+)$", p.target)
            if split:
                p.target, clause = split["t"], split["c"]
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


def _add_articles(s: str) -> str:
    """Typed prompts often drop "the" ("put bowl on stove"); the patterns expect it after verbs and
    prepositions."""
    words = s.split()
    out = []
    for i, w in enumerate(words):
        out.append(w)
        nxt = words[i + 1] if i + 1 < len(words) else None
        if nxt and nxt not in ("the", "a", "an", "it", "up", "down", "open", "and", "of", "to") \
                and (w in ("on", "in", "to", "of", "between", "and", "up", "open", "put", "place", "set", "drop",
                           "stick", "leave", "grab", "take", "move", "lift", "get", "bring")) \
                and not (w == "and" and nxt in ("place", "put", "set", "drop", "stick", "leave")) \
                and not (w in ("of",) and nxt == "cookies"):
            out.append("the")
    return " ".join(out)


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
    """Detector label vs. instruction noun. Labels are single nouns from the instruction (detect),
    so this is equality, with a head-word fallback ("bowl" for "black bowl")."""
    return label == noun or noun.split()[-1] == label.split()[-1]


def assign(S: np.ndarray, nouns: list[str], parsed: ParsedInstruction) -> tuple[dict, list[int]]:
    """Boxes for the nouns from the (boxes x nouns) score matrix.

    The destination and the relation's anchors name single objects: each takes the best-scoring free
    box, most confident noun first, one box per noun. The target may have look-alikes (two black
    bowls), so every remaining box that scores TARGET_SCORE for it is a candidate; resolve_target
    picks among them. Returns ({noun: box or None}, target candidates).
    """
    used, unique = set(), {}
    singles = [n for n in dict.fromkeys([parsed.destination, *parsed.anchors]) if n != parsed.target]
    for n in sorted(singles, key=lambda n: -S[:, nouns.index(n)].max()):
        col = S[:, nouns.index(n)]
        free = [i for i in np.argsort(-col) if i not in used and col[i] >= UNIQUE_SCORE]
        unique[n] = int(free[0]) if free else None
        if free:
            used.add(int(free[0]))
    ti = nouns.index(parsed.target)
    cand = [i for i in range(len(S)) if i not in used and S[i, ti] >= TARGET_SCORE]
    if not cand:                                   # weak target evidence: best remaining box
        rest = [i for i in range(len(S)) if i not in used]
        cand = [max(rest, key=lambda i: S[i, ti])] if rest else []
    return unique, cand


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

    def detect(self, rgb: np.ndarray, nouns: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """Candidate boxes (N, 4) and their score for every noun (N, len(nouns)).

        Grounding DINO scores every box against every text token. Its own post-processing labels a
        box with all tokens above a threshold, merging phrases ("black bowl ramekin"), and on these
        small renders it often rates a bowl higher as "plate" than as "bowl". So no label is fixed
        here: assign() decides from the full score matrix. Dropped: boxes larger than
        MAX_BOX_FRACTION of the image (robot, cabinet; our tabletop objects are small), boxes that
        best match a decoy phrase (the robot arm), weak boxes, and duplicates (class-agnostic NMS).
        """
        from PIL import Image

        prompts = list(dict.fromkeys([DETECTOR_PROMPTS.get(n, n) for n in nouns] + list(DECOY_PROMPTS)))
        noun_of = {DETECTOR_PROMPTS.get(n, n): n for n in nouns}          # prompt -> instruction noun
        inputs = self.det_proc(images=Image.fromarray(rgb), text=". ".join(prompts) + ".",
                               return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            out = self.det(**inputs)
        ids = inputs.input_ids[0].tolist()
        spans = []                                       # token positions of each prompt phrase
        for p in prompts:
            toks = self.det_proc.tokenizer(p, add_special_tokens=False).input_ids
            start = next(i for i in range(len(ids) - len(toks) + 1) if ids[i:i + len(toks)] == toks)
            spans.append(list(range(start, start + len(toks))))
        probs = out.logits[0].sigmoid().cpu().numpy()                                # (queries, tokens)
        phrase = np.stack([probs[:, s].mean(axis=1) for s in spans], axis=1)         # (queries, prompts)
        h, w = rgb.shape[:2]
        cx, cy, bw, bh = out.pred_boxes[0].cpu().numpy().T
        boxes = np.stack([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h], axis=1)
        noun_cols = [prompts.index(DETECTOR_PROMPTS.get(n, n)) for n in nouns]
        S = phrase[:, noun_cols]                                                     # (queries, nouns)
        decoy = np.array([prompts[i] not in noun_of for i in phrase.argmax(axis=1)])
        fixture = np.array([prompts[i] in FIXTURE_PROMPTS for i in phrase.argmax(axis=1)])
        size_ok = (bw * bh < MAX_BOX_FRACTION) | (fixture & (bw * bh < MAX_FIXTURE_FRACTION))
        keep = (S.max(axis=1) > MIN_SCORE) & size_ok & ~decoy
        boxes, S = boxes[keep], S[keep]
        k = nms(boxes, S.max(axis=1)) if len(boxes) else []
        return boxes[k], S[k]

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
        nouns = parsed.nouns
        boxes, S = self.detect(rgb, nouns)
        ms["detect"] = (time.perf_counter() - t) * 1e3
        h, w = rgb.shape[:2]
        empty = np.zeros((h, w), bool)
        if not len(boxes):
            return PerceptionResult(parsed, boxes, [], np.zeros(0), None, None, empty, empty.copy(), ms)

        t = time.perf_counter()
        centers = centers_fn(boxes)
        unique, cand = assign(S, nouns, parsed)
        ti = nouns.index(parsed.target)
        target = resolve_target(parsed, cand, centers, S[:, ti], unique) if cand else None
        dest = unique.get(parsed.destination)
        labels = [next((n for n, i in unique.items() if i == b), parsed.target if b in cand else "")
                  for b in range(len(boxes))]
        scores = np.array([S[b, nouns.index(lab)] if lab else S[b].max() for b, lab in enumerate(labels)])
        ms["resolve"] = (time.perf_counter() - t) * 1e3

        t = time.perf_counter()
        chosen = [i for i in (target, dest) if i is not None]
        masks = dict(zip(chosen, self.segment(rgb, [boxes[i] for i in chosen])))
        ms["segment"] = (time.perf_counter() - t) * 1e3
        return PerceptionResult(parsed, boxes, labels, scores, target, dest,
                                masks.get(target, empty), masks.get(dest, empty.copy()), ms)
