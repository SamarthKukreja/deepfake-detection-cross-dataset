"""
app.py — Deepfake Detection Demo (Streamlit)
==============================================
Runs ALL five trained backbone models on an uploaded video simultaneously.
Faces are detected once per frame with MTCNN; every backbone scores each
crop; an ensemble (mean) verdict is computed.  Results are shown in a
comparison panel styled after commercial deepfake-detection tools.

Usage
-----
    cd ml
    streamlit run app.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

try:
    import timm
except ImportError:
    st.error("timm not installed.  Run: pip install timm")
    st.stop()

try:
    from facenet_pytorch import MTCNN
except ImportError:
    st.error("facenet-pytorch not installed.  Run: pip install facenet-pytorch")
    st.stop()

try:
    import yt_dlp
    _YTDLP_AVAILABLE = True
except ImportError:
    _YTDLP_AVAILABLE = False

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────

RESULTS_ROOT = _HERE.parent / "results_bundle"
INPUT_SIZE   = 224
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

MODEL_DISPLAY: Dict[str, str] = {
    "EfficientNet-B4":  "efficientnet_b4",
    "Xception":         "xception",
    "ResNet-50":        "resnet50",
    "ConvNeXt-Tiny":    "convnext_tiny",
    "ViT-Base":         "vit_base_patch16_224",
}
_EFFNET_DEFAULT = "efficientnet_b4"

# ── Sensitivity presets ───────────────────────────────────────────────────────
# threshold  – per-model score cutoff to count as one "vote"
# min_votes  – how many model votes needed to declare DEEPFAKE DETECTED
SENSITIVITY_PRESETS = {
    "Normal": {
        "threshold": 0.50,
        "min_votes": 3,
        "desc": "3 or more models must flag → DEEPFAKE DETECTED.",
    },
    "Strict": {
        "threshold": 0.50,
        "min_votes": 2,
        "desc": "2 or more models must flag → DEEPFAKE DETECTED.",
    },
    "Very Strict": {
        "threshold": 0.50,
        "min_votes": 1,
        "desc": "Even 1 model flagging → DEEPFAKE DETECTED.",
    },
}

EXP_DISPLAY: Dict[str, str] = {
    "Exp 1 — FF++ → FF++ (in-distribution)":    "exp1_ffpp_to_ffpp",
    "Exp 2 — FF++ → Celeb-DF (cross-dataset)":  "exp2_ffpp_to_celebdf",
    "Exp 3 — Celeb-DF → FF++ (cross-dataset)":  "exp3_celebdf_to_ffpp",
    "Exp 4 — Mixed → DFD (recommended)":        "exp4_mixed_to_dfd",
}

EXP_SHORT: Dict[str, str] = {
    "exp1_ffpp_to_ffpp":    "Exp1",
    "exp2_ffpp_to_celebdf": "Exp2",
    "exp3_celebdf_to_ffpp": "Exp3",
    "exp4_mixed_to_dfd":    "Exp4",
}


def _run_dir(exp_name: str, model_name: str) -> Path:
    if model_name == _EFFNET_DEFAULT:
        return RESULTS_ROOT / exp_name
    return RESULTS_ROOT / f"{exp_name}__{model_name}__full"


def _ckpt_path(exp_name: str, model_name: str) -> Path:
    return _run_dir(exp_name, model_name) / "best_model.pth"


def _read_metrics(exp_name: str, model_name: str) -> Optional[Dict]:
    p = _run_dir(exp_name, model_name) / "metrics.json"
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text())
        return d.get("video_level", d)
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────────────────────
# MODEL LOADING
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def _load_single_model(ckpt_path: str, model_name: str, device_str: str) -> nn.Module:
    device = torch.device(device_str)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    try:
        model = timm.create_model(model_name, pretrained=False, num_classes=1, drop_rate=0.3)
    except TypeError:
        model = timm.create_model(model_name, pretrained=False, num_classes=1)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model


def load_all_models(exp_name: str, device_str: str) -> Dict[str, nn.Module]:
    """Load every available backbone for *exp_name*.  Returns {display_name: model}."""
    loaded: Dict[str, nn.Module] = {}
    for display, mn in MODEL_DISPLAY.items():
        ckpt = _ckpt_path(exp_name, mn)
        if ckpt.exists():
            loaded[display] = _load_single_model(str(ckpt), mn, device_str)
    return loaded


@st.cache_resource(show_spinner="Initialising face detector…")
def load_detector(device_str: str) -> MTCNN:
    return MTCNN(
        keep_all=True,
        device=torch.device(device_str),
        post_process=False,
        select_largest=False,
        min_face_size=40,
    )


# ──────────────────────────────────────────────────────────────────────────────
# INFERENCE
# ──────────────────────────────────────────────────────────────────────────────

_TFM = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
])


def score_crops(
    model: nn.Module,
    crops: List[Image.Image],
    device: torch.device,
) -> List[float]:
    if not crops:
        return []
    tensors = torch.stack([_TFM(c.convert("RGB")) for c in crops]).to(device)
    with torch.no_grad():
        s = torch.sigmoid(model(tensors).squeeze(1)).cpu().tolist()
    return s if isinstance(s, list) else [s]


def detect_faces(
    detector: MTCNN,
    frame_bgr: np.ndarray,
    margin: float = 0.20,
) -> Tuple[List[Tuple[int, int, int, int]], List[Image.Image]]:
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(frame_rgb)
    h, w = frame_bgr.shape[:2]
    boxes_raw, probs = detector.detect(pil)
    if boxes_raw is None:
        return [], []
    boxes, crops = [], []
    for box, prob in zip(boxes_raw, probs):
        if prob is None or prob < 0.90:
            continue
        x1, y1, x2, y2 = box
        bw, bh = x2 - x1, y2 - y1
        mx, my = int(bw * margin), int(bh * margin)
        cx1, cy1 = max(0, int(x1) - mx), max(0, int(y1) - my)
        cx2, cy2 = min(w, int(x2) + mx), min(h, int(y2) + my)
        boxes.append((cx1, cy1, cx2, cy2))
        crops.append(pil.crop((cx1, cy1, cx2, cy2)))
    return boxes, crops


def annotate_frame(
    frame_bgr: np.ndarray,
    boxes: List[Tuple[int, int, int, int]],
    ensemble_scores: List[float],
    threshold: float,
) -> np.ndarray:
    out = frame_bgr.copy()
    for (x1, y1, x2, y2), score in zip(boxes, ensemble_scores):
        is_fake = score >= threshold
        color = (30, 30, 220) if is_fake else (30, 180, 30)
        label = f"{'FAKE' if is_fake else 'REAL'}  {score:.0%}"
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.55, 1)
        cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(out, label, (x1 + 2, y1 - 3),
                    cv2.FONT_HERSHEY_DUPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# VIDEO PROCESSING — all models in one pass
# ──────────────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────────────
# URL DOWNLOAD (yt-dlp)
# ──────────────────────────────────────────────────────────────────────────────

# Sites whose URLs we recognise for a friendly icon/label
_SITE_ICONS: List[Tuple[str, str, str]] = [
    ("youtube.com",  "YouTube",   "🎬"),
    ("youtu.be",     "YouTube",   "🎬"),
    ("instagram.com","Instagram", "📸"),
    ("tiktok.com",   "TikTok",    "🎵"),
    ("twitter.com",  "Twitter/X", "🐦"),
    ("x.com",        "Twitter/X", "🐦"),
    ("facebook.com", "Facebook",  "👥"),
    ("fb.watch",     "Facebook",  "👥"),
    ("twitch.tv",    "Twitch",    "🎮"),
    ("vimeo.com",    "Vimeo",     "▶️"),
    ("reddit.com",   "Reddit",    "📰"),
    ("dailymotion.com","Dailymotion","▶️"),
]


def _detect_site(url: str) -> str:
    url_l = url.lower()
    for domain, name, icon in _SITE_ICONS:
        if domain in url_l:
            return f"{icon} {name}"
    return "🔗 Direct link"


def resolve_stream_url(url: str) -> Tuple[str, str, Optional[str]]:
    """
    Use yt-dlp to extract the direct CDN stream URL — NO download.

    OpenCV's VideoCapture can read frames directly from the returned URL,
    so the video never touches disk.

    Returns
    -------
    (stream_url, title, thumbnail_url_or_None)
    """
    if not _YTDLP_AVAILABLE:
        raise RuntimeError("yt-dlp not installed.  Run: pip install yt-dlp")

    ydl_opts = {
        # Video-only ≤ 480 p mp4; smaller = faster frame-by-frame reads
        "format": (
            "bestvideo[height<=480][ext=mp4]"
            "/best[height<=480][ext=mp4]"
            "/best[height<=480]"
            "/best"
        ),
        "quiet": True,
        "no_warnings": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)   # ← key: no download

    title     = info.get("title", "video")
    thumbnail = info.get("thumbnail")

    # Extract the raw CDN URL from the info dict
    if "url" in info:
        stream_url = info["url"]
    elif "formats" in info:
        fmts = info["formats"]
        # prefer mp4 video stream ≤ 480 p
        candidates = [
            f for f in fmts
            if f.get("vcodec", "none") != "none"
            and f.get("ext") == "mp4"
            and (f.get("height") or 9999) <= 480
        ]
        if not candidates:
            candidates = [f for f in fmts if f.get("vcodec", "none") != "none"]
        if not candidates:
            candidates = fmts  # last resort
        best = sorted(candidates,
                      key=lambda f: (f.get("height") or 0),
                      reverse=True)[0]
        stream_url = best["url"]
    else:
        raise ValueError("yt-dlp returned no usable stream URL for this link.")

    return stream_url, title, thumbnail


def get_video_metadata(path: str) -> Dict:
    cap = cv2.VideoCapture(path)
    fps    = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    duration = frames / fps if fps > 0 else 0
    return {
        "duration_s": duration,
        "resolution": f"{width} × {height}",
        "fps": f"{fps:.1f}" if fps > 0 else "—",
        "frames": frames,
    }


def process_video_all_models(
    video_path: str,
    models: Dict[str, nn.Module],
    detector: MTCNN,
    device: torch.device,
    threshold: float,
    frame_step: int,
    max_frames: int,
    progress_bar,
) -> Tuple[str, Dict[str, List[float]], List[np.ndarray]]:
    """
    Single-pass video processing:
      - Detect faces once per frame.
      - Score each crop with every loaded model.
      - Annotate with ensemble (mean) score.

    Returns
    -------
    out_path          : annotated output .mp4
    per_model_scores  : {display_name: [all face scores across video]}
    key_frames        : list of annotated RGB frames for preview
    """
    cap = cv2.VideoCapture(video_path)
    fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_fd, out_path = tempfile.mkstemp(suffix=".mp4")
    os.close(out_fd)
    writer = cv2.VideoWriter(
        out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    per_model_scores: Dict[str, List[float]] = {name: [] for name in models}
    per_model_frame_scores: Dict[str, List[float]] = {name: [] for name in models}
    key_frames: List[np.ndarray] = []
    key_frame_interval = max(1, max_frames // 6)

    frame_idx = sampled_idx = 0
    model_names = list(models.keys())
    n_models = len(model_names)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_step == 0 and sampled_idx < max_frames:
            boxes, crops = detect_faces(detector, frame)

            if crops:
                # Score with every model
                all_model_scores: Dict[str, List[float]] = {}
                for name in model_names:
                    s = score_crops(models[name], crops, device)
                    all_model_scores[name] = s
                    per_model_scores[name].extend(s)
                    per_model_frame_scores[name].append(float(np.mean(s)))

                # Mean ensemble per face for bounding-box colour
                ensemble_per_face = [
                    float(np.mean([all_model_scores[n][i] for n in model_names]))
                    for i in range(len(crops))
                ]
                annotated = annotate_frame(frame, boxes, ensemble_per_face, threshold)
            else:
                annotated = frame.copy()

            writer.write(annotated)

            if sampled_idx % key_frame_interval == 0:
                key_frames.append(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))

            sampled_idx += 1
            total_faces = sum(len(v) for v in per_model_scores.values()) // max(n_models, 1)
            progress_bar.progress(
                min(sampled_idx / max_frames, 1.0),
                text=f"Frame {sampled_idx}/{max_frames} · {total_faces} faces scored across {n_models} models",
            )
        else:
            writer.write(frame)

        frame_idx += 1

    cap.release()
    writer.release()
    return out_path, per_model_scores, per_model_frame_scores, key_frames


# ──────────────────────────────────────────────────────────────────────────────
# UI COMPONENTS
# ──────────────────────────────────────────────────────────────────────────────

def _model_label(score: float, threshold: float) -> Tuple[str, str]:
    """Colour-coded label for a single model's score."""
    if score >= threshold:
        return f"DEEPFAKE DETECTED ({score:.0%})", "#EF4444"
    elif score >= threshold * 0.80:
        return f"SUSPICIOUS ({score:.0%})", "#F59E0B"
    else:
        return f"NO DEEPFAKE DETECTED ({score:.0%})", "#22C55E"


def _vote_verdict(votes: int, n_models: int, min_votes: int) -> Tuple[str, str]:
    """Final verdict: DEEPFAKE if enough models voted, otherwise REAL."""
    if votes >= min_votes:
        return "DEEPFAKE DETECTED", "#EF4444"
    else:
        return "NO DEEPFAKE DETECTED", "#22C55E"


def temporal_fusion(mean_score: float, frame_scores: List[float], alpha: float = 0.7) -> float:
    """Blend spatial mean score with temporal variance — no retraining needed.

    High frame-to-frame score variance indicates inconsistent deepfake blending:
    some frames expose the swap, others blend well.  Even if mean score is low,
    high variance is a tell.

    fused = alpha * mean_score + (1 - alpha) * temporal_std
    """
    if len(frame_scores) < 2:
        return mean_score
    temporal_std = float(np.std(frame_scores))
    return float(min(1.0, alpha * mean_score + (1.0 - alpha) * temporal_std))


def render_results_panel(
    per_model_scores: Dict[str, List[float]],
    per_model_frame_scores: Dict[str, List[float]],
    meta: Dict,
    preset: Dict,
    exp_name: str,
    all_metrics_cache: Dict,
) -> None:
    """Render the three-column results panel (Model / Video / AUC)."""

    threshold  = preset["threshold"]
    min_votes  = preset["min_votes"]

    n_models_running = len(per_model_scores)
    min_votes = min(min_votes, n_models_running)

    # Per-model mean score (spatial)
    model_results: Dict[str, float] = {
        name: float(np.mean(scores)) if scores else 0.0
        for name, scores in per_model_scores.items()
    }

    # Temporal std per model (frame-to-frame variance)
    model_temporal_std: Dict[str, float] = {
        name: float(np.std(fscores)) if len(fscores) > 1 else 0.0
        for name, fscores in per_model_frame_scores.items()
    }

    # Temporal-fused score = 0.7 * mean + 0.3 * temporal_std
    model_fused: Dict[str, float] = {
        name: temporal_fusion(model_results[name], per_model_frame_scores.get(name, []))
        for name in model_results
    }

    n_models = n_models_running

    # Vote uses fused score
    votes = sum(1 for s in model_fused.values() if s >= threshold)

    # ── Verdict banner ────────────────────────────────────────────────────────
    verdict, color = _vote_verdict(votes, n_models, min_votes)
    st.markdown(
        f"""
        <div style="
            background:{color}22;border:2px solid {color};
            border-radius:12px;padding:14px 24px;text-align:center;margin:8px 0 16px;
        ">
            <span style="font-size:1.8rem;font-weight:700;color:{color}">{verdict}</span><br>
            <span style="color:#aaa;font-size:0.95rem">
                <b style="color:{color}">{votes}</b> / {n_models} models voted deepfake
                &nbsp;·&nbsp; need <b>{min_votes}</b> to confirm
                &nbsp;·&nbsp; spatial + temporal fusion
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Three-column panel ────────────────────────────────────────────────────
    col_model, col_video, col_auc = st.columns([2, 1.6, 1.4])

    with col_model:
        st.markdown("#### Model Results")
        for display_name, fused_score in model_fused.items():
            mean_score = model_results[display_name]
            tstd = model_temporal_std.get(display_name, 0.0)
            label, mcolor = _model_label(fused_score, threshold)
            flag = " 🚩" if fused_score >= threshold else ""
            st.markdown(
                f'<div style="margin-bottom:8px">'
                f'<span style="font-weight:600;text-decoration:underline">{display_name}:</span>&nbsp;'
                f'<span style="color:{mcolor};font-weight:600">{label}{flag}</span><br>'
                f'<span style="color:#777;font-size:0.82rem">'
                f'spatial {mean_score:.0%} · temporal σ {tstd:.2f} · fused {fused_score:.0%}'
                f'</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
        # Vote summary row
        v_label, v_color = _vote_verdict(votes, n_models, min_votes)
        st.markdown(
            f'<div style="margin-top:8px;border-top:1px solid #444;padding-top:8px">'
            f'<span style="font-weight:700">Verdict ({votes}/{n_models} votes):</span>&nbsp;'
            f'<span style="color:{v_color};font-weight:700">{v_label}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with col_video:
        st.markdown("#### Video Info")
        dur = meta.get("duration_s", 0)
        dur_str = f"{int(dur)} sec" if dur < 60 else f"{int(dur//60)}m {int(dur%60)}s"
        rows = [
            ("Duration",    dur_str),
            ("Resolution",  meta.get("resolution", "—")),
            ("Frame Rate",  meta.get("fps", "—")),
            ("Frames",      str(meta.get("frames", "—"))),
        ]
        for label, value in rows:
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;'
                f'border-bottom:1px solid #2d2d2d;padding:4px 0;">'
                f'<span style="color:#aaa">{label}</span>'
                f'<span style="color:#e5e5e5">{value}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

    with col_auc:
        st.markdown("#### Checkpoint AUC")
        model_name_map = MODEL_DISPLAY
        for display_name in model_results:
            mn = model_name_map[display_name]
            m  = all_metrics_cache.get((exp_name, mn))
            auc_str = f"{m['auc']:.3f}" if m and "auc" in m else "—"
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;'
                f'border-bottom:1px solid #2d2d2d;padding:4px 0;">'
                f'<span style="color:#aaa">{display_name}</span>'
                f'<span style="color:#93c5fd">{auc_str}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )


@st.cache_data(show_spinner=False)
def _load_all_metrics_cache() -> Dict:
    out = {}
    for mn in MODEL_DISPLAY.values():
        for en in EXP_DISPLAY.values():
            p = _run_dir(en, mn) / "metrics.json"
            if p.exists():
                try:
                    d = json.loads(p.read_text())
                    out[(en, mn)] = d.get("video_level", d)
                except Exception:
                    pass
    return out


def _exp_option_label(display: str, exp_n: str, all_metrics: Dict) -> str:
    aucs = [
        all_metrics[(exp_n, mn)]["auc"]
        for mn in MODEL_DISPLAY.values()
        if (exp_n, mn) in all_metrics and "auc" in all_metrics[(exp_n, mn)]
    ]
    if aucs:
        avg = np.mean(aucs)
        return f"{display}  (avg AUC {avg:.3f})"
    return display


# ──────────────────────────────────────────────────────────────────────────────
# MAIN APP
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="Deepfake Detector",
        page_icon="🔍",
        layout="wide",
    )
    st.title("🔍 Deepfake Detection")
    st.caption(
        "Run a single backbone or all five models on an uploaded video — "
        "bounding boxes on faces, per-model verdicts and an ensemble score."
    )

    device_str   = "cuda" if torch.cuda.is_available() else "cpu"
    all_metrics  = _load_all_metrics_cache()

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Configuration")

        exp_options = list(EXP_DISPLAY.keys())
        exp_labels  = [
            _exp_option_label(d, EXP_DISPLAY[d], all_metrics) for d in exp_options
        ]
        exp_idx = st.selectbox(
            "Training checkpoint set",
            range(len(exp_options)),
            format_func=lambda i: exp_labels[i],
            index=3,          # default = Exp 4 (most general)
            help=(
                "Exp 4 (Mixed → DFD) is recommended for unknown real-world videos.\n\n"
                "Exp 1 = in-distribution (highest AUC, overfit to FF++).\n"
                "Exp 3 = near-chance cross-dataset — avoid for real-world use."
            ),
        )
        exp_name = EXP_DISPLAY[exp_options[exp_idx]]

        # ── Backbone selector (for single-model run) ──────────────────────────
        st.markdown("**Backbone (single-model run)**")
        backbone_options = list(MODEL_DISPLAY.keys())

        def _backbone_label(dn: str) -> str:
            mn = MODEL_DISPLAY[dn]
            m  = all_metrics.get((exp_name, mn))
            auc_str = f"  · AUC {m['auc']:.3f}" if m and "auc" in m else ""
            ok  = "✓" if _ckpt_path(exp_name, mn).exists() else "✗"
            return f"{ok} {dn}{auc_str}"

        backbone_idx = st.selectbox(
            "Backbone model",
            range(len(backbone_options)),
            format_func=lambda i: _backbone_label(backbone_options[i]),
            index=0,
            label_visibility="collapsed",
        )
        selected_backbone = backbone_options[backbone_idx]

        # AUC table for chosen experiment
        with st.expander("📊 AUC for selected experiment", expanded=True):
            _render_exp_auc_table(exp_name, all_metrics)

        st.divider()

        # ── Sensitivity preset ────────────────────────────────────────────────
        sensitivity_name = st.radio(
            "Detection sensitivity",
            list(SENSITIVITY_PRESETS.keys()),
            index=0,
            help=(
                "**Normal** — balanced, fewer false alarms.\n\n"
                "**Strict** — lower threshold + max-ensemble; one model flagging is enough.\n\n"
                "**Very Strict** — most aggressive; expect more false positives on real videos."
            ),
        )
        preset = SENSITIVITY_PRESETS[sensitivity_name]
        st.caption(preset["desc"])

        frame_step = st.slider(
            "Sample every N frames",
            min_value=1, max_value=30, value=5,
            help="Higher = faster; lower = more thorough.",
        )
        max_frames = st.slider(
            "Max frames to analyse",
            min_value=10, max_value=300, value=60,
        )

        st.divider()
        st.caption(f"Device: **{device_str.upper()}**")

        # Availability summary
        available = [
            dn for dn, mn in MODEL_DISPLAY.items()
            if _ckpt_path(exp_name, mn).exists()
        ]
        missing = [dn for dn in MODEL_DISPLAY if dn not in available]
        st.success(f"{len(available)} / {len(MODEL_DISPLAY)} checkpoints found")
        if missing:
            st.warning("Missing: " + ", ".join(missing))

    # ── Load models (with progress) ───────────────────────────────────────────
    with st.spinner(f"Loading {len(available)} models…"):
        models = load_all_models(exp_name, device_str)
    detector = load_detector(device_str)
    device   = torch.device(device_str)

    # ── Video source (upload or URL) ──────────────────────────────────────────
    tab_upload, tab_url = st.tabs(["📁 Upload File", "🔗 Video URL"])

    # video_source  → local file path (upload) OR original user URL (URL tab)
    # is_url        → True means resolve stream URL before processing
    video_source: Optional[str] = None
    is_url = False

    with tab_upload:
        uploaded = st.file_uploader(
            "Upload a video file",
            type=["mp4", "avi", "mov", "mkv"],
            help="Short clips (< 60 s) work best.",
        )
        if uploaded is not None:
            suffix = Path(uploaded.name).suffix or ".mp4"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded.read())
                video_source = tmp.name

    with tab_url:
        if not _YTDLP_AVAILABLE:
            st.warning(
                "yt-dlp not installed — URL streaming unavailable.\n\n"
                "Fix: `pip install yt-dlp`"
            )
        else:
            st.caption(
                "Paste a link from **YouTube, Instagram, TikTok, Twitter/X, "
                "Facebook, Vimeo, Reddit, Twitch, Dailymotion**, or any of the "
                "[1000+ sites yt-dlp supports]"
                "(https://github.com/yt-dlp/yt-dlp/blob/master/supportedsites.md).  "
                "The video streams directly — nothing is saved to disk."
            )
            video_url = st.text_input(
                "Video URL",
                placeholder="https://www.youtube.com/watch?v=...",
                label_visibility="collapsed",
            )
            if video_url:
                site = _detect_site(video_url)
                st.info(f"Detected: **{site}** — click Run below to start streaming")
                video_source = video_url
                is_url = True

    if video_source is None:
        st.info("Upload a video or paste a URL above to start analysis.")
        return

    # Preview (only possible for local files; URLs need resolving first)
    if not is_url:
        meta = get_video_metadata(video_source)
        st.video(video_source)
    else:
        meta = {}

    btn_col1, btn_col2 = st.columns(2)
    run_selected = btn_col1.button(
        f"▶ Run  {selected_backbone}",
        use_container_width=True,
        help="Run only the backbone chosen in the sidebar.",
    )
    run_all = btn_col2.button(
        "🚀 Run All Models",
        type="primary",
        use_container_width=True,
        help="Run all five backbones and show an ensemble verdict.",
    )

    if not run_selected and not run_all:
        if not is_url:
            try:
                os.unlink(video_source)
            except OSError:
                pass
        return

    # ── Resolve stream URL if needed ─────────────────────────────────────────
    actual_source = video_source
    if is_url:
        resolve_ph = st.empty()
        try:
            with st.spinner("Resolving stream URL (no download)…"):
                stream_url, vid_title, thumbnail = resolve_stream_url(video_source)
            actual_source = stream_url
            meta = get_video_metadata(stream_url)
            resolve_ph.success(
                f"Streaming **{vid_title}**  ·  "
                f"{meta.get('resolution','?')}  ·  "
                f"{meta.get('duration_s', 0):.0f} sec"
            )
            if thumbnail:
                st.image(thumbnail, width=320, caption=vid_title)
        except Exception as e:
            resolve_ph.error(f"Could not resolve stream URL: {e}")
            return

    # ── Select which models to run ────────────────────────────────────────────
    if run_selected:
        if selected_backbone not in models:
            st.error(f"Checkpoint for {selected_backbone} not found for this experiment.")
            return
        models_to_run = {selected_backbone: models[selected_backbone]}
        spinner_msg   = f"Running {selected_backbone}…"
    else:
        models_to_run = models
        spinner_msg   = f"Running {len(models)} models across video frames…"

    prog = st.progress(0.0, text="Starting…")
    with st.spinner(spinner_msg):
        out_path, per_model_scores, per_model_frame_scores, key_frames = process_video_all_models(
            actual_source, models_to_run, detector, device,
            preset["threshold"], frame_step, max_frames, prog,
        )
    prog.empty()

    # ── Results panel ─────────────────────────────────────────────────────────
    st.divider()
    render_results_panel(per_model_scores, per_model_frame_scores, meta, preset, exp_name, all_metrics)
    st.divider()

    # ── Annotated video ───────────────────────────────────────────────────────
    c_vid, c_frames = st.columns([1, 1])
    with c_vid:
        st.subheader("📹 Annotated Output")
        try:
            with open(out_path, "rb") as f:
                video_bytes = f.read()
            st.video(video_bytes)
            st.download_button(
                "⬇️ Download annotated video",
                data=video_bytes,
                file_name="deepfake_annotated.mp4",
                mime="video/mp4",
            )
        except Exception as e:
            st.warning(f"Could not render video: {e}")

    with c_frames:
        if key_frames:
            st.subheader("🖼️ Key Frames")
            sub_cols = st.columns(2)
            for i, kf in enumerate(key_frames[:6]):
                sub_cols[i % 2].image(kf, use_container_width=True)

    # ── Score timeline ────────────────────────────────────────────────────────
    total_faces = sum(len(v) for v in per_model_scores.values())
    if total_faces > 0:
        st.subheader("📊 Per-model Score Distribution")
        import altair as alt
        import pandas as pd

        rows = []
        for model_name_d, scores in per_model_scores.items():
            for s in scores:
                rows.append({"Model": model_name_d, "score": s})
        df = pd.DataFrame(rows)
        if not df.empty:
            chart = (
                alt.Chart(df)
                .mark_bar(opacity=0.75)
                .encode(
                    x=alt.X("score:Q", bin=alt.Bin(maxbins=20),
                             title="Deepfake score (0=real, 1=fake)"),
                    y=alt.Y("count():Q", title="# face crops"),
                    color=alt.Color("Model:N",
                                    scale=alt.Scale(scheme="tableau10")),
                    tooltip=["Model:N", "count():Q"],
                )
                .properties(height=220)
            )
            st.altair_chart(chart, use_container_width=True)

    # Cleanup temp files (skip if source was a URL)
    cleanup = [out_path]
    if not is_url:
        cleanup.append(video_source)
    for p in cleanup:
        try:
            os.unlink(p)
        except OSError:
            pass


# ──────────────────────────────────────────────────────────────────────────────
# HELPER: AUC TABLE
# ──────────────────────────────────────────────────────────────────────────────

def _render_exp_auc_table(exp_name: str, all_metrics: Dict) -> None:
    """Compact table: backbone → AUC / Acc / F1 for the chosen experiment."""
    import pandas as pd

    rows = []
    for display, mn in MODEL_DISPLAY.items():
        m = all_metrics.get((exp_name, mn))
        if m:
            rows.append({
                "Model":    display,
                "AUC":      f"{m.get('auc', 0):.3f}",
                "Accuracy": f"{m.get('accuracy', 0):.3f}",
                "F1":       f"{m.get('f1', 0):.3f}",
            })
        else:
            rows.append({"Model": display, "AUC": "—", "Accuracy": "—", "F1": "—"})

    df = pd.DataFrame(rows).set_index("Model")
    st.dataframe(df, use_container_width=True, height=210)


if __name__ == "__main__":
    main()
