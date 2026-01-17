import json
import time
import shutil
from pathlib import Path
from datetime import datetime

import streamlit as st
import streamlit.components.v1 as components
from streamlit_autorefresh import st_autorefresh

import yaml
import pandas as pd
from PIL import Image

from ultralytics import YOLO


# ----------------------------
# Helpers
# ----------------------------
def ensure_dirs(*paths: Path):
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def is_image_file(p: Path) -> bool:
    return p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]


def newest_file(folder: Path):
    files = [p for p in folder.glob("*") if p.is_file()]
    if not files:
        return None
    return max(files, key=lambda x: x.stat().st_mtime)


def list_images_sorted(folder: Path):
    imgs = [p for p in folder.glob("*") if p.is_file() and is_image_file(p)]
    imgs.sort(key=lambda x: x.stat().st_mtime)
    return imgs


def safe_read_image(path: Path) -> Image.Image:
    # handle "file still being copied"
    for _ in range(6):
        try:
            return Image.open(path).convert("RGB")
        except Exception:
            time.sleep(0.2)
    return Image.open(path).convert("RGB")


def load_class_names_from_yaml(yaml_path: Path):
    if not yaml_path.exists():
        return None
    with open(yaml_path, "r", encoding="utf-8") as f:
        y = yaml.safe_load(f)
    names = y.get("names")
    if isinstance(names, list) and len(names) > 0:
        return names
    if isinstance(names, dict) and len(names) > 0:
        return [names[k] for k in sorted(names.keys())]
    return None


def append_result_csv(csv_path: Path, row: dict):
    df_row = pd.DataFrame([row])
    if csv_path.exists():
        df_row.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df_row.to_csv(csv_path, mode="w", header=True, index=False)


def maybe_rotate_log(csv_path: Path, archive_dir: Path, max_mb: int = 10, max_rows: int = 200_000):
    """Auto-archive results.csv if it gets too large."""
    if not csv_path.exists():
        return
    try:
        size_mb = csv_path.stat().st_size / (1024 * 1024)
        rotate = size_mb >= max_mb

        if not rotate:
            # lightweight row count check (fine for DEMO)
            try:
                rows = sum(1 for _ in open(csv_path, "rb")) - 1
                rotate = rows >= max_rows
            except Exception:
                pass

        if rotate:
            ensure_dirs(archive_dir)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            shutil.move(str(csv_path), str(archive_dir / f"results_{ts}.csv"))
    except Exception:
        pass


def play_fail_beep():
    components.html(
        """
        <script>
        (function() {
          try {
            const ctx = new (window.AudioContext || window.webkitAudioContext)();
            const o = ctx.createOscillator();
            const g = ctx.createGain();
            o.type = "sine";
            o.frequency.value = 880;
            o.connect(g);
            g.connect(ctx.destination);
            g.gain.setValueAtTime(0.0001, ctx.currentTime);
            g.gain.exponentialRampToValueAtTime(0.2, ctx.currentTime + 0.02);
            o.start();
            g.gain.exponentialRampToValueAtTime(0.0001, ctx.currentTime + 0.35);
            o.stop(ctx.currentTime + 0.36);
          } catch(e) {}
        })();
        </script>
        """,
        height=0,
    )


def popup_alert(message: str):
    st.toast(f"🚨 {message}", icon="🚨")
    # ❌ เอา alert() ออก เพราะมัน block autorefresh



def popup_info(message: str):
    st.toast(message, icon="ℹ️")


def run_inference_and_save(model: YOLO, img_path: Path, processed_dir: Path, conf: float, iou: float, imgsz: int):
    results = model.predict(source=str(img_path), conf=conf, iou=iou, imgsz=imgsz, verbose=False)
    r = results[0]

    annotated = r.plot()              # BGR numpy
    annotated_rgb = annotated[:, :, ::-1]
    out_img = Image.fromarray(annotated_rgb)

    out_path = processed_dir / img_path.name
    out_img.save(out_path)

    dets = []
    if r.boxes is not None and len(r.boxes) > 0:
        cls = r.boxes.cls.cpu().numpy().astype(int)
        confs = r.boxes.conf.cpu().numpy()
        xyxy = r.boxes.xyxy.cpu().numpy()
        for i in range(len(cls)):
            dets.append(
                {
                    "cls_id": int(cls[i]),
                    "conf": float(confs[i]),
                    "x1": float(xyxy[i][0]),
                    "y1": float(xyxy[i][1]),
                    "x2": float(xyxy[i][2]),
                    "y2": float(xyxy[i][3]),
                }
            )
    return out_path, dets


def decide_status(dets, class_names, expected_color: str, multiple_policy: str):
    if not dets:
        return ("FAIL", "Missing scoop", None, [], None)

    labels = []
    for d in dets:
        cid = d["cls_id"]
        if class_names and 0 <= cid < len(class_names):
            labels.append(class_names[cid])
        else:
            labels.append(str(cid))

    n = len(dets)
    detected_all = labels[:]

    if n > 1 and multiple_policy == "FAIL_IF_MULTIPLE":
        return ("FAIL", f"Multiple spoons detected (n={n})", None, detected_all, None)

    best_idx = max(range(n), key=lambda i: dets[i]["conf"])
    best_label = labels[best_idx]
    best_conf = dets[best_idx]["conf"]

    if best_label != expected_color:
        return (
            "FAIL",
            f"Wrong color (detected: {best_label}, expected: {expected_color})",
            best_label,
            detected_all,
            best_conf,
        )

    return ("PASS", f"Correct color: {best_label}", best_label, detected_all, best_conf)



def load_profiles(path: Path):
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_profiles(path: Path, profiles: dict):
    path.write_text(json.dumps(profiles, indent=2), encoding="utf-8")


def reset_soft():
    st.session_state["current_status"] = None
    st.session_state["current_reason"] = None
    st.session_state["current_file"] = None
    st.session_state["current_ts"] = None
    st.session_state["last_alert_key"] = None


def hard_reset(processed_dir: Path, results_csv: Path):
    # archive results instead of deleting
    if results_csv.exists():
        arch_dir = Path("archive")
        ensure_dirs(arch_dir)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        shutil.move(str(results_csv), str(arch_dir / f"results_{ts}.csv"))

    # clear processed
    if processed_dir.exists():
        for p in processed_dir.glob("*"):
            if p.is_file():
                p.unlink(missing_ok=True)

    reset_soft()


# ----------------------------
# App
# ----------------------------
st.set_page_config(page_title="Scoop Inspection Report", layout="wide")

# refresh every 3 seconds (no sleep)
st_autorefresh(interval=3000, key="refresh_3s")

st.title("🥄 Scoop Inspection Report")

# session state
st.session_state.setdefault("run_mode", True)
st.session_state.setdefault("confirm_hard", False)
st.session_state.setdefault("last_alert_key", None)
st.session_state.setdefault("current_status", None)
st.session_state.setdefault("current_reason", None)
st.session_state.setdefault("current_file", None)
st.session_state.setdefault("current_ts", None)

# Paths
MODEL_PATH_DEFAULT = "models/best.pt"
YAML_PATH_DEFAULT = "data.yaml"
PROFILES_PATH = Path("profiles.json")

INCOMING_DIR = Path("incoming")
PROCESSED_DIR = Path("processed")
RESULTS_CSV = Path("results/results.csv")
ARCHIVE_DIR = Path("archive")

ensure_dirs(INCOMING_DIR, PROCESSED_DIR, RESULTS_CSV.parent, ARCHIVE_DIR)

# profiles
profiles = load_profiles(PROFILES_PATH)

# ----------------------------
# Sidebar: controls
# ----------------------------
st.sidebar.header("🧰 Control Panel")

# Start/Stop
c1, c2 = st.sidebar.columns(2)
with c1:
    if st.button("▶️ Start", use_container_width=True):
        st.session_state["run_mode"] = True
with c2:
    if st.button("⏸ Stop", use_container_width=True):
        st.session_state["run_mode"] = False

st.sidebar.write(f"Status: {'RUNNING' if st.session_state['run_mode'] else 'STOPPED'}")
st.sidebar.divider()

# Product selector
product_list = ["Product A", "Product B", "Product C"]
product = st.sidebar.selectbox("Product", product_list, index=0)

# ensure profile exists
DEFAULT_EXPECTED = {"Product A": "Blue", "Product B": "Green", "Product C": "White"}
if product not in profiles:
    profiles[product] = {
        "expected_color": DEFAULT_EXPECTED[product],
        "conf": 0.50 if product != "Product C" else 0.55,
        "iou": 0.45,
        "imgsz": 640,
        "multiple_policy": "FAIL_IF_MULTIPLE",
        "alert_on_no_object": False,
    }

# Paths config
model_path = Path(st.sidebar.text_input("Model path", MODEL_PATH_DEFAULT))
yaml_path = Path(st.sidebar.text_input("YAML path (optional)", YAML_PATH_DEFAULT))

# class names from yaml (fallback later)
class_names = load_class_names_from_yaml(yaml_path)

expected_color = profiles[product]["expected_color"]
st.sidebar.caption(f"Expected color: **{expected_color}**")

# Per-product thresholds
st.sidebar.subheader("Thresholds (per product)")
conf = st.sidebar.slider("Confidence (conf)", 0.05, 0.95, float(profiles[product]["conf"]), 0.05)
iou = st.sidebar.slider("IoU", 0.10, 0.90, float(profiles[product]["iou"]), 0.05)
imgsz = st.sidebar.selectbox("Image size (imgsz)", [320, 480, 640, 800, 960],
                             index=[320, 480, 640, 800, 960].index(int(profiles[product]["imgsz"])))

multiple_policy_label = st.sidebar.selectbox(
    "Multiple detections handling",
    ["Fail if >1 spoon", "Use best confidence only"],
    index=0 if profiles[product]["multiple_policy"] == "FAIL_IF_MULTIPLE" else 1,
)
multiple_policy = "FAIL_IF_MULTIPLE" if multiple_policy_label.startswith("Fail") else "BEST_ONLY"

alert_on_no_object = st.sidebar.checkbox("Alert on NO OBJECT", value=bool(profiles[product].get("alert_on_no_object", False)))

# Save/Reset profile
st.sidebar.divider()
p1, p2 = st.sidebar.columns(2)
with p1:
    if st.button("💾 Save profile", use_container_width=True):
        profiles[product]["conf"] = float(conf)
        profiles[product]["iou"] = float(iou)
        profiles[product]["imgsz"] = int(imgsz)
        profiles[product]["multiple_policy"] = multiple_policy
        profiles[product]["alert_on_no_object"] = bool(alert_on_no_object)
        save_profiles(PROFILES_PATH, profiles)
        st.sidebar.success("Saved")
with p2:
    if st.button("↩️ Reset profile", use_container_width=True):
        profiles[product] = {
            "expected_color": DEFAULT_EXPECTED[product],
            "conf": 0.50 if product != "Product C" else 0.55,
            "iou": 0.45,
            "imgsz": 640,
            "multiple_policy": "FAIL_IF_MULTIPLE",
            "alert_on_no_object": False,
        }
        save_profiles(PROFILES_PATH, profiles)
        st.sidebar.warning("Reset to default")

# Reset buttons
st.sidebar.divider()
st.sidebar.subheader("Reset")
r1, r2 = st.sidebar.columns(2)
with r1:
    if st.button("🧽 Soft Reset", use_container_width=True):
        reset_soft()
        st.sidebar.success("Soft reset done")
with r2:
    if st.button("🧨 Hard Reset", use_container_width=True):
        st.session_state["confirm_hard"] = True

if st.session_state.get("confirm_hard", False):
    st.sidebar.warning("Hard Reset will clear processed images and archive current results log.")
    cf1, cf2 = st.sidebar.columns(2)
    with cf1:
        if st.button("✅ Confirm Hard Reset", use_container_width=True):
            hard_reset(PROCESSED_DIR, RESULTS_CSV)
            st.session_state["confirm_hard"] = False
            st.sidebar.success("Hard reset done")
    with cf2:
        if st.button("❌ Cancel", use_container_width=True):
            st.session_state["confirm_hard"] = False

st.sidebar.divider()
st.sidebar.caption("Copy images into **incoming/** to simulate camera snapshots.")


# ----------------------------
# Model loading (cached)
# ----------------------------
@st.cache_resource
def load_model(model_path_str: str):
    return YOLO(model_path_str)

if not model_path.exists():
    st.error(f"❌ Model not found: {model_path}")
    st.stop()

model = load_model(str(model_path))

# class_names fallback from model
if not class_names:
    names = getattr(model, "names", None)
    if isinstance(names, dict):
        class_names = [names[k] for k in sorted(names.keys())]
    elif isinstance(names, list):
        class_names = names
    else:
        class_names = ["Blue", "Green", "White"]


# ----------------------------
# Folder Watch Processing (Queue only)
# ----------------------------
incoming_images = list_images_sorted(INCOMING_DIR)

def pick_next_image_queue():
    # เลือกไฟล์ที่ยังไม่ถูก process (oldest first)
    for p in incoming_images:
        if not (PROCESSED_DIR / p.name).exists():
            return p
    return None

img_to_process = pick_next_image_queue()

if st.session_state["run_mode"] and img_to_process is not None:
    try:
        _, dets = run_inference_and_save(
            model=model,
            img_path=img_to_process,
            processed_dir=PROCESSED_DIR,
            conf=float(conf),
            iou=float(iou),
            imgsz=int(imgsz),
        )

        status, reason, detected_best, detected_all, best_conf = decide_status(
            dets=dets,
            class_names=class_names,
            expected_color=expected_color,
            multiple_policy=multiple_policy,
        )

        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        maybe_rotate_log(RESULTS_CSV, ARCHIVE_DIR)

        append_result_csv(
            RESULTS_CSV,
            {
                "timestamp": now_str,
                "file": img_to_process.name,
                "product": product,
                "expected_color": expected_color,
                "detected_best": detected_best if detected_best else "",
                "detected_all": ",".join(detected_all) if detected_all else "",
                "status": status,
                "reason": reason,
                "conf_setting": float(conf),
                "iou_setting": float(iou),
                "imgsz": int(imgsz),
                "multiple_policy": multiple_policy,
                "num_detections": len(dets),
                "best_conf": float(best_conf) if best_conf is not None else "",
            },
        )

        # alert (ไม่ซ้ำ)
        alert_key = f"{now_str}|{img_to_process.name}|{status}|{reason}"
        if alert_key != st.session_state.get("last_alert_key"):
            if status == "FAIL":
                play_fail_beep()
                popup_alert(reason)
                st.session_state["last_alert_key"] = alert_key

    except Exception as e:
        st.error(f"❌ Inference error: {e}")

total = len(incoming_images)
done = sum(1 for p in incoming_images if (PROCESSED_DIR / p.name).exists())

# ----------------------------
# UI
# ----------------------------
col_left, col_right = st.columns([2, 1], vertical_alignment="top")

with col_left:
    st.subheader("📸 Latest Snapshot (Processed)")
    latest_processed = newest_file(PROCESSED_DIR)
    if latest_processed and is_image_file(latest_processed):
        st.image(safe_read_image(latest_processed), use_container_width=True)
    else:
        st.info("No processed image yet. Drop/copy an image into the incoming folder.")

with col_right:
    st.subheader("🚦 Current Status")
    current = None
    if RESULTS_CSV.exists():
        try:
            df0 = pd.read_csv(RESULTS_CSV)
            if len(df0) > 0:
                current = df0.iloc[-1].to_dict()
        except Exception:
            current = None

    if current:
        status = current.get("status", "")
        reason = current.get("reason", "")
        prod = current.get("product", "")
        exp = current.get("expected_color", "")
        det_best = current.get("detected_best", "")
        fn = current.get("file", "")
        ts = current.get("timestamp", "")
        n = current.get("num_detections", "")

        if status == "PASS":
            st.success("PASS")
        elif status == "FAIL":
            st.error("FAIL")
        else:
            st.info(status or "UNKNOWN")

        st.write(reason)
        st.caption(f"Product: {prod}")
        st.caption(f"Expected: {exp} | Detected(best): {det_best or '-'} | n={n}")
        st.caption(f"Time: {ts}")
    else:
        st.info("No status yet.")

st.divider()

# Performance last 1 hour
st.subheader("📈 Last 1 hour performance")
if RESULTS_CSV.exists():
    df = pd.read_csv(RESULTS_CSV)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")

    now = pd.Timestamp.now()
    last1h = df[df["timestamp"] >= (now - pd.Timedelta(hours=1))].copy()

    if len(last1h) == 0:
        st.info("No data in the last 1 hour yet.")
    else:
        total = len(last1h)
        fail = int((last1h["status"] == "FAIL").sum())
        pass_ = int((last1h["status"] == "PASS").sum())
        fail_rate = (fail / total) * 100 if total else 0.0

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total (1h)", total)
        m2.metric("PASS (1h)", pass_)
        m3.metric("FAIL (1h)", fail)
        m4.metric("Fail rate (1h)", f"{fail_rate:.1f}%")

        last1h = last1h.set_index("timestamp")
        trend = (last1h["status"] == "FAIL").resample("1min").mean().fillna(0) * 100
        trend.name = "Fail rate % (per min)"
        st.line_chart(trend)
else:
    st.info("results.csv not created yet.")

st.divider()

# Results Log + search/filter
st.subheader("🧾 Results Log")
if RESULTS_CSV.exists():
    df = pd.read_csv(RESULTS_CSV)

    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        only_fail = st.checkbox("Show FAIL only", value=False)
    with c2:
        only_product = st.selectbox("Filter product", ["All", "Product A", "Product B", "Product C"], index=0)
    with c3:
        q = st.text_input("Search (reason)", value="")

    view = df.copy()
    if only_fail:
        view = view[view["status"] == "FAIL"]
    if only_product != "All":
        view = view[view["product"] == only_product]
    if q.strip():
        qn = q.strip().lower()
        # Search only in reason (ไม่โชว์/ไม่ค้น filename)
        view = view[view["reason"].astype(str).str.lower().str.contains(qn)]

    # ซ่อนคอลัมน์ file เพื่อไม่ให้เห็นชื่อภาพ
    view_display = view.drop(columns=["file"], errors="ignore")
    st.dataframe(view_display.tail(50), use_container_width=True)

    # Download CSV (ไม่ต้องมี expander เพื่อกัน indent พัง)
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download results.csv",
        data=csv_bytes,
        file_name="results.csv",
        mime="text/csv",
    )
else:
    st.info("No results yet. Start running and drop images into incoming/.")
