
import os, json, gc
import numpy as np
import cv2
import gradio as gr

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import mobilenet_v3, xception

MODELS_DIR = os.getenv("MODELS_DIR", "models")
LABELS_JSON = os.getenv("LABELS_JSON", "labels.json")
CALIB_JSON = os.getenv("CALIB_JSON", "mobilenetv3_calibration.json")
TITLE = "DeepFake Detector – Multi-Model (MobileNetV3 / Xception / SimpleCNN)"

# ----------------------------
# GPU memory safety
# ----------------------------
try:
    gpus = tf.config.list_physical_devices("GPU")
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)
except Exception:
    pass

# ----------------------------
# Discover models
# ----------------------------
def discover_models(models_dir):
    if not os.path.isdir(models_dir):
        return []
    items = []
    for fname in sorted(os.listdir(models_dir)):
        if fname.lower().endswith((".keras", ".h5", ".savedmodel")):
            items.append(os.path.join(models_dir, fname))
    return items

AVAILABLE_MODELS = discover_models(MODELS_DIR)
assert AVAILABLE_MODELS, f"No model files found in '{MODELS_DIR}'."

# ----------------------------
# Labels
# ----------------------------
def load_labels(p):
    if not os.path.exists(p):
        return {0: "Fake", 1: "Real"}, {"Fake": 0, "Real": 1}
    with open(p, "r") as f:
        raw = json.load(f)

    if all(isinstance(v, int) for v in raw.values()):
        label2idx = {str(k): int(v) for k, v in raw.items()}
        idx2label = {v: k for k, v in label2idx.items()}
    elif all(str(k).isdigit() for k in raw.keys()):
        idx2label = {int(k): str(v) for k, v in raw.items()}
        label2idx = {v: k for k, v in idx2label.items()}
    else:
        idx2label = {0: "Fake", 1: "Real"}
        label2idx = {"Fake": 0, "Real": 1}
    return idx2label, label2idx

IDX2LABEL, LABEL2IDX = load_labels(LABELS_JSON)

# ----------------------------
# Calibration
# ----------------------------
def load_calibration(calib_path):
    if os.path.exists(calib_path):
        with open(calib_path, "r") as f:
            return json.load(f)
    return None

CALIB = load_calibration(CALIB_JSON)

# ----------------------------
# Model cache + helpers
# ----------------------------
_model_cache = {}
_last_conv_cache = {}
_preproc_cache = {}
_imgsize_cache = {}

def load_model(path):
    if path in _model_cache:
        return _model_cache[path]
    m = keras.models.load_model(path, compile=False)
    _model_cache[path] = m
    return m

def infer_img_size(m):
    if m in _imgsize_cache:
        return _imgsize_cache[m]
    try:
        h = int(m.inputs[0].shape[1]) if m.inputs[0].shape[1] is not None else 224
        w = int(m.inputs[0].shape[2]) if m.inputs[0].shape[2] is not None else 224
    except Exception:
        h, w = 224, 224
    _imgsize_cache[m] = (w, h)
    return (w, h)

def pick_preproc(m, model_path):
    if m in _preproc_cache:
        return _preproc_cache[m]

    for l in m.layers:
        if isinstance(l, layers.Rescaling):
            _preproc_cache[m] = lambda z: z
            return _preproc_cache[m]

    name = (m.name or "").lower()
    p = model_path.lower()

    if "mobilenet" in name or "mobilenet" in p:
        _preproc_cache[m] = mobilenet_v3.preprocess_input
    elif "xception" in name or "xception" in p:
        _preproc_cache[m] = xception.preprocess_input
    else:
        _preproc_cache[m] = lambda z: z / 255.0

    return _preproc_cache[m]

def find_last_conv_layer(m):
    if m in _last_conv_cache:
        return _last_conv_cache[m]
    last_name = None
    for layer in reversed(m.layers):
        if isinstance(layer, layers.Conv2D):
            last_name = layer.name
            break
    _last_conv_cache[m] = last_name
    return last_name

# ----------------------------
# Face crop
# ----------------------------
FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def crop_largest_face(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )
    if len(faces) == 0:
        return bgr

    x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
    pad = int(0.15 * max(w, h))

    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(bgr.shape[1], x + w + pad)
    y1 = min(bgr.shape[0], y + h + pad)

    return bgr[y0:y1, x0:x1]

# ----------------------------
# Grad-CAM
# ----------------------------
@tf.function
def _grad_cam_inner(cam_model, img_tensor, class_index, positive_index):
    with tf.GradientTape() as tape:
        conv_out, preds = cam_model(img_tensor)
        if preds.shape[-1] == 1:
            p1 = preds[:, 0]
            target = p1 if class_index == positive_index else (1.0 - p1)
        else:
            target = preds[:, class_index]
    grads = tape.gradient(target, conv_out)
    weights = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)
    cam = tf.reduce_sum(weights * conv_out, axis=-1)
    return cam, preds

def grad_cam(m, bgr_img, preproc, target_size, target_class, positive_index):
    lname = find_last_conv_layer(m)
    if lname is None:
        h, w = bgr_img.shape[:2]
        return np.zeros((h, w), dtype=np.uint8), bgr_img.copy()

    rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    x = cv2.resize(rgb, target_size).astype("float32")
    x = preproc(x)
    x = np.expand_dims(x, 0)

    cam_model = keras.Model([m.inputs], [m.get_layer(lname).output, m.output])
    cam, preds = _grad_cam_inner(
        cam_model,
        tf.convert_to_tensor(x),
        tf.constant(int(target_class)),
        tf.constant(int(positive_index))
    )

    cam = cam.numpy()[0]
    cam = np.maximum(cam, 0)
    cam /= (cam.max() + 1e-8)

    heat = (cam * 255).astype(np.uint8)
    heat = cv2.resize(heat, (bgr_img.shape[1], bgr_img.shape[0]))
    heat_color = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(bgr_img, 1.0, heat_color, 0.5, 0)
    return heat, overlay

# ----------------------------
# Explain text
# ----------------------------
def artifact_reasons(heatmap_u8):
    attn = heatmap_u8.astype(np.float32) / 255.0
    hi = (attn > 0.6).mean()
    h, w = heatmap_u8.shape[:2]
    center = heatmap_u8[h//4:3*h//4, w//4:3*w//4]
    center_focus = (center > 180).mean()

    reasons = []
    if hi > 0.08:
        reasons.append(f"Strong localized artifacts detected ({hi*100:.1f}% high-saliency).")
    if center_focus > 0.10:
        reasons.append("Attention near facial center; possible blending or illumination inconsistencies.")
    if not reasons:
        reasons.append("No dominant artifact region; decision relies on subtle textures.")
    return reasons

# ----------------------------
# Prediction logic
# ----------------------------
def preprocess(pil_img, img_size, preproc, face_crop=False):
    rgb = np.array(pil_img.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    if face_crop:
        bgr = crop_largest_face(bgr)

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    x = cv2.resize(rgb, img_size).astype("float32")
    x = preproc(x)
    x = np.expand_dims(x, 0)
    return bgr, x

def get_binary_probs(raw, model_path):
    pred = np.array(raw).reshape(-1)

    if pred.size > 1:
        probs = pred.astype(np.float32)
        probs = probs / (probs.sum() + 1e-8)
        if probs.size != 2:
            probs = np.pad(probs, (0, max(0, 2 - probs.size)))[:2]
        pred_idx = int(np.argmax(probs))
        conf = float(probs[pred_idx])
        return probs, pred_idx, conf, None, None

    raw_sigmoid = float(pred[0])

    # Default assumptions
    positive_label = "Real"
    threshold = 0.5

    # Use calibration only for MobileNetV3
    if CALIB is not None and "mobilenet" in os.path.basename(model_path).lower():
        positive_label = CALIB.get("sigmoid_positive_label", "Real")
        threshold = float(CALIB.get("best_threshold_for_real", 0.5))

    if positive_label == "Real":
        p_real = raw_sigmoid
    else:
        p_real = 1.0 - raw_sigmoid

    p_fake = 1.0 - p_real

    pred_idx = 1 if p_real >= threshold else 0
    conf = p_real if pred_idx == 1 else p_fake
    probs = np.array([p_fake, p_real], dtype=np.float32)

    return probs, pred_idx, conf, raw_sigmoid, threshold

def run_inference(model_path, pil_img, face_crop):
    if pil_img is None or not model_path:
        return {"error": "Please select a model and upload an image."}, None, "Awaiting input."

    m = load_model(model_path)
    img_size = infer_img_size(m)
    preproc = pick_preproc(m, model_path)

    bgr, x = preprocess(pil_img, img_size, preproc, face_crop=face_crop)
    raw = m.predict(x, verbose=0)[0]

    probs, pred_idx, conf, raw_sigmoid, threshold = get_binary_probs(raw, model_path)
    pred_label = IDX2LABEL.get(pred_idx, f"class{pred_idx}")

    positive_index = 1
    if CALIB is not None and "mobilenet" in os.path.basename(model_path).lower():
        positive_index = 1 if CALIB.get("sigmoid_positive_label", "Real") == "Real" else 0

    try:
        heat, overlay = grad_cam(m, bgr, preproc, img_size, pred_idx, positive_index)
    except Exception:
        heat = np.zeros((bgr.shape[0], bgr.shape[1]), dtype=np.uint8)
        overlay = bgr.copy()

    reasons = artifact_reasons(heat)
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    out = {
        "Model": os.path.basename(model_path),
        "Prediction": pred_label,
        "Confidence": round(float(conf), 4),
        f"P({IDX2LABEL.get(0, 'class0')})": round(float(probs[0]), 4),
        f"P({IDX2LABEL.get(1, 'class1')})": round(float(probs[1]), 4),
        "InputSize": f"{img_size[1]}x{img_size[0]}",
        "FaceCrop": bool(face_crop),
    }

    if raw_sigmoid is not None:
        out["RawSigmoid"] = round(float(raw_sigmoid), 4)
        out["ThresholdUsed"] = round(float(threshold), 4)

    txt = "\n".join([f"• {r}" for r in reasons])

    del x, raw
    gc.collect()
    return out, overlay_rgb, txt

# ----------------------------
# Gradio UI
# ----------------------------
with gr.Blocks(title=TITLE) as demo:
    gr.Markdown(
        f"# {TITLE}\n"
        f"Select a model, upload an image, and analyze it.\n\n"
        f"**Models dir:** `{MODELS_DIR}`  |  **Labels:** `{LABELS_JSON}`  |  **Calibration:** `{CALIB_JSON}`"
    )

    with gr.Row():
        with gr.Column(scale=1):
            model_dd = gr.Dropdown(
                choices=AVAILABLE_MODELS,
                value=AVAILABLE_MODELS[0],
                label="Select model"
            )
            inp = gr.Image(type="pil", label="Upload / Webcam", sources=["upload", "webcam"])
            face_crop_cb = gr.Checkbox(value=True, label="Crop face before inference")
            analyze = gr.Button("Analyze", variant="primary")

        with gr.Column(scale=1):
            out_json = gr.JSON(label="Prediction")
            out_img = gr.Image(type="numpy", label="Grad-CAM Overlay")
            out_txt = gr.Markdown(label="Why did the model decide this?")

    analyze.click(run_inference, inputs=[model_dd, inp, face_crop_cb], outputs=[out_json, out_img, out_txt])
    inp.change(run_inference, inputs=[model_dd, inp, face_crop_cb], outputs=[out_json, out_img, out_txt])
    model_dd.change(run_inference, inputs=[model_dd, inp, face_crop_cb], outputs=[out_json, out_img, out_txt])
    face_crop_cb.change(run_inference, inputs=[model_dd, inp, face_crop_cb], outputs=[out_json, out_img, out_txt])

if __name__ == "__main__":
    demo.launch(ssr_mode=False)