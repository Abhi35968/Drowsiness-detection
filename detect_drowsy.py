import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import threading
from playsound import playsound


def _normalize_legacy_layer_config(config):
    cfg = config.copy()
    batch_shape = cfg.pop("batch_shape", None)
    if batch_shape is not None and "batch_input_shape" not in cfg:
        cfg["batch_input_shape"] = tuple(batch_shape)
    cfg.pop("optional", None)
    cfg.pop("quantization_config", None)

    dtype_cfg = cfg.get("dtype")
    if isinstance(dtype_cfg, dict) and dtype_cfg.get("class_name") == "DTypePolicy":
        dtype_name = dtype_cfg.get("config", {}).get("name")
        if dtype_name:
            cfg["dtype"] = tf.dtypes.as_dtype(dtype_name)

    return cfg


def _patch_layer_deserialization():
    layer_cls = tf.keras.layers.Layer
    input_layer_cls = tf.keras.layers.InputLayer

    original_layer_from_config = layer_cls.from_config
    original_input_from_config = input_layer_cls.from_config

    def patched_layer_from_config(cls, config):
        return original_layer_from_config.__func__(cls, _normalize_legacy_layer_config(config))

    def patched_input_from_config(cls, config):
        return original_input_from_config.__func__(cls, _normalize_legacy_layer_config(config))

    layer_cls.from_config = classmethod(patched_layer_from_config)
    input_layer_cls.from_config = classmethod(patched_input_from_config)

    def restore():
        layer_cls.from_config = original_layer_from_config
        input_layer_cls.from_config = original_input_from_config

    return restore


def _load_eye_model():
    _restore_layer_from_config = _patch_layer_deserialization()
    try:
        try:
            return tf.keras.models.load_model("best_eye_model.keras", compile=False, safe_mode=False)
        except (IOError, OSError, ValueError) as primary_error:
            print("Primary model load failed, trying H5 backup...", primary_error)
            return tf.keras.models.load_model("best_eye_model.h5", compile=False)
    finally:
        _restore_layer_from_config()


# ✅ Load trained model with backward-compatible layer handling
model = _load_eye_model()

IMG_SIZE = 64
CLOSED_LIMIT = 20   # closed eye frames for drowsy detection
closed_frames = 0
alarm_on = False

# ✅ MediaPipe Face Mesh
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(refine_landmarks=True)

# Eye landmark points (MediaPipe)
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

def alarm_sound():
    playsound("alarm.mpeg")

def crop_eye(image, landmarks, eye_points):
    h, w, _ = image.shape
    coords = [(int(landmarks[p].x * w), int(landmarks[p].y * h)) for p in eye_points]

    x_coords = [c[0] for c in coords]
    y_coords = [c[1] for c in coords]

    x_min, x_max = max(min(x_coords)-10, 0), min(max(x_coords)+10, w)
    y_min, y_max = max(min(y_coords)-10, 0), min(max(y_coords)+10, h)

    eye = image[y_min:y_max, x_min:x_max]
    return eye

# ✅ Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Webcam not working")
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    eye_state = "No Face"

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark

        left_eye = crop_eye(frame, landmarks, LEFT_EYE)

        if left_eye.size != 0:
            eye = cv2.resize(left_eye, (IMG_SIZE, IMG_SIZE))
            eye = eye / 255.0
            eye = np.expand_dims(eye, axis=0)

            pred = model.predict(eye, verbose=0)[0][0]

            # ✅ pred > 0.5 => Closed eye
            eye_state = "Open" if pred > 0.5 else "Closed"

            if eye_state == "Closed":
                closed_frames += 1
            else:
                closed_frames = 0
                alarm_on = False

    # ✅ Display info
    cv2.putText(frame, f"Eye: {eye_state}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.putText(frame, f"Closed Frames: {closed_frames}", (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    # ✅ Drowsiness Alert
    if closed_frames > CLOSED_LIMIT:
        cv2.putText(frame, "DROWSINESS ALERT!!!", (20, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        if not alarm_on:
            alarm_on = True
            threading.Thread(target=alarm_sound, daemon=True).start()

    cv2.imshow("Drowsiness Detection (Press Q to Quit)", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
