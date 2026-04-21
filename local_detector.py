import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import threading
import time
import pygame
import os

# ==========================================
# CONFIGURATION
# ==========================================
IMG_SIZE = 96          # Must match the training script!
CLOSED_THRESHOLD = 20  # Number of frames of drowsiness to trigger alarm
DEBUG_MODE = True      # Set to True to see what the model is actually seeing!

# Initialize Pygame Mixer for Alarm
pygame.mixer.init()
ALARM_PATH = "alarm.mpeg"
if os.path.exists(ALARM_PATH):
    pygame.mixer.music.load(ALARM_PATH)
else:
    print(f"Warning: {ALARM_PATH} not found. Alarm will not play sound.")

# Load Model (native .h5 format - no patches needed!)
print("Loading Unified Drowsiness Model...")
try:
    model = tf.keras.models.load_model("drowsiness_model.h5", compile=False)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please ensure 'drowsiness_model.h5' is in this directory.")
    exit()

# Mediapipe Setup
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Landmark Indices
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]

# State Variables
drowsy_frames = 0
alarm_thread = None
alarm_running = False

# ==========================================
# ALARM FUNCTION
# ==========================================
def play_alarm():
    global alarm_running
    alarm_running = True
    if os.path.exists(ALARM_PATH):
        pygame.mixer.music.play(-1)  # Loop indefinitely
        while alarm_running:
            time.sleep(0.1)
        pygame.mixer.music.stop()
    alarm_running = False

def trigger_alarm():
    global alarm_thread, alarm_running
    if not alarm_running:
        alarm_thread = threading.Thread(target=play_alarm, daemon=True)
        alarm_thread.start()

def stop_alarm():
    global alarm_running
    alarm_running = False

# ==========================================
# SMART CROP & PREPROCESS
# ==========================================
def extract_and_preprocess(frame, landmarks_list, padding_ratio=0.3):
    h, w, _ = frame.shape
    
    x_coords = [int(p.x * w) for p in landmarks_list]
    y_coords = [int(p.y * h) for p in landmarks_list]
    
    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)
    
    width = max_x - min_x
    height = max_y - min_y
    
    side_length = max(width, height)
    pad = int(side_length * padding_ratio)
    
    center_x = min_x + width // 2
    center_y = min_y + height // 2
    
    new_min_x = max(0, center_x - side_length // 2 - pad)
    new_max_x = min(w, center_x + side_length // 2 + pad)
    new_min_y = max(0, center_y - side_length // 2 - pad)
    new_max_y = min(h, center_y + side_length // 2 + pad)
    
    crop = frame[new_min_y:new_max_y, new_min_x:new_max_x]
    
    if crop.size == 0:
        return None, None
        
    debug_crop = crop.copy()

    # Preprocessing (MobileNetV2 handles internal scaling to [-1, 1])
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    crop = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
    crop = np.expand_dims(crop, axis=0).astype(np.float32)
    
    return crop, debug_crop

# ==========================================
# MAIN LOOP
# ==========================================
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)
    
    status = "Awake"
    is_drowsy = False
    debug_img1, debug_img2, debug_img3 = None, None, None

    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0].landmark
        
        # Extract features
        left_eye_tensor, debug_img1 = extract_and_preprocess(frame, [face_landmarks[i] for i in LEFT_EYE])
        right_eye_tensor, debug_img2 = extract_and_preprocess(frame, [face_landmarks[i] for i in RIGHT_EYE])
        mouth_tensor, debug_img3 = extract_and_preprocess(frame, [face_landmarks[i] for i in MOUTH])
        
        preds = []
        if left_eye_tensor is not None:
            preds.append(model.predict(left_eye_tensor, verbose=0)[0][0])
        if right_eye_tensor is not None:
            preds.append(model.predict(right_eye_tensor, verbose=0)[0][0])
        if mouth_tensor is not None:
            preds.append(model.predict(mouth_tensor, verbose=0)[0][0])
            
        if preds:
            # In your dataset, 0 = closed/yawn (drowsy), 1 = open (awake)
            eye_pred = sum(preds[:2]) / len(preds[:2]) if len(preds[:2]) > 0 else 1.0
            mouth_pred = preds[2] if len(preds) == 3 else 1.0
            
            # DEBUG: Print raw predictions to console
            print(f"L-Eye: {preds[0]:.4f}  R-Eye: {preds[1]:.4f}  Mouth: {mouth_pred:.4f}  | Eyes avg: {eye_pred:.4f}")
            
            if eye_pred < 0.5 or mouth_pred < 0.5:
                is_drowsy = True
                status = "Drowsy (Eyes Closed / Yawning)"
            
        if is_drowsy:
            drowsy_frames += 1
        else:
            drowsy_frames = max(0, drowsy_frames - 2)
            
    else:
        drowsy_frames = 0
        
    # --- ALARM LOGIC ---
    if drowsy_frames > CLOSED_THRESHOLD:
        cv2.putText(frame, "DROWSY - WAKE UP!", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
        trigger_alarm()
    else:
        stop_alarm()
        
    # --- UI DISPLAY ---
    color = (0, 0, 255) if is_drowsy else (0, 255, 0)
    cv2.putText(frame, f"State: {status} ({drowsy_frames}/{CLOSED_THRESHOLD})", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.imshow("Drowsiness Detection", frame)
    
    # --- DEBUG MODE DISPLAY ---
    if DEBUG_MODE:
        debug_panel = np.zeros((150, 450, 3), dtype=np.uint8)
        
        def paste_img(src, dest, x_offset):
            if src is not None:
                src_resized = cv2.resize(src, (150, 150))
                dest[0:150, x_offset:x_offset+150] = src_resized
                
        paste_img(debug_img1, debug_panel, 0)
        paste_img(debug_img2, debug_panel, 150)
        paste_img(debug_img3, debug_panel, 300)
        
        cv2.putText(debug_panel, "L-Eye", (50, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(debug_panel, "R-Eye", (200, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(debug_panel, "Mouth", (350, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.imshow("Debug: Model Vision", debug_panel)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
