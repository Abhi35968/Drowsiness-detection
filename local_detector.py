import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import threading
import time
import pygame
import os
from tensorflow.keras.layers import DepthwiseConv2D

# Fix for Keras version mismatch ("groups" argument error)
class PatchedDepthwiseConv2D(DepthwiseConv2D):
    def __init__(self, **kwargs):
        if 'groups' in kwargs:
            kwargs.pop('groups')
        super().__init__(**kwargs)

# ==========================================
# CONFIGURATION
# ==========================================
IMG_SIZE = 96          
CLOSED_THRESHOLD = 25  # Increased to prevent flicker
DEBUG_MODE = True      
MODEL_PATH = "drowsiness_model.keras" 

# CALIBRATION (Based on your latest screenshots)
# Your model is very confident (0.99+) for "Open" and slightly less (~0.98) for "Closed".
# We will use a very strict threshold to catch the difference.
EYE_INVERTED = False   
MOUTH_INVERTED = False 

EYE_THRESHOLD = 0.99   # If score drops below this, it's starting to close
MOUTH_THRESHOLD = 0.95 # If score drops below this, it's a yawn (Open mouth folder 0?)
# Wait, if closed mouth is 0.87, then Open mouth (Yawn) must be even LOWER.
# Let's use a dynamic approach.

# Initialize Pygame Mixer for Alarm
try:
    pygame.mixer.init()
    ALARM_PATH = "alarm.mpeg"
    if os.path.exists(ALARM_PATH):
        pygame.mixer.music.load(ALARM_PATH)
    else:
        print(f"Warning: {ALARM_PATH} not found.")
except Exception as e:
    print(f"Warning: Audio error: {e}")

# Load Model
print(f"Loading Model...")
try:
    # Fallback logic for Keras version mismatch
    model = tf.keras.models.load_model(
        "drowsiness_model.h5", 
        custom_objects={'DepthwiseConv2D': PatchedDepthwiseConv2D},
        compile=False
    )
    print("Loaded .h5 model successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
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
alarm_running = False

# ==========================================
# ALARM FUNCTIONS
# ==========================================
def play_alarm():
    global alarm_running
    alarm_running = True
    if os.path.exists(ALARM_PATH):
        pygame.mixer.music.play(-1)
        while alarm_running:
            time.sleep(0.1)
        pygame.mixer.music.stop()
    alarm_running = False

def trigger_alarm():
    global alarm_running
    if not alarm_running:
        threading.Thread(target=play_alarm, daemon=True).start()

def stop_alarm():
    global alarm_running
    alarm_running = False

# ==========================================
# SMART CROP & PREPROCESS (ENHANCED)
# ==========================================
def extract_and_preprocess(frame, landmarks_list, padding_ratio=0.15):
    h, w, _ = frame.shape
    x_coords = [int(p.x * w) for p in landmarks_list]
    y_coords = [int(p.y * h) for p in landmarks_list]
    
    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)
    
    width, height = max_x - min_x, max_y - min_y
    side_length = max(width, height)
    pad = int(side_length * padding_ratio)
    
    center_x, center_y = min_x + width // 2, min_y + height // 2
    
    new_min_x = max(0, center_x - side_length // 2 - pad)
    new_max_x = min(w, center_x + side_length // 2 + pad)
    new_min_y = max(0, center_y - side_length // 2 - pad)
    new_max_y = min(h, center_y + side_length // 2 + pad)
    
    crop = frame[new_min_y:new_max_y, new_min_x:new_max_x]
    if crop.size == 0: return None, None
        
    # --- ENHANCEMENT FOR DARK ROOMS ---
    # 1. Convert to grayscale
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    # 2. Equalize Histogram (Boosts contrast/brightness)
    equalized = cv2.equalizeHist(gray)
    # 3. Convert back to RGB (3 identical channels) to match MobileNetV2 input
    enhanced = cv2.cvtColor(equalized, cv2.COLOR_GRAY2RGB)
    
    debug_crop = enhanced.copy()
    
    crop_resized = cv2.resize(enhanced, (IMG_SIZE, IMG_SIZE))
    crop_final = np.expand_dims(crop_resized, axis=0).astype(np.float32)
    return crop_final, debug_crop

# ==========================================
# MAIN LOOP
# ==========================================
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    status, is_drowsy = "Awake", False
    debug_img1, debug_img2, debug_img3 = None, None, None

    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0].landmark
        
        # 1. Extract features
        l_tensor, debug_img1 = extract_and_preprocess(frame, [face_landmarks[i] for i in LEFT_EYE])
        r_tensor, debug_img2 = extract_and_preprocess(frame, [face_landmarks[i] for i in RIGHT_EYE])
        m_tensor, debug_img3 = extract_and_preprocess(frame, [face_landmarks[i] for i in MOUTH])
        
        # 2. Predict
        l_score = model.predict(l_tensor, verbose=0)[0][0] if l_tensor is not None else 1.0
        r_score = model.predict(r_tensor, verbose=0)[0][0] if r_tensor is not None else 1.0
        m_score = model.predict(m_tensor, verbose=0)[0][0] if m_tensor is not None else 1.0
            
        # 3. Apply Inversion Fixes
        if EYE_INVERTED:
            l_score = 1.0 - l_score
            r_score = 1.0 - r_score
        if MOUTH_INVERTED:
            m_score = 1.0 - m_score
            
        eye_avg = (l_score + r_score) / 2
        
        # LOGIC: Your model outputs ~0.99 for Open and ~0.98 for Closed.
        # This is a very small margin, so we use a strict threshold.
        # If EYE_INVERTED is False: 1.0=Open, 0.0=Closed.
        # If EYE_INVERTED is True: 0.0=Open, 1.0=Closed.
        
        # Adjusting logic for your specific model behavior:
        if not EYE_INVERTED:
            if eye_avg < 0.992: # Strict threshold for "Not fully open"
                is_drowsy = True
                status = "Drowsy (Eyes Closed)"
        else:
            if eye_avg > 0.01: # Inverse logic
                is_drowsy = True
                status = "Drowsy (Eyes Closed)"

        # Mouth Logic: If m_score drops significantly, it's likely a yawn or speech
        if m_score < 0.85:
            is_drowsy = True
            status = "Drowsy (Yawning)"
            
        print(f"Eye: {eye_avg:.4f} | Mouth: {m_score:.4f} | State: {status}")
        
        if is_drowsy:
            drowsy_frames += 1
        else:
            drowsy_frames = max(0, drowsy_frames - 1)
    else:
        drowsy_frames = 0
        
    # Alarm Logic
    if drowsy_frames > CLOSED_THRESHOLD:
        cv2.putText(frame, "DROWSY - WAKE UP!", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
        trigger_alarm()
    else:
        stop_alarm()
        
    # UI
    color = (0, 0, 255) if is_drowsy else (0, 255, 0)
    cv2.putText(frame, f"State: {status} ({drowsy_frames}/{CLOSED_THRESHOLD})", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(frame, f"Eye: {eye_avg:.4f} Mouth: {m_score:.4f}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.imshow("Drowsiness Detection", frame)
    
    if DEBUG_MODE:
        debug_panel = np.zeros((150, 450, 3), dtype=np.uint8)
        for i, img in enumerate([debug_img1, debug_img2, debug_img3]):
            if img is not None:
                debug_panel[:, i*150:(i+1)*150] = cv2.resize(img, (150, 150))
        cv2.imshow("Debug: Model Vision", debug_panel)
    
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
