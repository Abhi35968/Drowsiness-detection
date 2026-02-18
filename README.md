# Drowsiness Detection

## Overview
- Real-time driver fatigue detector that streams frames from the default webcam, isolates the eye region with MediaPipe Face Mesh, and classifies it as open/closed using a TensorFlow CNN.
- When the left eye stays closed for more than 20 consecutive frames, the app overlays a red alert and plays `alarm.mpeg` in a background thread via `playsound`.

## Requirements
- Python 3.9+ and an accessible webcam
- Dependencies: `opencv-python`, `numpy`, `tensorflow`, `mediapipe`, `playsound`
- Trained models: `best_eye_model.keras` (preferred) or `best_eye_model.h5` fallback stored at the repo root

Install packages with:

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not available, install manually:

```bash
pip install opencv-python numpy tensorflow mediapipe playsound
```

## Usage
1. Place the eye model files (`best_eye_model.keras` or `best_eye_model.h5`) and `alarm.mpeg` in the project root alongside `detect_drowsy.py`.
2. Connect a webcam and ensure no other app is using it.
3. Run the detector:

	```bash
	python detect_drowsy.py
	```

4. Observe the live preview window:
	- HUD shows current eye state and consecutive closed-frame count.
	- Close the window or press `q` to stop.

## Troubleshooting
- **Webcam not working:** Confirm the correct camera index in `cv2.VideoCapture(0)` or allow camera permissions.
- **TensorFlow layer errors:** Keep both `.keras` and `.h5` versions of the model in root so the deserialization shim can fall back.
- **Audio not playing:** Verify default system output device and that `alarm.mpeg` path is correct.

## Training
- The `model_training.ipynb` notebook contains the CNN training workflow if you need to retrain or fine-tune the eye classifier.