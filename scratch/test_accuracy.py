import tensorflow as tf
import numpy as np
import os
import cv2

IMG_SIZE = 96
DATASET_DIR = r"d:\PROJECT\Drowsiness_detection\final_dataset\val"
MODEL_PATH = r"d:\PROJECT\Drowsiness_detection\drowsiness_model.keras"

print(f"Loading model: {MODEL_PATH}")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

def test_folder(folder_name):
    path = os.path.join(DATASET_DIR, folder_name)
    if not os.path.exists(path):
        print(f"Folder not found: {path}")
        return
    
    files = os.listdir(path)[:10]
    print(f"\nTesting {folder_name} (Expected around {'0' if 'closed' in folder_name else '1'}):")
    
    scores = []
    for f in files:
        img_path = os.path.join(path, f)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = np.expand_dims(img, axis=0).astype(np.float32)
        
        score = model.predict(img, verbose=0)[0][0]
        scores.append(score)
        print(f"  {f}: {score:.4f}")
    
    print(f"Average for {folder_name}: {np.mean(scores):.4f}")

test_folder("closed")
test_folder("open")
