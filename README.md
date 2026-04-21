# Robust Real-Time Drowsiness & Yawn Detection

This project solves the "Domain Shift" problem (where a model trains well but fails on the webcam) by using **Smart Padding** and exact preprocessing alignment.

## 📁 Project Structure
*   `local_detector.py`: Main script for real-time webcam detection.
*   `drowsiness_model.h5`: The trained MobileNetV2 model (Keras 2 format).
*   `notebooks/`: Contains `train_on_cloud.ipynb` for retraining on Colab/Kaggle.
*   `utils/`: 
    *   `convert_model.py`: Converts Keras 3 `.keras` models to Keras 2 `.h5`.
    *   `explore_config.py`: Debugging tool to inspect model layer architecture.
*   `requirements.txt`: Dependencies for the project.

## 📊 Dataset Information
The model was trained on a combined dataset of over **90,000 images** to ensure high accuracy and robustness against different lighting conditions:
*   **MRL Eye Dataset**: Used for training eye-state detection (Open vs. Closed).
*   **Yawning Dataset**: Used for detecting mouth states related to drowsiness.
*   **Preprocessing**: All images were resized to 96x96 and normalized to the [-1, 1] range to match the MobileNetV2 architecture.

## 1. Cloud Training
To train the models without freezing your PC, use Google Colab or Kaggle:
1. Open Google Colab and set Runtime to GPU.
2. Upload the `notebooks/train_on_cloud.ipynb` file.
3. Follow the instructions in the notebook to download the Kaggle dataset.
4. Run the notebook! It will output `drowsiness_model.keras`.
5. Download the `.keras` file into the root directory.

## 2. Model Conversion
If your local environment uses Keras 2/TensorFlow 2.x, run the conversion tool:
```bash
python utils/convert_model.py
```
This will generate the `drowsiness_model.h5` file required by the detector.

## 3. Local Real-Time Detection
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the detector:
   ```bash
   python local_detector.py
   ```

### Debug Mode (The Domain Shift Fix)
By default, `DEBUG_MODE = True` in the detector. A second window will open showing exactly what the AI sees (the Left Eye, Right Eye, and Mouth). 
If these images don't look like the dataset you trained on, you can tweak the `padding_ratio` in `extract_and_preprocess()`.
