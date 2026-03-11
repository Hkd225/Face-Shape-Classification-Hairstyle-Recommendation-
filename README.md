# Face Shape Classification & Hairstyle Recommendation
## Deep Learning & Computer Vision Pipeline
### By Muhammad Auffa Hakim Aditya



This project presents an end-to-end Computer Vision pipeline designed to automatically detect a user's face, classify its geometric shape, and provide personalized hairstyle recommendations. Utilizing Transfer Learning via EfficientNetB0 and Face Detection via MediaPipe, the system accurately categorizes faces into distinct shapes and acts as a virtual styling assistant.

The project was developed by Muhammad Auffa Hakim Aditya to demonstrate advanced deep learning techniques, combining image classification with automated preprocessing (smart face cropping) to build a practical, user-facing application.

------------------------------------------------------------

PROJECT OBJECTIVES

1. Automatically download the processed Face Shape dataset from Kaggle.
2. Build a robust image data pipeline using `tf.data`, incorporating dynamic Data Augmentation (Random Flip, Rotation, Zoom, Contrast) to prevent overfitting.
3. Handle imbalanced image classes by automatically calculating and applying balanced class weights.
4. Train a Deep Learning image classifier using Transfer Learning (EfficientNetB0 pre-trained on ImageNet).
5. Implement a two-phase training strategy:
   - Phase 1: Train a custom dense top-layer (Feature Extraction).
   - Phase 2: Unfreeze the last 60 layers of EfficientNetB0 for Fine-Tuning.
6. Integrate Google's MediaPipe Face Detection to automatically locate and crop the user's face from raw uploaded photos prior to inference.
7. Map the predicted face shape to a custom rule-based Hairstyle Recommendation Engine.

------------------------------------------------------------

DATASET INFORMATION

Source          : Kaggle (zeyadkhalid/faceshape-processed)
Domain          : Computer Vision / Fashion & Beauty
Input Data      : RGB Images (Resized to 224x224)
Classes Detected: 
- Heart
- Oblong
- Round
- Square
- Oval

------------------------------------------------------------

PIPELINE ARCHITECTURE



1. Smart Preprocessing (MediaPipe):
   Rather than feeding raw images directly into the model, the script uses `mp.solutions.face_detection` to find the exact bounding box of the face, applies an 8% margin, and crops it. This ensures the classification model focuses strictly on facial structure rather than background noise.
   

2. Model Architecture:
   - Base: EfficientNetB0 (Include_top=False)
   - GlobalAveragePooling2D
   - BatchNormalization
   - Dense (256 units, ReLU) + Dropout (0.5)
   - Output Dense (Softmax activation for multi-class prediction)

3. Recommendation Logic:
   A built-in dictionary maps the winning class prediction to actionable styling advice, providing lists of "Recommended" and "Avoid" hairstyles.

------------------------------------------------------------

MODEL EVALUATION & EXPORT

The model evaluates its performance using a detailed Classification Report and visualizes the results using a Confusion Matrix. 

Exported Artifacts:
- best_face_shape_model.keras (Saved automatically via ModelCheckpoint)
- final_face_shape_model.keras (Final fine-tuned weights)
- label_map.json (Dictionary mapping numeric indices back to string class names)
- prediction_log.csv (A saved log of user inferences and recommendations)

------------------------------------------------------------

INSTALLATION

Install the required dependencies:

pip install tensorflow pandas numpy matplotlib scikit-learn opencv-python-headless mediapipe kagglehub

------------------------------------------------------------

HOW TO RUN

1. Clone this repository:
   git clone https://github.com/YOUR_USERNAME/face-shape-classifier.git

2. Install the required libraries.
3. Run the Python script or Google Colab Notebook.
4. The script will train the model and prompt you to upload a test image. Upload any photo containing a face, and the system will auto-crop it, predict the shape, and print out hairstyle recommendations.

------------------------------------------------------------

AUTHOR

Muhammad Auffa Hakim Aditya

This project was developed as an exploration of:
- Computer Vision and Image Classification
- Transfer Learning & Fine-Tuning (EfficientNet)
- Object Detection / Face Cropping (MediaPipe, OpenCV)
- TensorFlow `tf.data` API & Data Augmentation
- Rule-based Recommendation Systems

------------------------------------------------------------

KEYWORDS 

- Muhammad Auffa Hakim Aditya
- Computer Vision
- Face Shape Classification
- EfficientNetB0 TensorFlow
- MediaPipe Face Detection
- Deep Learning Portfolio
