# Facial Emotion Recognition System

This project is a **Facial Emotion Recognition System** that uses **Deep Learning** and **Computer Vision** techniques to detect emotions from images or real-time video streams. The system classifies faces into one of seven emotions: **angry**, **disgust**, **fear**, **happy**, **sad**, **surprise**, and **neutral**.

## Features
- Trainable CNN model for emotion classification.
- Real-time emotion detection via webcam using OpenCV.
- User-friendly interface for testing images or live feed.
- Pretrained model for quick demonstration.

---

## AI Methods Used

1. **Convolutional Neural Network (CNN)**  
   - A custom-built CNN architecture for emotion classification.
   - Includes multiple convolutional layers, max-pooling layers, and dense layers.
   - Softmax activation for multi-class classification.

2. **Image Preprocessing**  
   - Grayscale conversion and resizing of input images to 48x48 pixels.
   - Normalization to improve training efficiency.

3. **Categorical Crossentropy Loss**  
   - Loss function used to optimize multi-class classification.

4. **Real-Time Face Detection (OpenCV)**  
   - Haar cascades for face detection in live video streams.
   - Each detected face is resized and processed for emotion prediction.

---

## How It Works

1. **Data Preprocessing**
   - The FER2013 dataset is used, containing 48x48 grayscale facial images categorized into 7 emotions.
   - Images are normalized to values between 0 and 1 and reshaped for the CNN input.

2. **Model Training**
   - The CNN is trained using the **FER2013 dataset**.
   - Key architecture features:
     - Convolutional layers for feature extraction.
     - Max-pooling for dimensionality reduction.
     - Dense layers for decision-making.
   - Training is performed over 15 epochs with `adam` optimizer.

3. **Real-Time Detection**
   - The trained model is integrated with OpenCV for real-time video feed.
   - Detected faces are classified into one of the seven emotions.
   - Predictions are displayed on the live video with bounding boxes.

---

## Requirements

Install the required Python libraries:
```bash
pip install -r requirements.txt
```
## Usage

### 1. Preprocess the Dataset
Run the preprocessing script to prepare the data:
`python preprocess.py`


### 2. Train the Model
Train the CNN model:
`python train_model.py`


### 3. Test the Model
Evaluate the model on the test dataset:
`python test_model.py`


### 4. Run Real-Time Emotion Detection
Launch the real-time emotion detection using your webcam:
`python real_time_demo.py`


---

## Future Improvements
- Use a larger and more diverse dataset to improve model accuracy.
- Implement additional emotions or context-based emotion detection.
- Build a user-friendly GUI for enhanced interactivity.

---

## References
- [FER2013 Dataset](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data)
- [OpenCV Documentation](https://docs.opencv.org)
- [TensorFlow/Keras Documentation](https://www.tensorflow.org)
