# ISL
#  ISL Static Sign Language Translator

A computer vision and deep learning project for recognizing **Indian Sign Language (ISL) static hand gestures** using **MediaPipe Hands** and a **TensorFlow/Keras neural network**.

The system detects hand landmarks from images, converts them into numerical features, and classifies the gesture into one of **36 classes (0–9 and A–Z)**.

---

## 📌 Project Overview

Communication barriers can make interaction difficult for people who rely on sign language. This project explores a computer-vision-based approach to recognizing Indian Sign Language hand gestures automatically.

The system uses **MediaPipe Hands** to extract hand landmarks from images and a fully connected neural network to classify the resulting landmark features.

The model is designed to support **two-hand gestures**, with 126 numerical features extracted from:

* 2 hands
* 21 landmarks per hand
* 3 coordinates (x, y, z) per landmark

This results in:

**2 × 21 × 3 = 126 features**

---

## ✨ Key Features

* 🤟 Indian Sign Language gesture recognition
* ✋ Two-hand gesture support
* 👁️ MediaPipe-based hand landmark detection
* 🧠 TensorFlow/Keras neural network
* 🔢 126 landmark-based input features
* 🔤 Recognition of 36 gesture classes
* 📊 Classification report and confusion matrix evaluation
* 🎥 Live gesture recognition using the trained model

---

## 🛠️ Technologies Used

| Technology         | Purpose                                         |
| ------------------ | ----------------------------------------------- |
| Python             | Core programming language                       |
| OpenCV             | Image and video processing                      |
| MediaPipe          | Hand landmark detection                         |
| NumPy              | Numerical computation                           |
| TensorFlow / Keras | Neural network training                         |
| Scikit-learn       | Label encoding, train/test split and evaluation |
| Matplotlib         | Visualization                                   |
| Seaborn            | Confusion matrix visualization                  |
| Google Colab       | Model development and training                  |

---

## 🧠 Methodology

The project follows the pipeline:

**Input Image**

↓

**Hand Detection using MediaPipe**

↓

**Extract 21 Landmarks per Hand**

↓

**Normalize Landmarks relative to Wrist**

↓

**Generate 126-Dimensional Feature Vector**

↓

**Neural Network Classification**

↓

**Predicted ISL Gesture**

---

## 🔍 Hand Landmark Feature Extraction

MediaPipe Hands is used to detect up to two hands.

For each detected hand, the project extracts 21 landmarks and their x, y and z coordinates.

The landmark coordinates are normalized relative to the wrist position to reduce the effect of hand position in the image.

For two hands:

```text
2 hands × 21 landmarks × 3 coordinates
= 126 features
```

The extracted feature vector is then used as the input to the classification model.

---

## 🧠 Neural Network Architecture

The classification model uses a fully connected neural network:

```text
Input Layer
126 features
     ↓
Dense Layer
128 neurons + ReLU
     ↓
Dropout
30%
     ↓
Dense Layer
256 neurons + ReLU
     ↓
Dropout
40%
     ↓
Dense Layer
128 neurons + ReLU
     ↓
Output Layer
36 classes + Softmax
```

The model is trained using:

* Optimizer: Adam
* Loss function: Categorical Cross-Entropy
* Batch size: 32
* Epochs: 100

The architecture and training configuration are implemented in the training notebook.

---

## 📊 Dataset

The project uses a static ISL image dataset containing **36 gesture classes**:

```text
0 1 2 3 4 5 6 7 8 9
A B C D E F G H I J K L M
N O P Q R S T U V W X Y Z
```

The feature extraction process successfully generated features for **22,854 images**.

The dataset was divided into:

* Training samples: 18,283
* Testing samples: 4,571

The split uses an 80:20 train/test ratio with stratification.

---

## 📈 Model Performance

The trained model achieved a validation accuracy of approximately **98.6%** at the final training epoch.

The independent evaluation output reports approximately **99% accuracy** on the 4,571-sample test set.

The evaluation also includes:

* Precision
* Recall
* F1-score
* Confusion Matrix

The classification report shows approximately 0.99 macro and weighted averages.

---



## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY
```

Install the required packages:

```bash
pip install -r requirements.txt
```

---

## 📦 Requirements

```text
numpy
opencv-python
mediapipe
tensorflow
scikit-learn
matplotlib
seaborn
```

---

## 🚀 How to Use

### 1. Train the Model

Open:

```text
ISL_static_model_training.ipynb
```

The notebook performs:

1. Feature loading
2. Label encoding
3. Train/test splitting
4. Neural network construction
5. Model training
6. Model saving
7. Model evaluation
8. Confusion matrix generation

### 2. Run the Live Translator

Run:

```bash
python run_live_2hand_translator.py
```

The live application uses the trained model and label mapping for gesture prediction.

---

## 📊 Evaluation

The project evaluates the model using a classification report and confusion matrix.

Example evaluation metrics include:

* Accuracy
* Precision
* Recall
* F1-score

A confusion matrix is generated to visualize predictions across the 36 gesture classes.

---

## 🔮 Future Improvements

* Extend recognition from static gestures to dynamic sign sequences
* Add continuous sentence-level recognition
* Improve robustness under different lighting conditions
* Add more ISL vocabulary
* Develop a real-time user interface
* Improve handling of gestures where MediaPipe fails to detect hands
* Explore lightweight models for low-powered devices

---

## 🎯 Learning Outcomes

Through this project, I worked with:

* Computer Vision
* Hand Landmark Detection
* Feature Engineering
* Neural Network Classification
* Model Evaluation
* Real-Time Prediction
* Python and TensorFlow
* GitHub project organization

---

## 👩‍💻 Author

**Sneha S Karun**

MS AI & Data Science

GitHub: `https://github.com/snehaskarun22`

LinkedIn: `linkedin.com/in/sneha-s-karun-ba8397330`

---


