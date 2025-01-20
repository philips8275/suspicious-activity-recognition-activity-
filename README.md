download dataset https://www.kaggle.com/datasets/naveenk903/movies-fight-detection-dataset
# Suspicious Activity Detection Using LRCN Model

This project utilizes a Long-term Recurrent Convolutional Network (LRCN) to detect suspicious human activities in videos. The dataset is preprocessed and the model is trained to classify activities as `fights` or `noFights`. Below are the key components and steps:

---

## Requirements
Install dependencies:
```bash
pip install pafy youtube-dl moviepy opencv-python tensorflow
```

---

## Features
1. **Dataset Preparation**  
   - Extracts frames from videos, resizes, normalizes, and processes them.
   - Uses a sequence length of 30 frames for analysis.

2. **Model Architecture**  
   - Combines CNNs for feature extraction and LSTMs for sequence processing.
   - Output layer uses `softmax` for binary classification.

3. **Training**  
   - Trained with categorical cross-entropy loss and Adam optimizer.
   - Implements early stopping for optimal performance.

4. **Evaluation**  
   - Metrics include accuracy, precision, recall, and F1 score.

---

## Usage

1. **Train the Model**  
   Run the script to preprocess the dataset and train the model.  
   ```python
   model_training_history = model.fit(...)
   ```

2. **Save and Load Model**  
   ```python
   model.save("Suspicious_Human_Activity_Detection_LRCN_Model.h5")
   ```

3. **Test Model**  
   Evaluate accuracy and generate performance metrics on the test dataset.

---

## Results
- **Accuracy**: Achieved on the test dataset.  
- **F1 Score**: Indicates balanced performance between precision and recall.

---

## Visualization
- Training and validation loss/accuracy graphs for monitoring progress.  
- Displayed video frames annotated with detected activity class.  

---

This project demonstrates the integration of computer vision and deep learning for real-time suspicious activity detection.

