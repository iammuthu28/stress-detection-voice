# Stress Detection from Voice using Deep Learning

This project detects **human emotions from speech** and classifies them into 8 categories:
- Angry  
- Calm  
- Disgust  
- Fearful  
- Happy  
- Neutral  
- Sad  
- Surprised  

Achieved **94% accuracy** using **MFCC feature extraction** + **CNN + BiLSTM** hybrid model.  

---

## Dataset
- Dataset: [RAVDESS Emotional Speech Audio Dataset](https://zenodo.org/record/1188976)  
- Contains 24 professional actors (12 male, 12 female) vocalizing statements with different emotions.  

---

## Tech Stack
- **Python 3.11**
- **Librosa** for MFCC feature extraction  
- **TensorFlow/Keras** for CNN + BiLSTM  
- **Scikit-learn** for preprocessing  
- **Joblib** for saving scalers  
- **Matplotlib/Seaborn** for evaluation  

---

## Project Workflow
1. Extract MFCC features from audio files  
2. Normalize features with `StandardScaler`  
3. Train CNN + BiLSTM model  
4. Save trained model (`model.h5`), scaler, and class mapping  
5. Test on new audio files for real-time emotion detection  

---

## Results
- Model Accuracy: **94%**  
- Evaluation: Tested on unseen audio samples  

---

## OUTPUT

1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 440ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 33ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 23ms/step
kishore1.wav: Predicted Stress Level -> Sad
kishore2.wav: Predicted Stress Level -> Disgust
kishore3.wav: Predicted Stress Level -> Fearful