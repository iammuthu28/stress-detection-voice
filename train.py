import os
import numpy as np
import librosa
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Bidirectional, Dense, Dropout, BatchNormalization, Flatten
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical

# Function to extract MFCC features
def extract_features(file_path, max_pad_length=100):
    y, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    
    if mfccs.shape[1] < max_pad_length:
        pad_width = max_pad_length - mfccs.shape[1]
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfccs = mfccs[:, :max_pad_length]
    
    return mfccs.T  # Output shape: (time_steps, features)

# Function to load dataset
def load_data(data_dir):
    labels = []
    features = []
    class_names = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    
    if not class_names:
        raise ValueError("No class directories found in the dataset!")

    class_mapping = {class_name: i for i, class_name in enumerate(class_names)}

    for class_name in class_names:
        class_dir = os.path.join(data_dir, class_name)
        files = [f for f in os.listdir(class_dir) if f.endswith('.wav')]

        if not files:
            print(f"Warning: No audio files found in {class_dir}")
            continue

        for file in files:
            file_path = os.path.join(class_dir, file)
            features.append(extract_features(file_path))
            labels.append(class_mapping[class_name])

    if not labels:
        raise ValueError("No valid audio files found!")

    return np.array(features), np.array(labels), class_mapping

# Set dataset path
data_dir = r"C:stress_dataset(1-8)"

# Load dataset
X, y, class_mapping = load_data(data_dir)

# Convert labels to one-hot encoding
y = to_categorical(y)   

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

# Save the scaler for later use
joblib.dump(scaler, "scaler.pkl")
joblib.dump(class_mapping, "class_mapping.pkl")

# Define the improved CNN + Bi-LSTM model
input_shape = (X.shape[1], X.shape[2])  # (time_steps, features)

model = Sequential([
    Conv1D(filters=64, kernel_size=3, activation='relu', padding='same', input_shape=input_shape),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),

    Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),

    Conv1D(filters=256, kernel_size=3, activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),

    Bidirectional(LSTM(128, return_sequences=True)),
    Dropout(0.5),

    Bidirectional(LSTM(64)),
    Dropout(0.5),

    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),

    Dense(64, activation='relu'),
    Dropout(0.3),

    Dense(y.shape[1], activation='softmax')
])

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

# Save trained model
model.save("stress_detection_model.h5")

# Evaluate model
loss, acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {acc:.2f}")

# ==========================
# TESTING ON NEW AUDIO FILE
# ==========================
def predict_stress(model_path, test_file, scaler_path, class_mapping_path):
    """Load model and predict stress level from an audio file."""
    model = load_model(model_path)
    scaler = joblib.load(scaler_path)  # Load saved scaler
    class_mapping = joblib.load(class_mapping_path)  # Load saved class mapping

    # Extract features from test audio file
    features = extract_features(test_file)

    # Normalize using the loaded scaler
    features = scaler.transform(features.reshape(-1, features.shape[-1])).reshape(features.shape)

    # Make prediction
    features = features[np.newaxis, ...]  # Add batch dimension
    prediction = model.predict(features)
    predicted_class = np.argmax(prediction)

    # Get class label
    class_labels = {v: k for k, v in class_mapping.items()}
    return class_labels.get(predicted_class, "Unknown")

# # Example test audio file
# test_audio_path = r"C:/audio_speech_actors_01-24/Actor_01/03-01-01-01-01-01-01.wav"

# # Run stress prediction
# predicted_label = predict_stress("improved_stress_detection_model.h5", test_audio_path, "scaler.pkl", "class_mapping.pkl")
# print(f"Predicted Stress Level: {predicted_label}")
