import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from kymatio.numpy import Scattering2D

# Parameters
IMAGE_SIZE = 64
DATA_DIR = 'padded_imgs'

print("Loading and processing images...")
# Prepare class labels
class_names = sorted(os.listdir(DATA_DIR))
labels = []
features = []

# Initialize scattering transform
scattering = Scattering2D(J=2, shape=(IMAGE_SIZE, IMAGE_SIZE))

# Process each class folder
for label in class_names:
    folder_path = os.path.join(DATA_DIR, label)
    print(f"Processing class: {label}")
    for fname in os.listdir(folder_path):
        fpath = os.path.join(folder_path, fname)
        img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        img = img.astype(np.float32) / 255.0
        feat = scattering(img).flatten()
        features.append(feat)
        labels.append(label)

print(f"Processed {len(features)} images from {len(class_names)} classes")

# Convert to arrays
X = np.array(features)
y = np.array(labels)

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_categorical = to_categorical(y_encoded)

print(f"Feature shape: {X.shape}")
print(f"Classes: {le.classes_}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, stratify=y_encoded, random_state=42)

print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}")

# Build model
model = Sequential([
    Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(y_categorical.shape[1], activation='softmax')
])

model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

print("Model architecture:")
model.summary()

print("\nTraining model...")
# Train
early_stop = EarlyStopping(patience=5, restore_best_weights=True, verbose=1)
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.1,
    callbacks=[early_stop],
    verbose=1
)

# Evaluate
print("\nEvaluating model...")
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {acc:.4f}")

# Save model and label encoder
print("\nSaving model and label encoder...")
try:
    model.save("radarmodel.h5")
    print("✓ Model saved as 'radarmodel.h5'")
    
    joblib.dump(le, "label.pkl")
    print("✓ Label encoder saved as 'label.pkl'")
    
    # Verify files
    if os.path.exists("radarmodel.h5") and os.path.exists("label.pkl"):
        model_size = os.path.getsize("radarmodel.h5") / (1024*1024)  # MB
        label_size = os.path.getsize("label.pkl") / 1024  # KB
        print(f"✓ Files saved successfully!")
        print(f"  - radarmodel.h5: {model_size:.2f} MB")
        print(f"  - label.pkl: {label_size:.2f} KB")
    else:
        print("✗ Error: Files not found after saving")
        
except Exception as e:
    print(f"✗ Error saving files: {e}")

print(f"\nTraining completed!")
print(f"Final accuracy: {acc:.4f}")
print(f"Classes trained: {list(le.classes_)}")