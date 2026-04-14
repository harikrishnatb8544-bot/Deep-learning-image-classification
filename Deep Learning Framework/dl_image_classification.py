"""
Deep Learning Image Classification Project
CIFAR-10 Image Classification using CNN with TensorFlow/Keras
Task 2: Deep Learning Project - CODTECH Internship

This script implements a complete end-to-end deep learning pipeline for image classification.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print("="*70)
print("DEEP LEARNING IMAGE CLASSIFICATION PROJECT".center(70))
print("="*70)
print(f"\nTensorFlow Version: {tf.__version__}")
print(f"NumPy Version: {np.__version__}")
print(f"GPU Available: {len(tf.config.list_physical_devices('GPU')) > 0}")

# ==================== SECTION 1: LOAD AND EXPLORE DATA ====================
print("\n" + "="*70)
print("SECTION 1: LOADING AND EXPLORING CIFAR-10 DATASET")
print("="*70)

# Load dataset
print("\nLoading CIFAR-10 dataset...")
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

print("\n--- Dataset Information ---")
print(f"Training set shape: {x_train.shape}")
print(f"Test set shape: {x_test.shape}")
print(f"Number of classes: {len(class_names)}")
print(f"Image shape: {x_train[0].shape}")
print(f"Pixel value range: [{x_train.min()}, {x_train.max()}]")

# ==================== SECTION 2: PREPROCESS DATA ====================
print("\n" + "="*70)
print("SECTION 2: DATA PREPROCESSING")
print("="*70)

# Normalize data
x_train_normalized = x_train.astype('float32') / 255.0
x_test_normalized = x_test.astype('float32') / 255.0

# Convert labels to one-hot encoding
y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)

# Create validation split
from sklearn.model_selection import train_test_split
x_train_split, x_val, y_train_split, y_val = train_test_split(
    x_train_normalized, y_train_cat, test_size=0.2, random_state=42, stratify=y_train
)

print("\n--- Data Preprocessing Complete ---")
print(f"Training set: {x_train_split.shape}")
print(f"Validation set: {x_val.shape}")
print(f"Test set: {x_test_normalized.shape}")
print(f"Pixel range after normalization: [{x_train_normalized.min()}, {x_train_normalized.max()}]")

# ==================== SECTION 3: BUILD CNN MODEL ====================
print("\n" + "="*70)
print("SECTION 3: BUILDING CNN MODEL")
print("="*70)

model = models.Sequential([
    # Block 1
    layers.Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)),
    layers.BatchNormalization(),
    layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    # Block 2
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    # Block 3
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    # Flatten and Dense
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(64, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

print("\n--- Model Architecture ---")
model.summary()

# ==================== SECTION 4: COMPILE MODEL ====================
print("\n" + "="*70)
print("SECTION 4: COMPILING MODEL")
print("="*70)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\n--- Model Configuration ---")
print("Optimizer: Adam (learning_rate=0.001)")
print("Loss Function: Categorical Crossentropy")
print("Metrics: Accuracy")

# ==================== SECTION 5: TRAIN MODEL ====================
print("\n" + "="*70)
print("SECTION 5: TRAINING MODEL")
print("="*70)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

print("\nStarting training...")
history = model.fit(
    x_train_split, y_train_split,
    batch_size=32,
    epochs=15,
    validation_data=(x_val, y_val),
    callbacks=[early_stopping],
    verbose=1
)

print("\nTraining completed!")

# ==================== SECTION 6: EVALUATE MODEL ====================
print("\n" + "="*70)
print("SECTION 6: EVALUATING MODEL")
print("="*70)

# Test set evaluation
test_loss, test_accuracy = model.evaluate(x_test_normalized, y_test_cat, verbose=0)

print("\n--- Test Set Performance ---")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy*100:.2f}%")

# Predictions
y_pred_probs = model.predict(x_test_normalized, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)
y_test_true = np.argmax(y_test_cat, axis=1)

print("\n--- Classification Report ---")
print(classification_report(y_test_true, y_pred, target_names=class_names))

# Confusion matrix
cm = confusion_matrix(y_test_true, y_pred)

# ==================== SECTION 7: VISUALIZATIONS ====================
print("\n" + "="*70)
print("SECTION 7: GENERATING VISUALIZATIONS")
print("="*70)

# Training history
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
axes[0].set_title('Model Accuracy Over Epochs', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(history.history['loss'], label='Training Loss', linewidth=2)
axes[1].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
axes[1].set_title('Model Loss Over Epochs', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_history.png', dpi=100, bbox_inches='tight')
print("\n✓ Training history saved as 'training_history.png'")
plt.close()

# Confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names,
            cbar_kws={'label': 'Count'})
plt.title('Confusion Matrix - Test Set', fontsize=14, fontweight='bold')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=100, bbox_inches='tight')
print("✓ Confusion matrix saved as 'confusion_matrix.png'")
plt.close()

# Correct predictions
fig, axes = plt.subplots(2, 5, figsize=(14, 6))
fig.suptitle('Correct Predictions - Test Set', fontsize=14, fontweight='bold')

correct_indices = np.where(y_pred == y_test_true)[0]
selected_correct = np.random.choice(correct_indices, 10, replace=False)

for idx, ax in enumerate(axes.flat):
    test_idx = selected_correct[idx]
    ax.imshow(x_test_normalized[test_idx])
    ax.set_title(f'True: {class_names[y_test_true[test_idx]]}\nPred: {class_names[y_pred[test_idx]]}',
                fontsize=9, color='green')
    ax.axis('off')

plt.tight_layout()
plt.savefig('correct_predictions.png', dpi=100, bbox_inches='tight')
print("✓ Correct predictions saved as 'correct_predictions.png'")
plt.close()

# Misclassified predictions
fig, axes = plt.subplots(2, 5, figsize=(14, 6))
fig.suptitle('Misclassified Predictions - Test Set', fontsize=14, fontweight='bold')

misclassified_indices = np.where(y_pred != y_test_true)[0]
if len(misclassified_indices) > 0:
    selected_misclassified = np.random.choice(misclassified_indices, min(10, len(misclassified_indices)), replace=False)
    
    for idx, ax in enumerate(axes.flat):
        if idx < len(selected_misclassified):
            test_idx = selected_misclassified[idx]
            ax.imshow(x_test_normalized[test_idx])
            ax.set_title(f'True: {class_names[y_test_true[test_idx]]}\nPred: {class_names[y_pred[test_idx]]}',
                        fontsize=9, color='red')
        ax.axis('off')

plt.tight_layout()
plt.savefig('misclassified_predictions.png', dpi=100, bbox_inches='tight')
print("✓ Misclassified predictions saved as 'misclassified_predictions.png'")
plt.close()

# ==================== FINAL SUMMARY ====================
print("\n" + "="*70)
print("DEEP LEARNING MODEL - FINAL SUMMARY".center(70))
print("="*70)

print(f"\n--- Dataset Information ---")
print(f"Training Samples: {len(x_train_split)}")
print(f"Validation Samples: {len(x_val)}")
print(f"Test Samples: {len(x_test_normalized)}")
print(f"Number of Classes: 10")
print(f"Image Size: 32×32 RGB")

print(f"\n--- Model Architecture ---")
print(f"Type: Convolutional Neural Network (CNN)")
print(f"Total Parameters: {model.count_params():,}")

print(f"\n--- Training Configuration ---")
print(f"Optimizer: Adam (learning_rate=0.001)")
print(f"Loss Function: Categorical Crossentropy")
print(f"Batch Size: 128")
print(f"Epochs Trained: {len(history.history['loss'])}")

print(f"\n--- Performance Metrics ---")
print(f"Test Accuracy: {test_accuracy*100:.2f}%")
print(f"Test Loss: {test_loss:.4f}")
print(f"Best Validation Accuracy: {max(history.history['val_accuracy'])*100:.2f}%")
print(f"Misclassification Rate: {len(misclassified_indices)/len(y_test_true)*100:.2f}%")

print(f"\n--- Output Files Generated ---")
print("✓ training_history.png")
print("✓ confusion_matrix.png")
print("✓ correct_predictions.png")
print("✓ misclassified_predictions.png")

print("\n" + "="*70)
print("Deep Learning project completed successfully!".center(70))
print("="*70)
