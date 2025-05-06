import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Reshape, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import mlflow
import mlflow.tensorflow

# 🔧 경로 설정
base_dir = "hyochan/model_test/dataset/outputs/cnn_lstm"
X_path = os.path.join(base_dir, "X_lstm.npy")
y_path = os.path.join(base_dir, "y_lstm.npy")
model_save_path = os.path.join(base_dir, "cnn_lstm_model.h5")
plot_save_path = os.path.join(base_dir, "train_history_2class_segment2s.png")

# 📥 데이터 로드
X = np.load(X_path)
y = np.load(y_path)
print(f"✅ Data loaded: X shape = {X.shape}, y shape = {y.shape}")
print(f"🧾 Label distribution: {np.bincount(y)}")

X = X[..., np.newaxis]  # CNN 입력형태

# 📊 데이터 분할 (7:2:1)
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=2/9, stratify=y_temp, random_state=42)

# 🧠 실험 추적 시작
mlflow.set_experiment("CNN_LSTM_2class")

with mlflow.start_run():
    # 🔧 하이퍼파라미터 기록
    mlflow.log_param("model_type", "CNN+LSTM")
    mlflow.log_param("epochs", 30)
    mlflow.log_param("batch_size", 32)

    # 🧠 모델 정의
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(X.shape[1], X.shape[2], 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Reshape((X.shape[1] // 4, -1)),
        LSTM(64),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(2, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # 🏁 학습 시작
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=30,
        batch_size=32,
        callbacks=[early_stop],
        verbose=1
    )

    # 🎯 최종 평가
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"🧪 Test Accuracy: {test_acc:.4f}, Loss: {test_loss:.4f}")

    # 📈 시각화
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Acc', marker='o')
    plt.plot(history.history['val_accuracy'], label='Val Acc', marker='x')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss', marker='o')
    plt.plot(history.history['val_loss'], label='Val Loss', marker='x')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(plot_save_path)
    plt.show()
    print(f"📈 Plot saved: {plot_save_path}")

    # 💾 모델 및 파일 저장
    model.save(model_save_path)
    mlflow.tensorflow.log_model(model, "model")
    mlflow.log_metric("test_accuracy", test_acc)
    mlflow.log_metric("test_loss", test_loss)
    mlflow.log_artifact(plot_save_path)
    mlflow.log_artifact(model_save_path)

    print("✅ MLflow logging complete.")
