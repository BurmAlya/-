import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import random


print("=" * 100)
print("🚀 ГИБРИДНАЯ CNN-LSTM МОДЕЛЬ: Анализ эмоциональной направленности отзывов")
print("=" * 100)


# ============================================================
# 1. КОНФИГУРАЦИЯ
# ============================================================
config = {
    "max_words": 10000,
    "max_len": 300,              # Увеличено для лучшего охвата контекста
    "train_size": 0.8,
    "val_size": 0.1,
    "embedding_dim": 128,
    "cnn_filters": 64,           # Количество CNN фильтров
    "kernel_size": 5,            # Размер kernel для CNN
    "lstm_units": 128,
    "dense_units": 128,
    "dropout_rate": 0.3,
    "spatial_dropout": 0.2,
    "batch_size": 32,
    "epochs": 50,
    "initial_learning_rate": 1e-3,
    "optimizer": "adam",
    "loss": "binary_crossentropy",
    "early_stopping_patience": 6,
    "reduce_lr_patience": 3,
    "reduce_lr_factor": 0.5,
    "random_seed": 42
}


tf.random.set_seed(config["random_seed"])
np.random.seed(config["random_seed"])
random.seed(config["random_seed"])


print("⚙️ Конфигурация гибридной модели:")
for k, v in config.items():
    print(f"  {k}: {v}")


# ============================================================
# 2. Загрузка датасета
# ============================================================
print("\n📚 ЗАГРУЗКА ДАТАСЕТА IMDB...")

from tensorflow.keras.datasets import imdb

(x_train_raw, y_train_raw), (x_test_raw, y_test_raw) = imdb.load_data(num_words=config["max_words"])
print(f"✅ Загружено {len(x_train_raw) + len(x_test_raw)} отзывов")


# ============================================================
# 3. Подготовка данных
# ============================================================
from tensorflow.keras.preprocessing.sequence import pad_sequences

all_data = np.concatenate([x_train_raw, x_test_raw], axis=0)
all_labels = np.concatenate([y_train_raw, y_test_raw], axis=0)

all_data = pad_sequences(all_data, maxlen=config["max_len"], padding="post", truncating="post")

num_samples = len(all_data)
train_end = int(num_samples * config["train_size"])
val_end = int(num_samples * (config["train_size"] + config["val_size"]))

x_train = all_data[:train_end]
y_train = all_labels[:train_end]

x_val = all_data[train_end:val_end]
y_val = all_labels[train_end:val_end]

x_test = all_data[val_end:]
y_test = all_labels[val_end:]

print("\n✅ Размеры выборок:")
print(f"  Обучающая: {len(x_train)}")
print(f"  Валидационная: {len(x_val)}")
print(f"  Тестовая: {len(x_test)}")


# tf.data пайплайны
def make_dataset(features, labels, batch_size, shuffle=True):
    ds = tf.data.Dataset.from_tensor_slices((features, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(features), reshuffle_each_iteration=True)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


train_ds = make_dataset(x_train, y_train, config["batch_size"], shuffle=True)
val_ds = make_dataset(x_val, y_val, config["batch_size"], shuffle=False)
test_ds = make_dataset(x_test, y_test, config["batch_size"], shuffle=False)


# ============================================================
# 4. ГИБРИДНАЯ CNN-LSTM МОДЕЛЬ
# ============================================================
def build_cnn_lstm_model(config):
    """
    Гибридная модель, которая комбинирует:
    - CNN для извлечения локальных признаков
    - LSTM для понимания последовательных зависимостей
    """
    inputs = keras.Input(shape=(config["max_len"],), name="text_input")

    # Embedding layer
    x = layers.Embedding(
        input_dim=config["max_words"],
        output_dim=config["embedding_dim"],
        input_length=config["max_len"],
        mask_zero=False
    )(inputs)
    x = layers.SpatialDropout1D(config["spatial_dropout"])(x) 
# CNN слои для извлечения локальных паттернов
    x = layers.Conv1D(
        filters=config["cnn_filters"],
        kernel_size=config["kernel_size"],
        activation='relu',
        padding='same'
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(config["dropout_rate"])(x)

    # Второй CNN слой
    x = layers.Conv1D(
        filters=config["cnn_filters"] * 2,
        kernel_size=config["kernel_size"],
        activation='relu',
        padding='same'
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(config["dropout_rate"])(x)

    # Bidirectional LSTM для последовательных зависимостей
    x = layers.Bidirectional(
        layers.LSTM(config["lstm_units"], return_sequences=False)
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(config["dropout_rate"])(x)

    # Dense слои для классификации
    x = layers.Dense(config["dense_units"], activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(config["dropout_rate"])(x)

    outputs = layers.Dense(1, activation="sigmoid", name="sentiment_output")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="cnn_lstm_sentiment")

    # ВАЖНО: числовой learning_rate, без расписания
    opt = keras.optimizers.Adam(learning_rate=config["initial_learning_rate"])

    model.compile(
        optimizer=opt,
        loss=config["loss"],
        metrics=[
            "accuracy",
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.AUC(name="auc")
        ]
    )

    return model


print("\n🏗 ПОСТРОЕНИЕ ГИБРИДНОЙ CNN-LSTM МОДЕЛИ...")
model = build_cnn_lstm_model(config)
model.summary()


# ============================================================
# 5. Колбэки
# ============================================================
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=config["early_stopping_patience"],
    restore_best_weights=True,
    verbose=1
)

reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=config["reduce_lr_factor"],
    patience=config["reduce_lr_patience"],
    min_lr=1e-7,
    verbose=1
)


# ============================================================
# 6. Обучение
# ============================================================
print("\n🎓 ОБУЧЕНИЕ ГИБРИДНОЙ МОДЕЛИ...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=config["epochs"],
    callbacks=[early_stopping, reduce_lr],
    verbose=1,
    steps_per_epoch=100
)

print("✅ Обучение завершено!\n")


# ============================================================
# 7. Оценка на тесте
# ============================================================
print("📊 ОЦЕНКА НА ТЕСТОВОЙ ВЫБОРКЕ...")
test_metrics = model.evaluate(test_ds, verbose=1)
print("\n" + "=" * 60)
print("ФИНАЛЬНЫЕ МЕТРИКИ НА ТЕСТОВОЙ ВЫБОРКЕ:")
print("=" * 60)
for name, value in zip(model.metrics_names, test_metrics):
    print(f"  {name.upper()}: {value:.4f}")
print("=" * 60)


# ============================================================
# 8. ДЕТАЛЬНЫЙ АНАЛИЗ ПРЕДСКАЗАНИЙ
# ============================================================
print("\n🔍 ДЕТАЛЬНЫЙ АНАЛИЗ ПРЕДСКАЗАНИЙ:\n")

# Получаем словарь для декодирования
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review if i >= 3])

# Выбираем разные примеры
num_examples = 15
sample_indices = np.random.choice(len(x_test), num_examples, replace=False)

predictions = model.predict(x_test[sample_indices], verbose=0)
# Создаём DataFrame для лучшей визуализации
results = []
for idx, i in enumerate(sample_indices):
    review_text = decode_review(x_test[i])
    real_label = "Позитив" if y_test[i] == 1 else "Негатив"
    pred_prob = predictions[idx][0]
    pred_label = "Позитив" if pred_prob > 0.5 else "Негатив"
    correct = "✓" if real_label == pred_label else "✗"

    # Первые 150 символов отзыва
    review_short = (review_text[:147] + '...') if len(review_text) > 150 else review_text

    results.append({
        '№': idx + 1,
        'Отзыв (первые 150 символов)': review_short,
        'Реальный класс': real_label,
        'Предсказанный класс': pred_label,
        'Вероятность': f'{pred_prob:.4f}',
        'Верно': correct
    })

# Выводим таблицу
df_results = pd.DataFrame(results)
print(df_results.to_string(index=False))

# Статистика точности на примерах
correct_count = sum(1 for r in results if r['Верно'] == '✓')
print(f"\n\n📈 Точность на выборке из {num_examples} примеров: {correct_count}/{num_examples} ({100*correct_count/num_examples:.1f}%)")

# Сохраняем результаты
df_results.to_csv('prediction_examples.csv', index=False, encoding='utf-8')
print("\n✅ Примеры сохранены в файл: prediction_examples.csv")


 #============================================================
# 9. Визуализация
# ============================================================
train_loss = history.history["loss"]
val_loss = history.history["val_loss"]
train_acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]

# Loss
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(train_loss, label="train_loss")
plt.plot(val_loss, label="val_loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(train_acc, label="train_acc")
plt.plot(val_acc, label="val_acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('cnn_lstm_training_history.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n✅ Графики сохранены в файл: cnn_lstm_training_history.png")

# Сохраняем модель
model.save('cnn_lstm_sentiment_model.keras')
print("\n✅ Модель сохранена: cnn_lstm_sentiment_model.keras")
