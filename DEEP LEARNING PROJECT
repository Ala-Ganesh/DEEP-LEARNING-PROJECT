# Deep Learning image classification — single Colab script

# -----------------------
# 0) Installs (optional)
# -----------------------
# Uncomment if you need to upgrade/install packages (commented to save time in Colab)
# !pip -q install --upgrade tensorflow scikit-learn matplotlib

# -----------------------
# 1) Imports & seed
# -----------------------
import os, random, time
import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

SEED = 42
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)
random.seed(SEED)

# -----------------------
# 2) Try to load CIFAR-10 (preferred)
# -----------------------
def load_cifar10_safe():
    try:
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        y_train = y_train.squeeze()
        y_test = y_test.squeeze()
        # small validation split from train
        from sklearn.model_selection import train_test_split
        x_train, x_val, y_train, y_val = train_test_split(
            x_train, y_train, test_size=0.2, random_state=SEED, stratify=y_train
        )
        # normalize
        x_train = x_train.astype("float32")/255.0
        x_val   = x_val.astype("float32")/255.0
        x_test  = x_test.astype("float32")/255.0
        class_names = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
        print("Loaded CIFAR-10 successfully.")
        return (x_train, y_train), (x_val, y_val), (x_test, y_test), class_names
    except Exception as e:
        print("CIFAR-10 load failed:", str(e))
        return None

data = load_cifar10_safe()

# -----------------------
# 3) Fallback: synthetic 2-class RGB dataset
# -----------------------
if data is None:
    print("Falling back to synthetic dataset (2 classes, 28x28 RGB).")
    IMG_SIZE = 28
    NUM_IMAGES = 2000  # total
    rng = np.random.default_rng(SEED)
    half = NUM_IMAGES // 2
    # create base noise
    red_images = rng.random((half, IMG_SIZE, IMG_SIZE, 3)).astype("float32")
    green_images = rng.random((half, IMG_SIZE, IMG_SIZE, 3)).astype("float32")
    # tint
    red_images[...,0] = np.clip(red_images[...,0] + 0.6, 0, 1)
    green_images[...,1] = np.clip(green_images[...,1] + 0.6, 0, 1)
    X = np.vstack([red_images, green_images])
    y = np.array([0]*half + [1]*half)
    # shuffle
    perm = rng.permutation(NUM_IMAGES)
    X = X[perm]; y = y[perm]
    # split train/val/test
    from sklearn.model_selection import train_test_split
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=SEED, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=SEED, stratify=y_temp)
    class_names = ["red_tinted", "green_tinted"]
    # assign to same variable names for later code
    data = ((X_train, y_train), (X_val, y_val), (X_test, y_test), class_names)

# unpack
(x_train, y_train), (x_val, y_val), (x_test, y_test), class_names = data
print("Shapes -> train:", x_train.shape, "val:", x_val.shape, "test:", x_test.shape)

# -----------------------
# 4) Build model (small, generic CNN)
# -----------------------
def build_cnn(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)
    x = layers.Rescaling(1./1, name="rescale")(inputs)  # already normalized for CIFAR but okay
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs, name="simple_cnn")
    return model

input_shape = x_train.shape[1:]
num_classes = len(class_names)
model = build_cnn(input_shape, num_classes)
model.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
model.summary()

# -----------------------
# 5) Callbacks & Train
# -----------------------
ckpt_path = "/content/best_model.keras"
callbacks = [
    keras.callbacks.ModelCheckpoint(ckpt_path, monitor="val_accuracy", save_best_only=True, verbose=1),
    keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=6, restore_best_weights=True, verbose=1)
]

EPOCHS = 25
BATCH_SIZE = 128

start_time = time.time()
history = model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=2
)
print("Training time (s):", time.time() - start_time)

# -----------------------
# 6) Load best model & evaluate
# -----------------------
try:
    best = keras.models.load_model(ckpt_path)
    print("Loaded best checkpoint.")
except Exception:
    best = model
    print("Using last model (no checkpoint loaded).")

test_loss, test_acc = best.evaluate(x_test, y_test, verbose=0)
print(f"Test accuracy: {test_acc:.4f} | Test loss: {test_loss:.4f}")

# -----------------------
# 7) Plot training curves
# -----------------------
def plot_history(h):
    plt.figure(figsize=(6,4))
    plt.plot(h.history["accuracy"], label="train_acc")
    plt.plot(h.history["val_accuracy"], label="val_acc")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend(); plt.grid(True)
    plt.title("Accuracy")
    plt.show()

    plt.figure(figsize=(6,4))
    plt.plot(h.history["loss"], label="train_loss")
    plt.plot(h.history["val_loss"], label="val_loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.grid(True)
    plt.title("Loss")
    plt.show()

plot_history(history)

# -----------------------
# 8) Confusion matrix & classification report
# -----------------------
y_pred_probs = best.predict(x_test, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)

print("Classification report:")
print(classification_report(y_test, y_pred, target_names=class_names))

cm = confusion_matrix(y_test, y_pred)
def plot_cm(cm, classes, normalize=False, title="Confusion matrix"):
    if normalize:
        cm_disp = cm.astype("float") / (cm.sum(axis=1, keepdims=True)+1e-12)
    else:
        cm_disp = cm
    plt.figure(figsize=(6,6))
    plt.imshow(cm_disp, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha="right")
    plt.yticks(tick_marks, classes)
    fmt = ".2f" if normalize else "d"
    thresh = cm_disp.max() / 2.
    for i, j in itertools.product(range(cm_disp.shape[0]), range(cm_disp.shape[1])):
        plt.text(j, i, format(cm_disp[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm_disp[i, j] > thresh else "black")
    plt.ylabel("True")
    plt.xlabel("Pred")
    plt.tight_layout()
    plt.show()

plot_cm(cm, class_names, normalize=False, title="Confusion Matrix (counts)")
plot_cm(cm, class_names, normalize=True, title="Confusion Matrix (normalized)")

# -----------------------
# 9) Show sample predictions
# -----------------------
def show_samples(X, y_true, y_pred, class_names, n=16):
    n = min(n, X.shape[0])
    idxs = np.random.choice(X.shape[0], n, replace=False)
    cols = int(np.sqrt(n))
    rows = (n + cols - 1)//cols
    plt.figure(figsize=(cols*2.2, rows*2.2))
    for i, idx in enumerate(idxs):
        plt.subplot(rows, cols, i+1)
        img = X[idx]
        # image range might be 0-1 or 0-255
        if img.max() <= 1.0:
            plt.imshow(img)
        else:
            plt.imshow(img.astype("uint8"))
        plt.title(f"T:{class_names[int(y_true[idx])]}\nP:{class_names[int(y_pred[idx])]}", fontsize=8)
        plt.axis("off")
    plt.tight_layout()
    plt.show()

show_samples(x_test, y_test, y_pred, class_names, n=16)

# -----------------------
# 10) Save final model and done
# -----------------------
final_path = "/content/model.keras"
best.save(final_path)
print("Saved final model to:", final_path)

# Quick note printed for user
print("\nAll done. If you ran this in Colab, download /content/model.keras or copy to Drive:")
print("Example: from google.colab import drive; drive.mount('/content/drive'); !cp /content/model.keras /content/drive/MyDrive/")
