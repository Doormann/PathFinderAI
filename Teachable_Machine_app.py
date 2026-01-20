import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.applications import MobileNetV2
from PIL import Image, ImageOps
import numpy as np
import cv2
import tkinter as tk
from tkinter import filedialog


# --- 1. FUNCTION TO REBUILD THE MODEL (TEACHABLE MACHINE VERSION) ---
def create_clean_model(weights_path, num_classes):
    """
    Rebuilds the Teachable Machine model with a clean graph.
    """
    # 1. Load the MobileNetV2 base model with pre-trained ImageNet weights
    #    This is how Teachable Machine models are designed to work.
    base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')

    # 2. Freeze the base model
    #    The base model's weights are not changed during training.
    base_model.trainable = False

    # 3. Create the new classifier head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # The final layer has a number of units equal to your number of classes
    outputs = Dense(num_classes, activation='softmax')(x)

    # 4. Create the final "clean" model
    clean_model = Model(inputs=base_model.input, outputs=outputs)

    # 5. Load ONLY the weights for the classifier head from your H5 file
    #    'by_name=True' is crucial here. It matches layers by name and ignores the mismatch.
    clean_model.load_weights(weights_path, by_name=True)

    print("Successfully built Teachable Machine model with a clean graph.")
    return clean_model


# --- 2. THE GRAD-CAM HEATMAP FUNCTION (No changes here) ---
def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    grad_model = Model(
        model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        top_pred_index = tf.argmax(preds[0])
        top_class_channel = preds[:, top_pred_index]

    grads = tape.gradient(top_class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


# --- 3. YOUR MAIN SCRIPT ---

# --- Setup ---
np.set_printoptions(suppress=True)
class_names = open("labels.txt", "r").readlines()
NUM_CLASSES = len(class_names)  # Automatically determine the number of classes

# Create the clean model and load your weights into it
model = create_clean_model("keras_Model.h5", NUM_CLASSES)

# Get image path from file dialog
root = tk.Tk()
root.withdraw()
image_path = filedialog.askopenfilename()
if not image_path:
    print("No file selected.")
    exit()

# --- Image Processing ---
original_img = cv2.imread(image_path)
original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
img_for_model = Image.open(image_path).convert("RGB")
size = (224, 224)
img_for_model = ImageOps.fit(img_for_model, size, Image.Resampling.LANCZOS)
image_array = np.asarray(img_for_model)
normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
data[0] = normalized_image_array

# --- Prediction ---
prediction = model.predict(data)
index = np.argmax(prediction)
class_name = class_names[index].strip()
confidence_score = prediction[0][index]
print("Class:", class_name)
print("Confidence Score:", confidence_score)

# --- 4. GENERATE AND DISPLAY THE HEATMAP ---
last_conv_layer_name = "out_relu"

heatmap = make_gradcam_heatmap(data, model, last_conv_layer_name)

# Resize and apply colormap
heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

# Superimpose the heatmap on the original image
superimposed_img = heatmap * 0.4 + original_img
superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)

# Display the final image
final_image = Image.fromarray(superimposed_img)
final_image.show()