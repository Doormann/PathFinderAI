import cv2
import tkinter as tk
from tkinter import filedialog
from ultralytics import YOLO
from PIL import Image
import os

# --- CONFIGURATION ---
# Path to your custom-trained YOLOv8 model.
# This is the 'best.pt' file located in the 'runs/detect/train/weights/' folder.
MODEL_PATH = "C:/Users/ESOC-BOB/PycharmProjects/runs/detect/train11/weights/best.pt"

# Confidence threshold: Detections with a confidence score below this will be ignored.
CONFIDENCE_THRESHOLD = 0.05

def main():
    """
    Main function to run the footprint detection process.
    """
    # 1. Load your custom-trained YOLOv8 model
    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        print(f"Error loading model: {e}")
        print(f"Please make sure the file '{MODEL_PATH}' exists.")
        return

    # 2. Open a file dialog to select an image
    root = tk.Tk()
    root.withdraw()  # Hide the main tkinter window
    image_path = filedialog.askopenfilename(
        title="Select an image for detection",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")]
    )

    if not image_path:
        print("No image selected. Exiting.")
        return

    # 3. Read the selected image using OpenCV
    # We use OpenCV so we can easily draw on it later.
    original_image = cv2.imread(image_path)
    if original_image is None:
        print("Error reading the image file.")
        return

    print(f"Running detection on {os.path.basename(image_path)}...")

    # 4. Run the YOLOv8 model on the image
    results = model.predict(source=original_image, conf=CONFIDENCE_THRESHOLD)

    # The result object is a list, we work with the first element
    result = results[0]

    print(f"Found {len(result.boxes)} objects.")

    # 5. Process the detection results and draw on the image
    for box in result.boxes:
        # Get the coordinates of the bounding box
        # .xyxy[0] gets the [x1, y1, x2, y2] coordinates
        coords = box.xyxy[0].tolist()
        x1, y1, x2, y2 = map(int, coords)

        # Get the confidence score and class ID
        conf = box.conf[0].item()
        class_id = int(box.cls[0].item())
        class_name = model.names[class_id]

        # Draw the bounding box on the image
        cv2.rectangle(original_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Create the label text (class name + confidence)
        label = f"{class_name}: {conf:.2f}"

        # Calculate text size to draw a background rectangle
        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)

        # Draw a filled rectangle for the text background
        cv2.rectangle(original_image, (x1, y1 - text_height - 10), (x1 + text_width, y1), (0, 255, 0), -1)

        # Draw the text on the image
        cv2.putText(original_image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    # 6. Display the final image with detections
    # Convert the OpenCV image (BGR) to a Pillow image (RGB) for display
    final_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    final_image_pil = Image.fromarray(final_image_rgb)
    final_image_pil.show()


    output_path = "detection_result.jpg"
    cv2.imwrite(output_path, original_image)
    print(f"Result saved to {output_path}")


if __name__ == "__main__":
    main()