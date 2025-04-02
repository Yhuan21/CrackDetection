import sys
import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DIR_DATASET = os.path.join(BASE_DIR, "kaggle_datasets")
DIR_PROCESSED_IMAGES = os.path.join(BASE_DIR, "processed_images")
DIR_TEST_IMAGES = os.path.join(BASE_DIR, "test_images")

for directory in [DIR_DATASET, DIR_PROCESSED_IMAGES, DIR_TEST_IMAGES]:
    os.makedirs(directory, exist_ok=True)


def load_images(dataset_path):
    images = []
    labels = []
    categories = {"Positive": 1, "Negative": 0}
    for category, label in categories.items():
        folder_path = os.path.join(dataset_path, category, "Images")
        if not os.path.exists(folder_path):
            print(f"Warning: {folder_path} does not exist. Skipping.")
            continue

        for filename in os.listdir(folder_path):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img is not None:
                img = cv2.resize(img, (100, 100))
                img = img.flatten().astype(np.float32) / 255.0
                images.append(img)
                labels.append(label)

    return np.array(images), np.array(labels)


def detect_cracks(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blurred, 50, 150)

    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)
    eroded = cv2.erode(dilated, kernel, iterations=1)

    contours, _ = cv2.findContours(eroded,
                                   cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    min_length = 80
    min_aspect_ratio = 4
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = max(w, h) / (min(w, h) + 1)
        length = max(w, h)

        if length > min_length and aspect_ratio > min_aspect_ratio:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img,
                        f"Length: {length}px",
                        (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1)

    output_path = os.path.join(DIR_PROCESSED_IMAGES,
                               os.path.basename(image_path))
    cv2.imwrite(output_path, img)
    print(f"Image with bounding boxes saved to {output_path}")


def predict_image(image_path, model):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return "Unable to read image"

    img_resized = cv2.resize(img, (100, 100))
    img_flattened = img_resized.flatten().astype(np.float32) / 255.0
    img_flattened = img_flattened.reshape(1, -1)
    prediction = model.predict(img_flattened)

    if prediction[0] == 1:
        detect_cracks(image_path)
        return "Crack detected"
    else:
        return "No crack detected"


def train_model(DIR_DATASET):
    X, y = load_images(DIR_DATASET)
    X_train, X_test, y_train, y_test = (
        train_test_split(X, y,
                         test_size=0.2,
                         random_state=42))
    model = SVC(kernel="rbf", class_weight="balanced")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy * 100:.2f}%")
    return model


model = train_model(DIR_DATASET)

user_input = input("Choose an option below (1) Test all images "
                   "or (2) Enter a test image file name: ")
number = int(user_input)
test_images_folder = DIR_TEST_IMAGES

if number == 1:
    for filename in os.listdir(test_images_folder):
        if filename.endswith(('.jpg',
                              '.png',
                              '.jpeg')):
            test_image_path = os.path.join(test_images_folder, filename)
            prediction = predict_image(test_image_path, model)
            print(f"Prediction for {filename}: {prediction}")
elif number == 2:
    image_filename = input("Enter the test image file name "
                           "(e.g., image.jpg): ")
    test_image_path = os.path.join(test_images_folder, image_filename)
    result = predict_image(test_image_path, model)
    print(f"Prediction for the test image: {result}")
else:
    print("Invalid option.")
    sys.exit(1)
