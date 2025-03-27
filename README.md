# Python Machine Learning Model to Detect Hairline Cracks
This project uses OpenCV and scikit-learn to detect cracks in images. It trains a Support Vector Machine (SVM) model on a dataset of positive (crack) and negative (no crack) images, and then uses this model to predict and detect cracks in new images.
## Dependencies

* Python 3.x
* OpenCV (`cv2`)
* NumPy (`numpy`)
* scikit-learn (`sklearn`)

To install the required libraries, run:

```bash
pip install opencv-python numpy scikit-learn
```
## Project Structure
- Crack Detection Project
  - kaggle_datasets/   **Dataset of images (Positive/Negative)** 
  - processed_images/ **Images with detected cracks (bounding boxes)** 
  - test_images/ **Images for testing the model** 
- script.py **Python script** 
## Dataset
Download the dataset here [Kaggel Dataset](https://www.kaggle.com/datasets/e4f4cce4e9557260964754f68d182178842c1b213ea62efc8fa68da2f35ca642/data)
- kaggle_datasets/
  - Positive/
    - Images/
      - image1.jpg
      - image2.png
  - Negative/
    - Images/
      - image3.jpg
      - image4.png
        
- Positive/Images/ contains images with cracks.
- Negative/Images/ contains images without cracks.
## Usage
1. Prepare your dataset: Place your image dataset in the kaggle_datasets directory, following the structure described above.
2. Prepare test images: Place the images you want to test in the test_images directory.
3. Run the script:
```bash
script.py
```
