# **Image Augmentation Validation using FID and KID**

## **Overview**
This project evaluates the quality of augmented images generated through data augmentation techniques by calculating **Fréchet Inception Distance (FID)** and **Kernel Inception Distance (KID)**. These metrics compare the statistical similarity between the original images and augmented images using features extracted from the **InceptionV3 model**.

## **Key Features**
- Calculates FID and KID scores for augmented image datasets.
- Supports various augmentation methods (rotation, flipping, noise addition, etc.).
- Efficient feature extraction using the **InceptionV3 model**.
- Implements multiprocessing for faster feature extraction.
- Results indicate the fidelity and diversity of augmented images.

---

## **Requirements**
To run this project, you need the following software and libraries:

- **Python 3.8 or later**
- **TensorFlow 2.x**
- Required Python packages:
  - `numpy`
  - `scipy`
  - `tensorflow`
  - `Pillow`

Install the dependencies using:
```bash
pip install -r requirements.txt
