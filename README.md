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

Usage
1. Prepare the Data
Place the original images in a folder (e.g., Test).
Place the augmented images in subfolders under a common directory (e.g., images/ with subfolders for each augmentation type like rotated, flipped).
2. Run the Script
Execute the script to calculate FID and KID scores:

bash
Copy code
python validation_metrics.py
3. Outputs
The script prints FID and KID scores for the augmented images compared to the original dataset.
Project Structure
bash
Copy code
project-folder/
│
├── Test/                             # Directory containing original images
├── images/                           # Directory containing augmented image folders
│   ├── rotated/
│   ├── flipped/
│   ├── noisy/
│   └── color_adjusted/
│
├── validation_metrics.py             # Main script for FID and KID calculation
├── requirements.txt                  # Dependencies for the project
└── README.md                         # Project documentation
Methodology
Load and preprocess original and augmented images (resize and normalize).
Extract deep features using the pre-trained InceptionV3 model.
Calculate FID:
Compute mean and covariance for original and augmented datasets.
Use the FID formula to determine the distance between distributions.
Calculate KID:
Use polynomial kernels to compare original and augmented features.
Display FID and KID scores to assess augmentation quality.
Metrics Explanation
FID Score: Lower values indicate better similarity between original and augmented datasets.
KID Score: Like FID but uses kernel-based comparison. Zero indicates perfect similarity.
Results
FID and KID scores indicate the effectiveness of augmentation. A low FID/KID score suggests that the augmented images are close to the original dataset's distribution.
Contributing
Contributions are welcome! Feel free to fork this repository and submit pull requests for improvements or new features.

License
This project is licensed under the MIT License. See the LICENSE file for details.

Acknowledgments
InceptionV3 Model: Pre-trained model used for feature extraction.
TensorFlow/Keras: Framework for implementing the deep learning components.
