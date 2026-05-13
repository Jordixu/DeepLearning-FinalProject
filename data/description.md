# Dataset descriptions

> [!NOTE] The information contained here are extracted from the original webs.

## ASL Hand Signs Dataset

Link: [https://www.kaggle.com/datasets/vignonantoine/combinedasldatasets?select=combine_asl_dataset]

This dataset contains images representing American Sign Language (ASL) hand signs for numbers (0–9) and letters (A–Z). Each image is a labeled representation of an individual ASL sign, providing a valuable resource for building machine-learning models related to gesture recognition and sign language translation.

Features:

- Categories: 36 classes (26 letters + 10 numbers).
- Image Data: Consistent resolution across all samples, with a uniform background for better preprocessing.
- Applications:
  - Gesture recognition systems.
  - Sign language translation tools.
  - Human-computer interaction models.

Format: Images are organized by labels, with corresponding annotations for each class.

This dataset is ideal for practicing computer vision techniques like classification, feature extraction, and model evaluation.

### About the directory

The dataset is organized into folders, with each folder representing a specific class (e.g., 0, 1, A, B, etc.).
Each folder contains multiple 400x400 pixel image files corresponding to that class.
File names are unique and consistent within their respective folders for easy identification and traceability.

> [!NOTE] The dataset contains augmented data (specifically, cropped pictures).


## ASL-HG: American Sign Language Hand Gesture Image Dataset

Link: [https://data.mendeley.com/datasets/j4y5w2c8w9/1]

This dataset provides a comprehensive collection of American Sign Language (ASL) hand gesture images designed to support research in gesture recognition, computer vision, deep learning, and assistive communication technologies. The dataset consists of 36,000 high-resolution JPG images across 36 ASL classes, covering the full English alphabet (A–Z) and digits (0–9).

Data were collected from 10 volunteers in Mirpur, Dhaka, Bangladesh during May–June 2025. Each participant contributed 100 images per class, producing a balanced dataset of 1,000 images for each gesture category. Images were captured using smartphone HD cameras in both indoor and outdoor environments to ensure diversity in lighting, backgrounds, skin tones, and hand orientations.

To avoid class confusion between the visually similar gestures for the letter “O” and the digit “0”, the dataset explicitly includes the standard two-handed ASL sign for “zero,” which is commonly used in real-world alphanumeric communication. This distinction supports more accurate gesture-based recognition across alphabetic and numeric classes.

With its balanced distribution, high quality, and dual-format availability (raw + processed), this dataset stands as a state-of-the-art ASL gesture resource. It is suitable for research in sign language recognition, assistive technology, human–computer interaction, gesture-controlled systems, and pattern recognition benchmarking.

### About the directory

The dataset is provided under the root directory “ASL_HG_36000”, which contains two separate ZIP files:

ASL_Raw_Images.zip

1. Contains the original unprocessed gesture images.

2. Preserves natural variations in lighting, background, angle, and hand shape.

ASL_Processed_Images.zip

1. Includes MediaPipe-segmented hand regions with clean backgrounds.

2. Organized into predefined train–test splits (80% training, 20% testing).

3. Provides standardized images suitable for direct model training.

Each ZIP file contains 36 subfolders representing the 36 gesture classes, making the dataset well-structured and easy to integrate into computer vision pipelines.

## ASL Alphabet

Link: [https://www.kaggle.com/datasets/grassknoted/asl-alphabet/data?select=asl_alphabet_train]

The data set is a collection of images of alphabets from the American Sign Language, separated in 29 folders which represent the various classes.

The training data set contains 87,000 images which are 200x200 pixels. There are 29 classes, of which 26 are for the letters A-Z and 3 classes for SPACE, DELETE and NOTHING.
These 3 classes are very helpful in real-time applications, and classification.
The test data set contains a mere 29 images, to encourage the use of real-world test images.

> [!WARNING] It contains SPACE, DELETE and NOTHING; which should be deleted since we do not have this information in the other datasets and this one is not big enough.

### Discarted data

