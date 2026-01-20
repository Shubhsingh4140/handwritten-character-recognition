An end-to-end Handwritten Character Recognition system built using a Convolutional Neural Network (CNN) trained on the EMNIST Balanced dataset and deployed as an interactive Streamlit web application.

The application supports digits and letters, provides confidence-aware predictions, and includes a drawing canvas to improve real-world accuracy.



📌 Features

 CNN trained on EMNIST Balanced (47 classes)

 Supports digits (0–9) and letters (A–Z, selected lowercase)

 Interactive drawing canvas (high accuracy)

 Image upload support

 Confidence score with uncertainty warning

 Real-time prediction

 Clean and reproducible ML environment

 Why the Drawing Canvas?

Uploaded images often differ from the training distribution (lighting, stroke width, centering).
The drawing canvas:

Matches EMNIST handwriting style

Centers the character

Normalizes stroke width

Removes background noise

👉 This improves accuracy more than retraining the model.



EMNIST Balanced

47 character classes

Image size: 28 × 28

Grayscale

Includes visually ambiguous characters (e.g. I / l / 1, O / 0)

⚠️ Some ambiguity is unavoidable.
The app handles this using confidence-aware predictions.


Input (28×28×1)
↓
Conv2D (32, 3×3) + BatchNorm + MaxPool
↓
Conv2D (64, 3×3) + BatchNorm + MaxPool
↓
Conv2D (128, 3×3) + BatchNorm + MaxPool
↓
Flatten
↓
Dense (256) + BatchNorm + Dropout (0.5)
↓
Dense (47) + Softmax


3×3 kernels for efficient feature extraction

Increasing filters for hierarchical learning

Batch Normalization for training stability

Dropout to prevent overfitting


Optimizer: Adam

Loss: Sparse Categorical Crossentropy

Epochs: 20

Batch size: 128

Data augmentation: rotation, zoom, translation

🖥️ Application Workflow

User draws a character or uploads an image

Input is preprocessed (resize, normalize, invert)

CNN predicts class probabilities

Most likely character is shown

Confidence score is displayed

Warning appears if confidence < 70%

⚙️ Environment & Dependencies
Required Versions (Important)
Library	Version
TensorFlow	2.15.1
NumPy	1.26.4
OpenCV	4.8.1 (headless)
Streamlit	1.53.0