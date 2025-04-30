# Real-Time Face Mask Detection

A deep learning-based system for real-time face mask detection using TensorFlow, OpenCV, and Haar Cascade classifiers. The system can detect faces and classify them into three categories: correctly worn masks, incorrectly worn masks, and no masks.

## Features

- Real-time face mask detection through webcam
- Display of confidence scores for each prediction class
- Enhanced face detection with preprocessing techniques
- Automatic frame capture and saving capability
- Semi-transparent overlay for better text visibility
- Support for both real-time and image file processing

## Project Structure

```
Real_Time_Face_Mask_Detection/
├── models/                     # Trained model files
│   ├── best_face_mask_model.h5
│   ├── best_face_mask_model.keras
│   └── best_face_mask_model1.h5
├── captured_frames/           # Directory for saved webcam frames
├── main.py                    # Main application script
├── haarcascade_frontalface_default.xml  # Face detection classifier
└── README.md
```

## Setup Instructions

1. Clone the repository:
```bash
git clone [repository-url]
cd Real_Time_Face_Mask_Detection
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Install required dependencies:
```bash
pip install tensorflow opencv-python numpy
```

## Usage

### Real-time Webcam Detection
Run the main script:
```bash
python main.py
```

Controls:
- Press 's' to save the current frame
- Press 'q' to quit the application

### Model Performance

The system uses a deep learning model trained on a diverse dataset of face mask images. The model provides confidence scores for three classes:
- Mask Worn Correctly
- Mask Worn Incorrectly
- No Mask

### Image Processing Features

- Automatic face detection using Haar Cascade classifier
- Image preprocessing including:
  - Resolution standardization (640x480)
  - Brightness and contrast enhancement
  - Gaussian blur for noise reduction
  - Adaptive face detection parameters

## Contributors:

- [David Lam](https://github.com/davidgit3000) - built and trained a model
- [Huynh Pham](https://github.com/HuynhPham0302) - adjusted the face detection model parameters
- [Minh Nhat Doan](https://github.com/nhatminh23-03) - attempted with different techniques
- [Henry Do](https://github.com/Henry1997Do) - tested the model

## Contributing

Contributions are welcome! Here's how you can help:

1. Fork the repository
2. Create a feature branch:
```bash
git checkout -b feature/YourFeature
```
3. Commit your changes:
```bash
git commit -m 'Add some feature'
```
4. Push to the branch:
```bash
git push origin feature/YourFeature
```
5. Submit a pull request

### Areas for Improvement

- Model accuracy enhancement
- Support for multiple face detection
- Additional mask types classification
- Performance optimization
- Cross-platform testing

## Acknowledgments

- TensorFlow team for the deep learning framework
- OpenCV community for computer vision tools
- Contributors and maintainers of the Haar Cascade classifiers
