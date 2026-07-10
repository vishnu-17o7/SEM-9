# 17 · Facial Recognition Authentication

Implement a biometric authentication system using facial recognition techniques. Detects, recognizes, and verifies user identities using facial features.

## Quick Start

```bash
# Run the full pipeline (data + train)
python run.py all

# Or run individual steps
python run.py data       # Generate face embeddings and user database
python run.py train      # Train SVM face classifier
python run.py classify   # Run face recognition (--webcam, --image, --test)
```

## Files

| File | Purpose |
|---|---|
| `setup_face_data.py` | Generate synthetic face embeddings and user database |
| `recognize_face.py` | Face detection + SVM classifier + webcam recognition |
| `data/users.json` | Registered user database |
| `data/faces/` | Per-user embedding storage |
| `results/models/face_classifier.joblib` | Trained SVM classifier |

## How It Works

1. **Setup**: Generates 128-dim synthetic face embeddings per user + impostor samples
2. **Training**: SVM (RBF kernel) classifier trained on embeddings
3. **Recognition**: Uses cosine similarity + SVM probability for verification
4. **Fallback**: OpenCV Haar cascade for face detection + simple LBP/raw feature embedding

## Usage

```bash
cd 17-facial-recognition-auth
pip install -r requirements.txt

# 1. Setup synthetic face data
python setup_face_data.py

# 2. Train classifier and run tests
python recognize_face.py --train --test

# 3. Webcam recognition
python recognize_face.py --webcam

# 4. Image file recognition
python recognize_face.py --image path/to/face.jpg
```

Press 'q' to quit webcam mode. Green box = recognized, Red box = unknown.
