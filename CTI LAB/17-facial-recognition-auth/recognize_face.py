"""
Facial recognition authentication system.

Uses OpenCV face detection + embedding-based recognition.
Supports:
- Webcam-based face capture & recognition
- Image file-based recognition
- Synthetic embedding matching (when no camera)
"""

import json
from pathlib import Path

import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from joblib import dump, load

DATA_DIR = Path(__file__).parent / "data"
RESULTS_DIR = Path(__file__).parent / "results"
MODEL_DIR = RESULTS_DIR / "models"
EMBEDDING_DIM = 128
SIMILARITY_THRESHOLD = 0.50


def load_user_db():
    """Load registered users from database."""
    db_path = DATA_DIR / "users.json"
    if not db_path.exists():
        print("No user database found. Run setup_face_data.py first.")
        return []
    with open(db_path) as f:
        data = json.load(f)
    return data["users"]


def load_all_embeddings(users):
    """Load all user embeddings for training the classifier."""
    X, y = [], []
    user_ids = []
    for user in users:
        emb_path = DATA_DIR / "faces" / user["name"].replace(" ", "_") / "embedding.npy"
        if emb_path.exists():
            emb = np.load(emb_path)
            X.append(emb)
            y.append(user["name"])
            user_ids.append(user["user_id"])

    # Add impostor samples
    imp_dir = DATA_DIR / "faces" / "impostors"
    if imp_dir.exists():
        for imp_path in sorted(imp_dir.glob("*.npy")):
            emb = np.load(imp_path)
            X.append(emb)
            y.append("UNKNOWN")
            user_ids.append("unknown")

    return np.array(X), np.array(y), user_ids


def train_classifier():
    """Train an SVM classifier on face embeddings."""
    print("  Training face recognition classifier...")
    X, y, user_ids = load_all_embeddings(load_user_db())
    if len(X) == 0:
        print("  No training data available.")
        return None

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    clf = SVC(kernel="rbf", C=10, gamma="scale", probability=True, random_state=42)
    clf.fit(X, y_enc)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    dump({"model": clf, "label_encoder": le, "user_ids": user_ids}, MODEL_DIR / "face_classifier.joblib")
    print(f"  Trained on {len(X)} samples, {len(le.classes_)} classes")
    return clf, le


def recognize_from_embedding(embedding, clf=None, le=None, users=None):
    """Recognize a face from its embedding vector."""
    if users is None:
        users = load_user_db()
    if users is None or len(users) == 0:
        return {"name": "Unknown", "confidence": 0, "recognized": False}

    if clf is None or le is None:
        # Load classifier
        model_path = MODEL_DIR / "face_classifier.joblib"
        if model_path.exists():
            data = load(model_path)
            clf = data["model"]
            le = data["label_encoder"]
        else:
            # Use cosine similarity fallback
            return recognize_cosine_fallback(embedding, users)

    embedding = embedding.reshape(1, -1)
    pred = clf.predict(embedding)[0]
    probas = clf.predict_proba(embedding)[0]
    confidence = float(max(probas))
    name = le.inverse_transform([pred])[0]

    # Find matching user
    user_info = None
    for u in users:
        if u["name"] == name:
            user_info = u
            break

    return {
        "name": name,
        "user_id": user_info["user_id"] if user_info else None,
        "role": user_info["role"] if user_info else None,
        "confidence": round(confidence, 4),
        "recognized": confidence >= SIMILARITY_THRESHOLD and name != "UNKNOWN",
    }


def recognize_cosine_fallback(embedding, users):
    """Fallback: recognize using cosine similarity against stored embeddings."""
    best_sim = -1
    best_user = None

    for user in users:
        emb_path = DATA_DIR / "faces" / user["name"].replace(" ", "_") / "embedding.npy"
        if not emb_path.exists():
            continue
        stored = np.load(emb_path)
        sim = cosine_similarity(embedding.reshape(1, -1), stored.reshape(1, -1))[0][0]
        if sim > best_sim:
            best_sim = sim
            best_user = user

    return {
        "name": best_user["name"] if best_user else "Unknown",
        "user_id": best_user["user_id"] if best_user else None,
        "role": best_user["role"] if best_user else None,
        "confidence": round(float(best_sim), 4),
        "recognized": best_sim >= SIMILARITY_THRESHOLD,
    }


def detect_face_from_image(image_path):
    """Detect face from an image file and return embedding."""
    img = cv2.imread(str(image_path))
    if img is None:
        return None, "Cannot read image"

    # Use Haar cascade for face detection
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(50, 50))

    if len(faces) == 0:
        return None, "No face detected"

    # Take the largest face
    (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])
    face_roi = gray[y:y + h, x:x + w]
    face_resized = cv2.resize(face_roi, (100, 100))

    # Generate a simple embedding from LBP + HOG-like features
    embedding = extract_simple_embedding(face_resized)

    return embedding, face_resized


def extract_simple_embedding(face_img):
    """Extract a simple face embedding from a detected face region.
    
    Uses LBP-like features + raw pixel features as a simple embedding.
    For production, use dlib/FaceNet/DeepFace models.
    """
    # LBP-like features
    lbp = np.zeros(256)
    h, w = face_img.shape
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            center = face_img[i, j]
            code = 0
            code |= (face_img[i - 1, j - 1] > center) << 7
            code |= (face_img[i - 1, j] > center) << 6
            code |= (face_img[i - 1, j + 1] > center) << 5
            code |= (face_img[i, j + 1] > center) << 4
            code |= (face_img[i + 1, j + 1] > center) << 3
            code |= (face_img[i + 1, j] > center) << 2
            code |= (face_img[i + 1, j - 1] > center) << 1
            code |= (face_img[i, j - 1] > center) << 0
            lbp[code] += 1
    lbp = lbp / np.sum(lbp)

    # Downsampled raw pixels
    raw = cv2.resize(face_img, (16, 16)).flatten() / 255.0

    embedding = np.concatenate([lbp, raw]).astype(np.float32)
    embedding = embedding / np.linalg.norm(embedding)
    return embedding


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Facial Recognition Authentication")
    parser.add_argument("--train", action="store_true", help="Train the classifier")
    parser.add_argument("--image", help="Recognize face from image file")
    parser.add_argument("--webcam", action="store_true", help="Use webcam for recognition")
    parser.add_argument("--test", action="store_true", help="Run quick test with synthetic data")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)

    if args.train or not (MODEL_DIR / "face_classifier.joblib").exists():
        clf, le = train_classifier()

    if args.test:
        print("\n  Running recognition test...")
        users = load_user_db()
        model_path = MODEL_DIR / "face_classifier.joblib"
        if model_path.exists():
            data = load(model_path)
            clf, le = data["model"], data["label_encoder"]
        else:
            clf, le = None, None

        # Test with each user's stored embedding
        for user in users:
            emb_path = DATA_DIR / "faces" / user["name"].replace(" ", "_") / "embedding.npy"
            if emb_path.exists():
                emb = np.load(emb_path)
                result = recognize_from_embedding(emb, clf, le, users)
                status = "RECOGNIZED" if result["recognized"] else "FAILED"
                print(f"  [{status:10s}] {user['name']:20s} confidence: {result['confidence']:.4f}")

        # Test with impostor
        imp_dir = DATA_DIR / "faces" / "impostors"
        if imp_dir.exists():
            for imp_path in sorted(imp_dir.glob("*.npy"))[:3]:
                emb = np.load(imp_path)
                result = recognize_from_embedding(emb, clf, le, users)
                status = "REJECTED" if not result["recognized"] else "FALSE_ACCEPT"
                print(f"  [{status:10s}] {'Impostor':20s} confidence: {result['confidence']:.4f}")

    if args.image:
        emb, face = detect_face_from_image(args.image)
        if emb is not None:
            users = load_user_db()
            result = recognize_from_embedding(emb, users=users)
            print(f"\n  Recognition Result:")
            print(f"    Name:       {result['name']}")
            print(f"    Role:       {result['role']}")
            print(f"    Confidence: {result['confidence']:.2%}")
            print(f"    Access:     {'GRANTED' if result['recognized'] else 'DENIED'}")
        else:
            print(f"  Face detection failed: {face}")

    if args.webcam:
        print("\n  Starting webcam face recognition (press 'q' to quit)...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("  Cannot open webcam.")
            return

        users = load_user_db()
        model_path = MODEL_DIR / "face_classifier.joblib"
        if model_path.exists():
            data = load(model_path)
            clf, le = data["model"], data["label_encoder"]
        else:
            clf, le = None, None

        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))

            for (x, y, w, h) in faces:
                face_roi = gray[y:y + h, x:x + w]
                face_resized = cv2.resize(face_roi, (100, 100))
                emb = extract_simple_embedding(face_resized)
                result = recognize_from_embedding(emb, clf, le, users)

                color = (0, 255, 0) if result["recognized"] else (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                label = f"{result['name']} ({result['confidence']:.0%})"
                cv2.putText(frame, label, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            cv2.imshow("Face Recognition Auth", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

    if not any([args.train, args.test, args.image, args.webcam]):
        parser.print_help()


if __name__ == "__main__":
    main()
