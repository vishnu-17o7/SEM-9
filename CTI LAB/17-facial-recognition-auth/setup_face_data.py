"""
Setup face data for facial recognition authentication.

Generates synthetic face embeddings and creates user profiles
for the facial recognition system. When a webcam is available,
it can capture real face data instead.
"""

import base64
import hashlib
import json
import random
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).parent / "data"
RESULTS_DIR = Path(__file__).parent / "results"
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Sample user profiles for the system
REGISTERED_USERS = [
    {"name": "Alice Johnson", "role": "Admin", "email": "alice@company.com"},
    {"name": "Bob Smith", "role": "Engineer", "email": "bob@company.com"},
    {"name": "Carol Davis", "role": "HR", "email": "carol@company.com"},
    {"name": "Dan Wilson", "role": "Finance", "email": "dan@company.com"},
    {"name": "Eve Martin", "role": "Exec", "email": "eve@company.com"},
]

# Embedding dimension (matching common face recognition models)
EMBEDDING_DIM = 128


def generate_synthetic_embeddings():
    """Generate synthetic face embeddings for registered users.
    
    Each user gets a unique embedding vector with slight variations
    (simulating different lighting/angle captures).
    """
    print("  Generating synthetic face embeddings...")
    for user in REGISTERED_USERS:
        name = user["name"]
        user_dir = DATA_DIR / "faces" / name.replace(" ", "_")
        user_dir.mkdir(parents=True, exist_ok=True)

        # Base embedding for this user (unique identity)
        rng = np.random.RandomState(hashlib.md5(name.encode()).digest()[0])
        base_embedding = rng.randn(EMBEDDING_DIM).astype(np.float32)
        base_embedding = base_embedding / np.linalg.norm(base_embedding)

        # Save base embedding
        np.save(user_dir / "embedding.npy", base_embedding)

        # Generate sample variations (simulating different captures)
        for i in range(5):
            noise = np.random.randn(EMBEDDING_DIM).astype(np.float32) * 0.05
            variant = base_embedding + noise
            variant = variant / np.linalg.norm(variant)
            np.save(user_dir / f"sample_{i}.npy", variant)

        print(f"    {name:20s} — embedding saved")

    # Generate impostor embeddings (unauthorized users)
    imposter_dir = DATA_DIR / "faces" / "impostors"
    imposter_dir.mkdir(parents=True, exist_ok=True)
    for i in range(10):
        imp_emb = np.random.randn(EMBEDDING_DIM).astype(np.float32)
        imp_emb = imp_emb / np.linalg.norm(imp_emb)
        np.save(imposter_dir / f"impostor_{i}.npy", imp_emb)

    print(f"    {'Impostors':20s} — 10 synthetic faces generated")


def create_user_database():
    """Create user database JSON."""
    users = []
    for user in REGISTERED_USERS:
        name_clean = user["name"].replace(" ", "_")
        emb_path = DATA_DIR / "faces" / name_clean / "embedding.npy"
        if emb_path.exists():
            embedding = np.load(emb_path)
            users.append({
                "user_id": hashlib.md5(user["name"].encode()).hexdigest()[:12],
                "name": user["name"],
                "role": user["role"],
                "email": user["email"],
                "embedding_path": str(emb_path),
                "embedding_b64": base64.b64encode(embedding.tobytes()).decode(),
            })

    db_path = DATA_DIR / "users.json"
    with open(db_path, "w") as f:
        json.dump({"users": users, "total": len(users)}, f, indent=2)
    print(f"\n  User database saved to {db_path}")
    return users


def create_sample_images():
    """Create simple synthetic face images for demo/testing."""
    img_dir = DATA_DIR / "sample_images"
    img_dir.mkdir(parents=True, exist_ok=True)

    for user in REGISTERED_USERS:
        name_clean = user["name"].replace(" ", "_")
        arr = np.random.randint(100, 200, (100, 100, 3), dtype=np.uint8)
        import cv2
        cv2.imwrite(str(img_dir / f"{name_clean}.png"), arr)

    print(f"  Sample images saved to {img_dir}")


def main():
    DATA_DIR.mkdir(exist_ok=True); RESULTS_DIR.mkdir(exist_ok=True)

    print("=" * 50)
    print("  Facial Recognition — Data Setup")
    print("=" * 50)

    generate_synthetic_embeddings()
    users = create_user_database()
    create_sample_images()

    print(f"\n  Ready: {len(users)} registered users, 128-dim embeddings")
    print("  Run 'python recognize_face.py' to test recognition")


if __name__ == "__main__":
    main()
