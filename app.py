import os
import warnings

warnings.filterwarnings("ignore")
os.environ["OPENCV_LOG_LEVEL"] = "OFF"

import cv2
import joblib
import numpy as np
import pandas as pd
from skimage.feature import hog
from xgboost import XGBClassifier


# Registered users: name -> customer ID + face image
USERS = {
    "arsene":    {"customer_id": 151, "image": "raw images/arsene_neutral.jpg"},
    "hamed":     {"customer_id": 160, "image": "raw images/hamed_neutral.jpg"},
    "loic":      {"customer_id": 174, "image": "raw images/loic_neutral.jpeg"},
    "principie": {"customer_id": 187, "image": "raw images/principie_neutral.jpg"},
}

FACE_MODEL_PATH = "models/facial_recognition_model.json"
VOICE_MODEL_PATH = "models/voiceprint_model.joblib"
REC_MODEL_PATH = "models/final_product_rec_model.json"
MERGED_DATA_PATH = "dataset/merged_customer_data.csv"
AUDIO_FEATURES_PATH = "dataset/audio_features.csv"

FACE_THRESHOLD = 0.60


def extract_face_features(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    img = cv2.resize(img, (128, 128))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return hog(gray, pixels_per_cell=(16, 16),
               cells_per_block=(2, 2), feature_vector=True)


def verify_face(image_path, claimed_name, face_bundle):
    features = extract_face_features(image_path)
    if features is None:
        print("  [!] Could not read the image file.")
        return False

    probs = face_bundle["model"].predict_proba([features])[0]
    predicted_name = face_bundle["label_encoder"].classes_[int(probs.argmax())]
    confidence = float(probs.max())

    print(f"  Face model says: {predicted_name} (confidence {confidence:.2f})")
    return (predicted_name.lower() == claimed_name.lower()) and (confidence >= FACE_THRESHOLD)


def extract_voice_features(audio_path, sr=22050, n_mfcc=13):
    import librosa

    signal, sr = librosa.load(audio_path, sr=sr)
    signal, _ = librosa.effects.trim(signal, top_db=25)
    if signal.size == 0:
        signal = np.zeros(sr, dtype="float32")

    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc)
    rolloff = librosa.feature.spectral_rolloff(y=signal, sr=sr)
    centroid = librosa.feature.spectral_centroid(y=signal, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(signal)
    rms = librosa.feature.rms(y=signal)

    row = {}
    for j in range(n_mfcc):
        row[f"mfcc_{j+1}_mean"] = float(mfcc[j].mean())
        row[f"mfcc_{j+1}_std"] = float(mfcc[j].std())
    row["rolloff_mean"] = float(rolloff.mean())
    row["centroid_mean"] = float(centroid.mean())
    row["zcr_mean"] = float(zcr.mean())
    row["rms_mean"] = float(rms.mean())
    return row


def load_features_from_csv(speaker, phrase="approve"):
    df = pd.read_csv(AUDIO_FEATURES_PATH)
    rows = df[(df["speaker"] == speaker) & (df["phrase"] == phrase)
              & (df["variant"] == "original")]
    if rows.empty:
        return None
    return rows.iloc[0].to_dict()


def verify_voice(audio_input, claimed_name, voice_bundle):
    if audio_input.startswith("csv:"):
        parts = audio_input.split(":")
        phrase = parts[2] if len(parts) > 2 else "approve"
        row = load_features_from_csv(parts[1], phrase)
        if row is None:
            print("  [!] No such speaker/phrase in audio_features.csv")
            return False
    elif os.path.exists(audio_input):
        row = extract_voice_features(audio_input)
    else:
        print("  [!] Audio file not found.")
        return False

    feats = np.array([[row[c] for c in voice_bundle["feature_cols"]]])
    feats = voice_bundle["scaler"].transform(feats)
    probs = voice_bundle["model"].predict_proba(feats)[0]
    predicted_name = voice_bundle["label_encoder"].classes_[int(probs.argmax())]
    confidence = float(probs.max())

    print(f"  Voice model says: {predicted_name} (confidence {confidence:.2f})")
    return (predicted_name.lower() == claimed_name.lower()) and (confidence >= voice_bundle["threshold"])


def recommend_product(customer_id, rec_bundle):
    df = pd.read_csv(MERGED_DATA_PATH)
    customer = df[df["customer_id_legacy"] == customer_id]
    if customer.empty:
        print(f"  [!] Customer {customer_id} not found in the merged database.")
        return

    X = customer[["engagement_score", "purchase_interest_score", "customer_rating",
                  "social_media_platform", "review_sentiment"]].copy()
    X = pd.get_dummies(X, columns=["social_media_platform", "review_sentiment"])
    X = X.reindex(columns=rec_bundle["features"], fill_value=0).astype(float)

    probs = rec_bundle["model"].predict_proba(X)
    class_names = rec_bundle["label_encoder"].classes_
    adjusted = probs / np.array([rec_bundle["thresholds"][c] for c in class_names])
    predictions = class_names[np.argmax(adjusted, axis=1)]

    top_category = pd.Series(predictions).mode()[0]

    print(f"\n  Customer profile ({len(customer)} records found):")
    print(f"    Past purchases: {', '.join(customer['product_category'].unique())}")
    print(f"    Average rating given: {customer['customer_rating'].mean():.1f}")
    print(f"\n  >>> RECOMMENDED PRODUCT: {top_category} <<<")

# MAIN APP EXECUTION BLOCK
def main():
    print("=" * 60)
    print("   MULTI-MODAL BIOMETRIC AUTHENTICATION SYSTEM")
    print("=" * 60)

    # Load Face model
    face_model = XGBClassifier()
    face_model.load_model(FACE_MODEL_PATH)
    # Load the extras (encoder, threshold)
    face_bundle = joblib.load("models/facial_recognition_bundle.joblib")
    # Attach the model into the bundle for consistency
    face_bundle["model"] = face_model

    # Load Voice model
    voice_bundle = joblib.load(VOICE_MODEL_PATH)

    # Load recommender model from JSON
    rec_model = XGBClassifier()
    rec_model.load_model(REC_MODEL_PATH)
    # Load extras
    rec_bundle = joblib.load("models/final_product_rec_bundle.joblib")
    rec_bundle["model"] = rec_model

    print(f"\nRegistered users: {', '.join(USERS.keys())}")
    claimed_name = input("\nWho are you? Enter your name: ").strip().lower()

    if claimed_name not in USERS:
        print("\n[X] ACCESS DENIED - unknown user.")
        return

    print("\n[STEP 1/2] FACE VERIFICATION")
    default_img = USERS[claimed_name]["image"]
    image_path = input(f"   Image path (Enter = {default_img}): ").strip() or default_img
    face_ok = verify_face(image_path, claimed_name, face_bundle)
    print(f"   Face check: {'PASSED' if face_ok else 'FAILED'}")

    # EARLY EXIT LOGIC: Terminate immediately if face verification fails
    if not face_ok:
        print("\n" + "=" * 60)
        print("[X] ACCESS DENIED - face verification failed.")
        print("=" * 60)
        return

    print("\n[STEP 2/2] VOICE VERIFICATION")
    print("   Enter a .wav path, or 'csv:<speaker>:<phrase>' to use a")
    print("   saved sample from audio_features.csv (e.g. csv:arsene:approve)")
    default_audio = f"csv:{claimed_name}:approve"
    audio_input = input(f"   Audio (Enter = {default_audio}): ").strip() or default_audio
    voice_ok = verify_voice(audio_input, claimed_name, voice_bundle)
    print(f"   Voice check: {'PASSED' if voice_ok else 'FAILED'}")

    print("\n" + "=" * 60)
    if voice_ok:  # No need to check face_ok here since it's pre-validated
        customer_id = USERS[claimed_name]["customer_id"]
        print(f"[OK] ACCESS GRANTED - welcome {claimed_name.capitalize()} (customer {customer_id})")
        print("=" * 60)
        print("\nGenerating your product recommendation...")
        recommend_product(customer_id, rec_bundle)
    else:
        print("[X] ACCESS DENIED - voice verification failed.")
        print("=" * 60)


if __name__ == "__main__":
    main()