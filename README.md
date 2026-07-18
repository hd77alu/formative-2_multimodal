# Formative 2 — Multi-Modal Biometric Authentication & Product Recommendation

A demo system that authenticates a user with **two biometric modalities** (face + voice)
and, once access is granted, recommends a product category from the merged customer database.

## Project structure

```
├── app.py                          # The demo app (run this!)
├── requirements.txt
├── dataset/
│   ├── merged_customer_data.csv    # Merged transactions + social profiles
│   ├── image_features.csv          # HOG features extracted from raw images
│   └── audio_features.csv          # MFCC/spectral features from voice clips
├── models/
│   ├── facial_recognition_model.joblib   # XGBoost identity classifier (HOG features)
│   ├── voiceprint_model.joblib           # RandomForest speaker verifier (MFCC features)
│   ├── final_product_rec_model.joblib    # XGBoost product recommender
│   └── label_encoder.pkl
├── notebooks/                      # Training notebooks for all three models
└── raw images/                     # Face images for the 4 registered users
```

## Setup

```bash
pip install -r requirements.txt
```

## Run the demo

```bash
python app.py
```

The app will:

1. Ask **who you are** (registered users: `arsene`, `hamed`, `loic`, `principie` —
   bound to customer IDs 151, 160, 174, 187).
2. **Face verification** — enter an image path (or press Enter to use the registered
   image). The image is resized to 128×128, converted to grayscale, HOG features are
   extracted and classified by the XGBoost model.
3. **Voice verification** — enter a `.wav` path, or use demo mode
   `csv:<speaker>:<phrase>` (e.g. `csv:arsene:approve`) which loads precomputed
   features from `dataset/audio_features.csv`. Features are scaled and classified by
   the RandomForest voiceprint model (confidence threshold 0.65).
4. **Gatekeeping rule** — access is granted **only if BOTH face and voice match**
   the claimed identity. If either fails, access is denied.
5. **On success** — the app opens `merged_customer_data.csv`, pulls that customer's
   records, and runs the product recommendation model to suggest a product category.

## Demo scenarios to try

| Scenario | Input | Expected result |
|---|---|---|
| Legitimate user | `arsene` + Enter + Enter | ACCESS GRANTED + recommendation |
| Impostor voice | `arsene` + Enter + `csv:loic:approve` | ACCESS DENIED |
| Wrong face | `arsene` + `raw images/loic_neutral.jpeg` + Enter | ACCESS DENIED |
