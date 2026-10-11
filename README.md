# Facial Emotion Recognition

**Group members:** Mateus BASTOS SOARES, Júlia Ellen DIAS LEITE, Helena GUACHALLA DE ANDRADE

**Course:** RO311 - Apprentissage pour la robotique @ ENSTA Paris

**Date:** September 2026

This project aims to classify emotions from images.

## Repository Structure

```
├── models/                            # trained models saved as .onnx files
├── notebooks/                         # notebooks with training and results for each network
├── config.json                        # live app files: models and labels
├── features.js                        # LBP (KNN) and HOG (SVM) extraction, as in scikit-image
├── index.html
├── script.js                          # image input and ONNX inference
└── README.md
```

## Running locally

The app is static (hosted on GitHub Pages). To run it locally, serve the folder:

```
python -m http.server
```

and open http://localhost:8000. Upload a photo or use the camera, choose a model and click *Predict emotion*.
