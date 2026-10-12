# Facial Emotion Recognition

**Group members:** Mateus BASTOS SOARES, Júlia Ellen DIAS LEITE, Helena GUACHALLA DE ANDRADE

**Course:** RO311 - Apprentissage pour la robotique @ ENSTA Paris

**Date:** October 2026

This project aims to detect and classify emotions from pictures. We use the [FER-2013](https://www.kaggle.com/datasets/msambare/fer2013) dataset and explore three different models: a KNN, a Linear SVM (both using preprocessing feature extraction methods LBP and HOG, respectively), and an implementation of the mini-Xception CNN, all trained in Google Colab.

## Live Demo

You can try the trained models with you own recordings [here](https://diasjulia.github.io/FaceEmotionRecognizer/).

Upload or take a picture of your facial expression directly in the browser. The app extracts features from the image, and runs the selected trained network to show the predicted emotion.

## Repository Structure

```
├── models/                            # trained models saved as .onnx files
├── notebooks/                         # notebooks with training and results for each network
├── README.md                        
├── config.json                        # live app files: models and labels
├── features.js                        # implementation of LBP and HOG feature extraction, as in scikit-image
├── index.html
├── script.js                          # image input and ONNX inference
└── style.css
```
