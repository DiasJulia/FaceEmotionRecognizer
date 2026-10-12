# Facial Emotion Recognition

**Group members:** Mateus BASTOS SOARES, Júlia Ellen DIAS LEITE, Helena GUACHALLA DE ANDRADE

**Course:** RO311 - Apprentissage pour la robotique @ ENSTA Paris

**Date:** October 2026

This project aims to detect and classify emotions from pictures. We use the [FER-2013](https://www.kaggle.com/datasets/msambare/fer2013) dataset and explore three different models: a KNN, a Linear SVM (both using preprocessing feature extraction methods LBP and HOG, respectively), and an implementation of the [mini-Xception](https://arxiv.org/abs/1710.07557) CNN, all trained in Google Colab.

## Live Demo

You can try the trained models with you own recordings [here](https://diasjulia.github.io/FaceEmotionRecognizer/).

Upload or take a picture of your facial expression directly in the browser. The app extracts features from the image, and runs the three trained models side by side to compare their predicted emotions.

## Results

Using the full test set, the models obtained the following accuracies:

| Model | Test accuracy |
|---|---|
| LBP + KNN | 40.85% |
| HOG + Linear SVM | 43.42% |
| mini-Xception | 61.2% |

Afterwards, accuracy and average latency were also measured on 50 test samples. Since the first two models have preprocessing steps, the total time was split into feature extraction and inference, to highlight where each pipeline spends its time. 

| Model | Accuracy | Feature extraction [ms] | Inference [ms] | Total [ms] |
|---|---|---|---|---|
| LBP + KNN | 0.36 | 4.203 | 120.947 | 125.150 |
| HOG + Linear SVM | 0.50 | 0.934 | 0.045 | 0.979 |
| mini-Xception | 0.68 | 0.012 | 0.323 | 0.335 |

## Repository Structure

```
├── models/              # trained models saved as .onnx files
├── notebooks/           # notebooks with training and results for each network
├── README.md                        
├── config.json          # live app files: models and labels
├── features.js          # implementation of LBP and HOG feature extraction, as in scikit-image
├── index.html
├── script.js            # image input and ONNX inference
└── style.css
```
