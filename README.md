# 🌿 Plant Leaf Disease Detection using CBAM-VGG16

**Paper:** Interpretable Plant Leaf Disease Detection Using Attention-Enhanced CNN (arXiv:2512.17864v2)

## Results
| Dataset | Accuracy |
|---------|----------|
| PlantVillage | 98.72% |
| Rice | 98.87% |
| Apple | 95.42% |
| Maize | 95.00% |
| Embrapa | 94.20% |

## Tech Stack
Python, TensorFlow, Keras, OpenCV, Grad-CAM, Explainable AI (XAI)

## Architecture
- VGG16 backbone + CBAM (Channel + Spatial Attention) after each of 5 conv blocks
- CLAHE preprocessing + L2 regularization
- Grad-CAM heatmaps for interpretability

## Links
- 📊 [Kaggle Notebook](https://www.kaggle.com/code/vaibhavpandey74/cnn-project)
