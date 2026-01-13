# 🔬 Cancer Cell Classification

This repository contains the implementation of an **Explainable AI (XAI)** system for histopathology image analysis. Using the **Vision Transformer (ViT)** architecture, the model identifies malignancy in tissue patches and provides visual "Heatmaps" to explain its diagnostic reasoning.



## 🚀 Deployment & Demo
- **Live Interactive Demo:** https://vanu282003-cancer-ai-app.hf.space

## 🧠 Project Overview
In medical diagnostics, "black-box" AI models are difficult to integrate into clinical workflows. This project addresses the "Interpretability Gap" by utilizing **Self-Attention mechanisms** to highlight exactly which cellular structures influenced the AI's decision.

### Key Features:
- **Architecture:** Transformer-based Image Classification (ViT-Base).
- **Explainability:** Attention Rollout implementation for real-time heatmap generation.
- **Accuracy:** ~94.5% validation accuracy on histopathology datasets.
- **UI:** Professional Aqua-Black laboratory-grade interface built with Gradio.

## 🛠️ Tech Stack
- **Deep Learning:** PyTorch, Hugging Face Transformers
- **Computer Vision:** OpenCV, PIL
- **Interface:** Gradio (deployed on Hugging Face Spaces)
- **Data Handling:** NumPy

## 📊 How to use the Notebook
1. Open the `Medical_AI.ipynb` file in **Google Colab**.
2. Ensure you have a GPU runtime enabled for faster inference (`Runtime > Change runtime type > T4 GPU`).
3. Run all cells to install dependencies and initialize the model.
4. Use the Gradio link generated at the bottom of the notebook to test the model with your own tissue patch images.

## 🌡️ Understanding the Heatmap
- 🔴 **Red Hotspots:** High-risk cellular nuclei or anomalies identified by the model.
- 🔵 **Blue Zones:** Background tissue or normal structures ignored by the model.

---
*Developed by Vanashree Hegde*
