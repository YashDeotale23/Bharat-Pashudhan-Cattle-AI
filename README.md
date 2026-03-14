# 🐄 Bharat Pashudhan: Cattle Breed Identification 
**Achieved 90% Accuracy in Local Breed Classification**

## 📖 Project Overview
This project is an AI-powered tool designed to support the **National Digital Livestock Mission**. It accurately identifies over 40 indigenous Indian cattle breeds from images. By leveraging Deep Learning, it helps farmers and veterinarians maintain digital records and verify breeds instantly.

## 🚀 Key Features
- **High Accuracy:** Fine-tuned Xception model achieving **90% test accuracy**.
- **Mobile Access:** Runs a local Gradio server accessible via any smartphone on the same Wi-Fi.
- **Transfer Learning:** Built on the pre-trained ImageNet weights for robust feature detection.
- **Local Privacy:** No data is sent to external clouds; all processing happens on your local machine.

## 🛠️ Tech Stack
- **Languages:** Python 3.x
- **Deep Learning:** TensorFlow, Keras (Xception architecture)
- **Interface:** Gradio
- **Data Handling:** NumPy, Pillow, Matplotlib

## ⚠️ Known Limitations & Challenges
While the model achieves 90% accuracy, there are specific "Fine-Grained" challenges where the AI may struggle:
* **Color Similarity:** Many indigenous Indian breeds (e.g., Tharparkar, Ongole, Hariana) primarily have white or light-grey skin. The AI sometimes confuses these breeds if the lighting is flat.
* **Morphological Overlap:** Breeds with similar hump sizes or ear shapes can lead to lower confidence scores.
* **Lighting Conditions:** Overexposed (too bright) photos of white cattle can "wash out" the texture details the AI needs to differentiate breeds.
* **Crossbreeding:** The model is trained on purebred characteristics; crossbred animals may produce unexpected results.

## 📦 Installation & Setup
1. **Download the Model:**
   [Click Here to Download the Trained .keras Model](https://github.com/YashDeotale23/Bharat-Pashudhan-Cattle-AI/releases/download/v1.0/bharat_pashudhan_final.keras)
   *(Important: Place this file inside your project folder after downloading)*.

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
