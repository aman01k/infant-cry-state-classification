🍼 Infant Cry State Classification
Hybrid Deep Learning System for Infant Cry Analysis
<p align="center"> <img src="output images/result images/architecture_diagram.png" width="700"/> </p> <p align="center"> <b>CNN + Acoustic Feature Fusion · Audio AI · Healthcare Assistive Tech</b> </p>
🚀 Overview

Infants communicate only through crying — but interpreting the reason is difficult even for trained caregivers.

This project builds a hybrid deep learning model that classifies infant cries into meaningful physiological states using audio signal processing and deep learning.

🎯 Goal: Assist caregivers (especially NICU environments) with automated cry interpretation.

🧠 Model Highlights
🔊 Processes raw 16 kHz WAV audio
🎼 Uses Log-Mel Spectrograms + Acoustic Features
🧩 Hybrid architecture (CNN + Dense fusion)
📊 Predicts 8 cry states
🧬 Architecture
<p align="center"> <img src="output images/architecture_diagram.png" width="800"/> </p>
Dual-Branch Design
Branch	Input	Purpose
CNN Branch	Spectrograms	Learn spatial/audio patterns
Feature Branch	70-dim vector	Capture handcrafted acoustic traits

Fusion → Dense Layers → Softmax Output

📊 Results
Metric	Score
Accuracy	42%
Macro F1	0.47
🟢 Strong Performance
Laugh → F1: 1.00
Silence → F1: 1.00
🔴 Challenging Classes
Tired
Discomfort
🎧 Dataset
📦 Source: Kaggle Baby Crying Sounds Dataset
📁 ~1,197 audio samples
⏱ ~7 sec per clip
Classes
Belly Pain
Burping
Discomfort
Hungry
Laugh
Silence
Tired
Cold/Hot (merged into Hungry)
⚙️ Feature Engineering
Spectral (CNN Input)
Log-Mel Spectrograms
Acoustic (70 Features)
MFCC
Chroma
Zero Crossing Rate
RMS Energy
Spectral Contrast
Pitch
📁 Project Structure
infant-cry-state-classification/
│
├── notebook/
│   ├── Final_Phase_HybridFusion.ipynb
│   └── Phase2_DL.ipynb
│
├── output images/
│   ├── architecture_diagram.png
│   ├── confusion_matrix.png
│   └── spectrograms_comparison.png
│
├── report/
│
└── README.md
⚙️ Installation
pip install tensorflow librosa scikit-learn imbalanced-learn matplotlib seaborn
▶️ Usage
Open:
notebook/Final_Phase_HybridFusion.ipynb
Update dataset path
Run all cells
🔍 Key Insights
CNN captures frequency patterns well
Acoustic features improve class separability
Model struggles with temporal patterns (rhythm)
🚧 Limitations
Small dataset
No temporal modeling
Class imbalance
🔮 Future Work
🔁 Add LSTM / temporal modeling
📈 Data augmentation
🧠 Hierarchical classification
🏥 NICU real-world deployment
📸 Outputs
<p align="center"> <img src="output images/confusion_matrix.png" width="500"/> <img src="output images/spectrograms_comparison.png" width="500"/> </p>
👨‍💻 Author

Aman

⭐ Notes
Large dataset & .h5 models are excluded
Use .gitignore to prevent large file uploads
💡 Tech Stack
Python
TensorFlow / Keras
Librosa
Scikit-learn
Matplotlib / Seaborn
📌 Summary

This project demonstrates how hybrid deep learning + audio signal processing can be applied to a real-world healthcare problem.

🧠 From sound → signal → features → intelligence
