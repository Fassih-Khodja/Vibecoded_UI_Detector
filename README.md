# Vibecoded Detector 🎨

![Vibecoded Detector](https://img.shields.io/badge/Status-Active-brightgreen)
![Python](https://img.shields.io/badge/Python-3.x-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep_Learning-EE4C2C)

**Vibecoded Detector** is a Machine Learning project that determines whether a website or app screenshot is "vibecoded" (aesthetically pleasing, modern, and stylized). This repository contains the entire pipeline—from data collection and preprocessing to deep learning model training (using ResNet18) and inference.



## Project Overview

The core objective of this project is to create an automated "aesthetic scorer" for digital interfaces. It includes:
1. **Data Collection & Building**: Automated scripts to gather and label interface screenshots.
2. **Preprocessing**: Normalizing, cropping, and transforming images into a standardized square format suitable for neural networks.
3. **Deep Learning Model**: Fine-tuning a pre-trained **ResNet18** model in PyTorch to classify the "vibe" of an interface.
4. **Web UI**: A simple web application interface to upload screenshots and receive instant vibe scores.

##  Repository Structure

The project has been organized into a clean, modular structure:

```text
├── data/                  # Datasets, metadata, and large raw files (Ignored in Git)
├── models/                # Saved PyTorch models (e.g., .pt files)
├── notebooks/             # Jupyter Notebooks for experimentation
│   ├── 01_data_collection.ipynb
│   ├── 02_data_preprocessing.ipynb
│   └── 03_model_training.ipynb
├── src/                   # Python source code for data processing
│   └── data_preprocessing.py
├── web_app/               # Web application files
│   └── templates/
│       └── index.html
├── .gitignore
└── README.md              # Project documentation
```

##  How It Works

### 1. Data Pipeline (`notebooks/01_data_collection.ipynb` & `02_data_preprocessing.ipynb`)
- **Collection**: Screenshots are gathered and labeled.
- **Preprocessing**: The images are padded to a 1:1 aspect ratio, resized, and normalized using standard ImageNet statistics to maintain consistency for the ResNet18 model.

### 2. Model Training (`notebooks/03_model_training.ipynb`)
- Built using **PyTorch**.
- Leverages transfer learning via **ResNet18**.
- The model is fine-tuned to classify images, optimizing for the best **F1-score** to handle potential class imbalances.
- The best performing weights and the classification threshold are saved as a model package.

### 3. Web Inference (`web_app/`)
- A localized web interface that allows users to drag-and-drop screenshots.
- The images are passed through the same preprocessing pipeline and fed to the model.
- An intuitive "vibe meter" displays the probability that the interface is "vibecoded".

##  Installation & Setup

To run this project locally, follow these steps:

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/vibecoded-detector.git
   cd vibecoded-detector
   ```

2. **Set up a Virtual Environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Web App**
   ```bash
   cd web_app
   python app.py
   ```
   Then open `http://127.0.0.1:5000` in your browser.

5. **Run the Notebooks (Optional)**
   Navigate to the `notebooks/` directory and run the notebooks in sequence to recreate the dataset and retrain the model if desired.



