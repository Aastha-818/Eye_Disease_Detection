# 👁️ Eye Disease Detection System

**Vision Care** is an AI-based project designed to detect eye diseases like **Diabetic Retinopathy**, **Cataract**, and **Glaucoma** using deep learning and pupillometric analysis. This system enables early detection to help prevent irreversible vision loss and assists ophthalmologists with faster, more accurate diagnoses.

## 🚀 Features
- Automated eye disease detection using CNN
- Web interface for image upload and testing
- Detailed model performance display
- High accuracy in disease classification
- Organized and easy-to-use project structure

## 📊 Model Performance
- **Training Accuracy**: 96%
- **Validation Accuracy**: 92%
- **Testing Accuracy**: 93%
- **Model Type**: Convolutional Neural Network (CNN)
- **Input Shape**: (224, 224, 3)

## 📂 Project Structure
```
├── static/                  # UI Images
├── README.md                 # Project Documentation
├── classification.html       # Classification result page
├── index.html                # Main homepage
├── main model file.ipynb      # Jupyter Notebook (Model Training)
├── main.py                   # Flask App (Main Server)
├── model_info.html           # Model Metrics and Summary
├── script.js                 # JavaScript File
├── styles.css                # Styling (CSS)
```

## ⚙️ Technologies Used
- **Frontend**: HTML5, CSS3, Bootstrap 5, JavaScript
- **Backend**: Python (Flask Framework)
- **Deep Learning**: TensorFlow / Keras
- **Design Tool**: Figma

## 🛠️ How to Run the Project
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/eye-disease-detection.git
   ```
2. Navigate to the project folder:
   ```bash
   cd eye-disease-detection
   ```
3. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```
   *(If no requirements.txt yet, you mainly need Flask, TensorFlow/Keras.)*

4. Run the Flask server:
   ```bash
   python main.py
   ```
5. Open your browser and go to:
   ```
   http://127.0.0.1:5000/
   ```
6. Start uploading eye images and detect diseases!

