# 👁️ Eye Disease Detection System

**Vision Care** is an AI-powered project designed to detect eye diseases like **Diabetic Retinopathy**, **Cataract**, and **Glaucoma** using deep learning and pupillometric analysis. The system aims for early detection to prevent irreversible vision loss and assist ophthalmologists with faster, more accurate diagnoses.

## 🚀 Features
- Automated eye disease detection using CNN
- Easy-to-use web interface (HTML, CSS, Bootstrap)
- Upload retinal or pupil images directly
- High model accuracy for reliable results
- Detailed model architecture and metrics display

## 📊 Model Performance
- **Training Accuracy**: 96%
- **Validation Accuracy**: 92%
- **Testing Accuracy**: 93%
- **Model Type**: Convolutional Neural Network (CNN)
- **Input Shape**: (224, 224, 3)

## 📂 Project Structure
```
├── index.html        # Main website interface
├── model_info.html   # Model summary and details
├── static/           # Images, CSS files
├── templates/        # HTML templates (if Flask used)
├── app.py            # Backend server (Flask, if applicable)
├── README.md         # Project documentation
```

## ⚙️ Technologies Used
- **Frontend**: HTML5, CSS3, Bootstrap 5
- **Backend**: Python (Flask)
- **AI/ML**: TensorFlow / Keras
- **Design Tools**: Figma

## 🛠️ How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/eye-disease-detection.git
   ```
2. Navigate to the project directory.
3. Run the server:
   ```bash
   python app.py
   ```
4. Open `http://localhost:5000` in your browser.
5. Upload eye images and start testing!

## 📌 Future Improvements
- Add real-time eye tracking
- Expand dataset for higher accuracy
- Deploy to cloud servers (AWS/GCP/Azure)
- Mobile version for accessibility

## ✨ Acknowledgements
- Thanks to **RCOEM** and **Biospectronics** for guidance.
- Inspired by the need for early, accessible eye disease detection.
