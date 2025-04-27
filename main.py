import os
from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import cv2
import predict_grad  # Your existing prediction script
import google.generativeai as genai  # Import Gemini API
import tensorflow as tf
import io
from contextlib import redirect_stdout
import markdown  # Add this import for markdown conversion

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Replace with your actual Gemini API key
GEMINI_API_KEY = "AIzaSyAvoGshgmdac0pUaMQfVZJjXgZvPmL12gQ"
genai.configure(api_key=GEMINI_API_KEY)

def allowed_file(filename):
    """Check if the uploaded file has an allowed extension"""
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def generate_disease_report(disease_name, confidence):
    """Generates a report using Gemini API based on the disease."""

    model = genai.GenerativeModel('gemini-2.0-flash') #define model
    prompt = f"""
    Generate a detailed but concise report about {disease_name}, including:

    *   A description of the disease.
    *   Common symptoms.
    *   Possible causes and risk factors.
    *   Recommended precautions and lifestyle changes.
    *   Potential treatment options.
    *   Information on when to seek medical advice.
    *   Reliable resources for further information.

    The report should be written in a clear, easy-to-understand style suitable for a general audience. Focus on practical advice and actionable steps the user can take. Limit the report to approximately 400-500 words.  Mention that this report is AI generated and not a substitute for a consultation from a professional doctor.
    """

    try:
        response = model.generate_content(prompt)
        print(response)
        # Convert markdown to HTML
        return markdown.markdown(response.text)  # Convert markdown to HTML
    except Exception as e:
        print(f"Error generating report with Gemini: {e}")
        return "An error occurred while generating the disease report."  # Handle errors gracefully

def create_baselineCNN():
    """Create the baseline CNN model as defined in your code"""
    # Using Input layer first to avoid warning
    inputs = tf.keras.layers.Input(shape=(224, 224, 3))
    
    # First block
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu', padding='same')(inputs)
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPool2D(2)(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    # Second block
    x = tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPool2D(2)(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    # Third block
    x = tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPool2D(2)(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    # Dense layers
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    outputs = tf.keras.layers.Dense(4, activation='softmax')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), 
                  optimizer='adam', 
                  metrics=['accuracy'])
    
    return model

def get_model_info():
    """Get detailed information about the model"""
    model = create_baselineCNN()
    
    # Capture model summary
    summary_io = io.StringIO()
    with redirect_stdout(summary_io):
        model.summary()
    model_summary = summary_io.getvalue()
    
    # Get model information
    trainable_count = np.sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    non_trainable_count = np.sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])
    
    # Get layer information
    layers_info = []
    for layer in model.layers:
        config = {}
        if isinstance(layer, tf.keras.layers.Conv2D):
            config['filters'] = layer.filters
            config['kernel_size'] = layer.kernel_size
            config['activation'] = layer.activation.__name__ if callable(layer.activation) else str(layer.activation)
        elif isinstance(layer, tf.keras.layers.Dense):
            config['units'] = layer.units
            config['activation'] = layer.activation.__name__ if callable(layer.activation) else str(layer.activation)
        elif isinstance(layer, tf.keras.layers.Dropout):
            config['rate'] = layer.rate
        
        # Get output shape safely
        output_shape = "Unknown"
        try:
            if hasattr(layer, 'output_shape'):
                output_shape = str(layer.output_shape)
            elif hasattr(layer, 'output'):
                output_shape = str(layer.output.shape)
        except:
            pass
            
        layers_info.append({
            'name': layer.name,
            'type': layer.__class__.__name__,
            'output_shape': output_shape,
            'params': layer.count_params(),
            'config': config
        })
    
    # Model info dictionary
    model_info = {
        'summary': model_summary,
        'total_params': f"{trainable_count + non_trainable_count:,}",
        'trainable_params': f"{trainable_count:,}",
        'non_trainable_params': f"{non_trainable_count:,}",
        'layers': layers_info,
        'optimizer': model.optimizer.__class__.__name__,
        'loss_function': 'SparseCategoricalCrossentropy',
        'accuracy': '96.2%',  # Replace with actual metrics
        'val_accuracy': '94.5%',  # Replace with actual metrics
        'classes': ['Normal', 'Covid', 'Viral Pneumonia', 'Bacterial Pneumonia']  # Replace with your actual classes
    }
    
    return model_info

@app.route('/')
def index():
    """Render the main index page"""
    return render_template('index.html')

@app.route('/model-info')
def model_info():
    """Render the model information page"""
    try:
        info = get_model_info()
        return render_template('model_info.html', model_info=info)
    except Exception as e:
        print(f"Error getting model information: {e}")
        return f"An error occurred while retrieving model information: {str(e)}", 500

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload"""
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)

    if file and allowed_file(file.filename):
        # Save the uploaded file
        filename = os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded_image.jpg')
        file.save(filename)

        # Redirect to classification page with the uploaded image
        return redirect(url_for('classify', filename='uploaded_image.jpg'))

    return redirect(request.url)

@app.route('/classify')
def classify():
    """Classify the uploaded image and render results"""
    filename = request.args.get('filename', '')

    if not filename:
        return redirect(url_for('index'))

    # Path to your trained model
    MODEL_PATH = 'base_cnn_model.h5'

    try:
        # Use your prediction function
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        results = predict_grad.predict_disease(image_path, MODEL_PATH)

        # Save Grad-CAM heatmap
        if 'grad_cam' in results:
            cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], 'grad_cam_heatmap.jpg'), results['grad_cam'])

        # Generate disease report using Gemini
        disease_name = results['predicted_disease'].replace('_', ' ').title()
        confidence = results['confidence']
        disease_report = generate_disease_report(disease_name, confidence)

        return render_template('classification.html',
                            original_image=f'uploads/{filename}',
                            grad_cam_image='uploads/grad_cam_heatmap.jpg',
                            results=results,
                            disease_report=disease_report) # Pass report to template

    except Exception as e:
        print(f"Error during classification: {e}")
        return f"An error occurred during image classification: {str(e)}", 500

if __name__ == '__main__':
    app.run(debug=True, port=3000)