import os
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions

def predict_disease(image_path, model_path, apply_grad_cam=True):
    """
    Predict disease from an image and optionally apply Grad-CAM visualization
    
    Args:
        image_path (str): Path to the input image
        model_path (str): Path to the saved disease classification model
        apply_grad_cam (bool): Whether to generate Grad-CAM visualization
    
    Returns:
        dict: Prediction results and optional Grad-CAM visualization
    """
    # Validate file paths
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path not found: {model_path}")
    
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image path not found: {image_path}")
    
    # Disease categories
    class_names = ['cataract', 'diabetic_retinopathy', 'glaucoma', 'normal']
    
    # Load the saved model
    try:
        disease_model = load_model(model_path)
    except Exception as e:
        print(f"Error loading disease model: {e}")
        raise
    
    # Load and preprocess the image
    try:
        # Use os.path.normpath to handle different path formats
        normalized_path = os.path.normpath(image_path)
        img = load_img(normalized_path, target_size=(224, 224))
        
        # Display the input image
        # plt.figure(figsize=(10, 10))
        # plt.imshow(img)
        # plt.title('Input Image')
        # plt.axis('off')
        # plt.show()
        
        # Convert image to array
        img_array = img_to_array(img)
        img_array_normalized = np.expand_dims(img_array, axis=0)
        img_array_normalized /= 255.0  # Normalize pixel values to [0,1]
        
        # Make prediction
        predictions = disease_model.predict(img_array_normalized)
        predicted_class_index = np.argmax(predictions[0])
        predicted_disease = class_names[predicted_class_index]
        confidence = predictions[0][predicted_class_index]
        
        # Prepare detailed results
        results = {
            'predicted_disease': predicted_disease,
            'confidence': float(confidence),
            'all_probabilities': dict(zip(class_names, predictions[0]))
        }
        
        # Apply Grad-CAM if requested
        if apply_grad_cam:
            results['grad_cam'] = apply_grad_cam_visualization(image_path)
        
        return results
    
    except Exception as e:
        print(f"Error processing image: {e}")
        raise

def apply_grad_cam_visualization(img_path, layer_name="conv5_block3_out"):
    """
    Apply Grad-CAM visualization to highlight important regions in the image
    
    Args:
        img_path (str): Path to the input image
        layer_name (str): Layer to use for Grad-CAM
    
    Returns:
        tuple: Original image and Grad-CAM overlay
    """
    # Load pre-trained ResNet50 model
    base_model = ResNet50(weights="imagenet")
    model = Model(inputs=base_model.input, outputs=[base_model.get_layer(layer_name).output, base_model.output])
    
    # Preprocess image
    img_array = load_img(img_path, target_size=(224, 224))
    img_array = img_to_array(img_array)
    img_array_processed = np.expand_dims(img_array, axis=0)
    img_array_processed = preprocess_input(img_array_processed)
    
    # Get predictions
    conv_output, predictions = model.predict(img_array_processed)
    class_idx = np.argmax(predictions[0])
    
    # Compute gradients
    with tf.GradientTape() as tape:
        conv_output, predictions = model(img_array_processed)
        loss = predictions[:, class_idx]
    
    grads = tape.gradient(loss, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Weight feature maps
    conv_output = conv_output.numpy()[0]
    for i in range(conv_output.shape[-1]):
        conv_output[:, :, i] *= pooled_grads[i]
    
    # Create heatmap
    heatmap = np.mean(conv_output, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    
    # Load and resize original image
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    
    # Resize and color heatmap
    heatmap_colored = cv2.resize(heatmap, (224, 224))
    heatmap_colored = np.uint8(255 * heatmap_colored)
    heatmap_colored = cv2.applyColorMap(heatmap_colored, cv2.COLORMAP_JET)
    
    # Overlay heatmap
    superimposed_img = cv2.addWeighted(heatmap_colored, 0.6, img, 1 - 0.6, 0)
    
    return superimposed_img

def display_prediction_results(results):
    """
    Display prediction results in a formatted manner
    
    Args:
        results (dict): Prediction results dictionary
    """
    print("\nPrediction Results:")
    print(f"Detected Disease: {results['predicted_disease'].replace('_', ' ').title()}")
    print(f"Confidence: {results['confidence']*100:.2f}%")
    
    print("\nDetailed Probabilities:")
    for disease, prob in results['all_probabilities'].items():
        print(f"{disease.replace('_', ' ').title()}: {prob*100:.2f}%")
    
    # Display Grad-CAM if available
    if 'grad_cam' in results:
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB))
        plt.title("Original Image")
        plt.axis("off")
        
        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(results['grad_cam'], cv2.COLOR_BGR2RGB))
        plt.title("Grad-CAM Heatmap")
        plt.axis("off")
        
        plt.show()

# Example usage
if __name__ == "__main__":
    # Use raw string or forward slashes to avoid escape character issues
    image_path = r'archive (1)\dataset\diabetic_retinopathy\100_left.jpeg'
    model_path = r'base_cnn_model.h5'
    
    try:
        results = predict_disease(image_path, model_path)
        display_prediction_results(results)
    except Exception as e:
        print(f"An error occurred: {e}")