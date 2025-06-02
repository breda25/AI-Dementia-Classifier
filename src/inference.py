import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

class DementiaClassifier:
    def __init__(self, model_path, image_size=(128, 128)):
        """
        Initialize the classifier
        
        Args:
            model_path: Path to the saved model
            image_size: Target image size for inference
        """
        self.model = load_model(model_path)
        self.image_size = image_size
        self.class_names = ['Non Demented', 'Mild Dementia', 
                           'Moderate Dementia', 'Very Mild Dementia']
                           
    def preprocess_image(self, img_path):
        """
        Load and preprocess an image for prediction
        
        Args:
            img_path: Path to the image file
            
        Returns:
            Preprocessed image as numpy array
        """
        try:
            img = Image.open(img_path)
            img = img.resize(self.image_size)
            img_array = np.array(img)
            
            # Check if image has correct dimensions
            if img_array.shape != (*self.image_size, 3):
                raise ValueError(f"Image should have shape {self.image_size + (3,)}, "
                               f"but got {img_array.shape}")
                               
            # Normalize and expand dimensions
            img_array = img_array / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            return img_array
            
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            return None
            
    def predict(self, img_path, display_result=True):
        """
        Predict dementia class for an image
        
        Args:
            img_path: Path to the image file
            display_result: Whether to display the image with prediction
            
        Returns:
            Dictionary with prediction results
        """
        # Preprocess the image
        img_array = self.preprocess_image(img_path)
        if img_array is None:
            return {
                'success': False,
                'error': 'Failed to process image'
            }
            
        # Make prediction
        prediction = self.model.predict(img_array, verbose=0)
        class_index = np.argmax(prediction[0])
        confidence = prediction[0][class_index] * 100
        
        result = {
            'success': True,
            'class_index': int(class_index),
            'class_name': self.class_names[class_index],
            'confidence': float(confidence),
            'probabilities': {
                self.class_names[i]: float(prediction[0][i] * 100)
                for i in range(len(self.class_names))
            }
        }
        
        # Display result if requested
        if display_result:
            self.display_prediction(img_path, result)
            
        return result
        
    def display_prediction(self, img_path, result):
        """Display the image with prediction results"""
        # Load the original image
        img = Image.open(img_path)
        
        plt.figure(figsize=(10, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.title("Input MRI Scan")
        plt.axis('off')
        
        # Create a bar chart for probabilities
        plt.subplot(1, 2, 2)
        classes = list(result['probabilities'].keys())
        probs = list(result['probabilities'].values())
        colors = ['green' if i == result['class_index'] else 'blue' 
                 for i in range(len(classes))]
        
        y_pos = np.arange(len(classes))
        plt.barh(y_pos, probs, color=colors)
        plt.yticks(y_pos, classes)
        plt.xlabel('Probability (%)')
        plt.title('Prediction Probabilities')
        
        plt.suptitle(f"Prediction: {result['class_name']} ({result['confidence']:.2f}% confidence)")
        plt.tight_layout()
        plt.show()

    def batch_predict(self, image_directory, extensions=('.jpg', '.jpeg', '.png')):
        """
        Run predictions on all images in a directory
        
        Args:
            image_directory: Directory containing images
            extensions: Tuple of valid file extensions
            
        Returns:
            Dictionary of results keyed by filename
        """
        results = {}
        
        for filename in os.listdir(image_directory):
            if filename.lower().endswith(extensions):
                img_path = os.path.join(image_directory, filename)
                print(f"Processing {filename}...")
                result = self.predict(img_path, display_result=False)
                results[filename] = result
                
        return results