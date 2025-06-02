import os
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm
import cv2
import random
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class DementiaDataProcessor:
    def __init__(self, data_dir, image_size=(128, 128), batch_size=32):
        """
        Initialize data processor
        
        Args:
            data_dir: Directory containing the dataset
            image_size: Target size for images (width, height)
            batch_size: Batch size for training
        """
        self.data_dir = data_dir
        self.image_size = image_size
        self.batch_size = batch_size
        self.class_map = {
            'Non_Demented': 0, 
            'Mild_Dementia': 1, 
            'Moderate_Dementia': 2, 
            'Very_Mild_Dementia': 3
        }
        self.encoder = OneHotEncoder(sparse_output=False)
        self.encoder.fit(np.array([[0], [1], [2], [3]]))
        
    def scan_dataset(self):
        """Scan dataset directory and count files by category"""
        counts = {}
        total = 0
        
        for class_name in self.class_map.keys():
            class_dir = os.path.join(self.data_dir, class_name)
            if not os.path.exists(class_dir):
                print(f"Warning: {class_dir} does not exist")
                counts[class_name] = 0
                continue
                
            files = [f for f in os.listdir(class_dir) 
                    if f.endswith(('.jpg', '.jpeg', '.png'))]
            counts[class_name] = len(files)
            total += len(files)
            
        print(f"Found {total} images across {len(self.class_map)} classes")
        for cls, count in counts.items():
            print(f"  - {cls}: {count} images ({count/total*100:.1f}%)")
            
        return counts
    
    def create_data_generators(self):
        """Create train, validation and test data generators"""
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            validation_split=0.2  # 20% for validation
        )
        
        # Only rescaling for validation/testing
        test_datagen = ImageDataGenerator(rescale=1./255)
        
        # Training generator
        train_generator = train_datagen.flow_from_directory(
            self.data_dir,
            target_size=self.image_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='training'
        )
        
        # Validation generator
        validation_generator = train_datagen.flow_from_directory(
            self.data_dir,
            target_size=self.image_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='validation'
        )
        
        # For a separate test set if you have it in a different directory
        # test_generator = test_datagen.flow_from_directory(
        #    test_dir,
        #    target_size=self.image_size,
        #    batch_size=self.batch_size,
        #    class_mode='categorical',
        # )
        
        return train_generator, validation_generator
    
    def load_dataset_memory(self, max_per_class=None, verbose=True):
        """
        Load entire dataset into memory (use with caution for large datasets)
        
        Args:
            max_per_class: Maximum number of images to load per class (None for all)
            verbose: Whether to show progress bars
            
        Returns:
            x_train, x_test, y_train, y_test: Train/test split data
        """
        data = []
        labels = []
        
        if verbose:
            print("Loading dataset into memory...")
            
        for class_name, class_index in self.class_map.items():
            class_dir = os.path.join(self.data_dir, class_name)
            if not os.path.exists(class_dir):
                print(f"Warning: {class_dir} does not exist")
                continue
                
            files = [f for f in os.listdir(class_dir) 
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            if max_per_class is not None:
                files = files[:max_per_class]
                
            if verbose:
                print(f"Processing {len(files)} images for class {class_name}...")
                
            for file in tqdm(files, disable=not verbose):
                try:
                    img_path = os.path.join(class_dir, file)
                    img = Image.open(img_path).resize(self.image_size)
                    img_array = np.array(img)
                    
                    # Only include RGB images with correct dimensions
                    if img_array.shape == (self.image_size[0], self.image_size[1], 3):
                        data.append(img_array)
                        label = self.encoder.transform([[class_index]])
                        labels.append(label.flatten())
                except Exception as e:
                    if verbose:
                        print(f"Error processing {file}: {str(e)}")
        
        data = np.array(data) / 255.0  # Normalize to [0,1]
        labels = np.array(labels)
        
        # Split into training and testing sets
        x_train, x_test, y_train, y_test = train_test_split(
            data, labels, test_size=0.15, shuffle=True, random_state=42
        )
        
        if verbose:
            print(f"Dataset loaded: {x_train.shape[0]} training samples, {x_test.shape[0]} testing samples")
        
        return x_train, x_test, y_train, y_test
    
    def visualize_samples(self, samples=5):
        """Visualize sample images from each class"""
        for class_name in self.class_map.keys():
            class_dir = os.path.join(self.data_dir, class_name)
            if not os.path.exists(class_dir):
                print(f"Warning: {class_dir} does not exist")
                continue
                
            files = [f for f in os.listdir(class_dir) 
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            if not files:
                continue
                
            sample_files = random.sample(files, min(samples, len(files)))
            
            plt.figure(figsize=(15, 3))
            plt.suptitle(f"Sample images from {class_name}")
            
            for i, file in enumerate(sample_files):
                img_path = os.path.join(class_dir, file)
                img = Image.open(img_path)
                
                plt.subplot(1, samples, i+1)
                plt.imshow(img)
                plt.title(f"Sample {i+1}")
                plt.axis('off')
                
            plt.tight_layout()
            plt.show()