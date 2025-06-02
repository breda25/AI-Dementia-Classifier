import numpy as np 
import os
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Dropout, BatchNormalization, MaxPooling2D

class DementiaModel:
    def __init__(self, input_shape=(128, 128, 3), num_classes=4):
        """
        Initialize the model
        
        Args:
            input_shape: Shape of input images (height, width, channels)
            num_classes: Number of classes for classification
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self.build_model()
        
    def build_model(self):
        """Build and compile the CNN model"""
        model = Sequential()
        
        # First Convolutional Block
        model.add(Conv2D(32, kernel_size=(3, 3), input_shape=self.input_shape, padding='Same'))
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='Same'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        
        # Second Convolutional Block
        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='Same'))
        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='Same'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Dropout(0.25))
        
        # Third Convolutional Block (deeper network for 86K dataset)
        model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='Same'))
        model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='Same'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Dropout(0.25))
        
        # Flatten and Dense layers
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.num_classes, activation='softmax'))
        
        # Compile model
        model.compile(
            loss='categorical_crossentropy',
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, train_data, validation_data=None, epochs=10, batch_size=32, callbacks=None):
        """
        Train the model
        
        Args:
            train_data: Training data as (x_train, y_train) tuple or data generator
            validation_data: Validation data as (x_val, y_val) tuple or data generator
            epochs: Number of training epochs
            batch_size: Batch size for training (if using data tuples)
            callbacks: List of Keras callbacks
            
        Returns:
            Training history
        """
        # Create default callbacks if none provided
        if callbacks is None:
            checkpoint_path = os.path.join('models', 'checkpoint_{epoch:02d}_{val_accuracy:.4f}.h5')
            os.makedirs('models', exist_ok=True)
            
            callbacks = [
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=checkpoint_path,
                    save_best_only=True,
                    monitor='val_accuracy',
                    mode='max',
                    verbose=1
                ),
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=5,
                    restore_best_weights=True
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.2,
                    patience=3,
                    min_lr=0.00001
                ),
                tf.keras.callbacks.TensorBoard(
                    log_dir='./logs',
                    histogram_freq=1
                )
            ]
        
        # Train with either generators or numpy arrays
        if isinstance(train_data, tuple):
            x_train, y_train = train_data
            x_val, y_val = validation_data if validation_data else (None, None)
            
            history = self.model.fit(
                x_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=(x_val, y_val) if validation_data else None,
                callbacks=callbacks,
                verbose=1
            )
        else:
            # Assuming train_data is a generator
            history = self.model.fit(
                train_data,
                epochs=epochs,
                validation_data=validation_data,
                callbacks=callbacks,
                verbose=1
            )
            
        return history
    
    def evaluate(self, test_data):
        """Evaluate the model on test data"""
        if isinstance(test_data, tuple):
            x_test, y_test = test_data
            return self.model.evaluate(x_test, y_test, verbose=1)
        else:
            # Assuming test_data is a generator
            return self.model.evaluate(test_data, verbose=1)
    
    def predict(self, image):
        """
        Make a prediction on a single image
        
        Args:
            image: Numpy array of shape (height, width, channels) or (1, height, width, channels)
            
        Returns:
            Class prediction and confidence
        """
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
            
        prediction = self.model.predict(image)
        class_index = np.argmax(prediction[0])
        confidence = prediction[0][class_index] * 100
        
        return class_index, confidence
    
    def save(self, filepath):
        """Save the model to disk"""
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath):
        """Load a model from disk"""
        model_instance = cls()
        model_instance.model = load_model(filepath)
        print(f"Model loaded from {filepath}")
        return model_instance
    
    def plot_training_history(self, history):
        """Plot training and validation metrics"""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        
        plt.tight_layout()
        plt.show()