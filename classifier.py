import os
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

class CatDogClassifier:
    def __init__(self, img_size=(128, 128)):
        self.img_size = img_size
        self.model = SVC(kernel='linear', probability=True)
        
    def _extract_features(self, img):
        # Resize image
        img = cv2.resize(img, self.img_size)
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Calculate HOG features
        hog = cv2.HOGDescriptor()
        features = hog.compute(gray)
        return features.flatten()
    
    def prepare_data(self, data_dir):
        features = []
        labels = []
        
        # Process training and test sets
        for split in ['train', 'test']:
            split_dir = os.path.join(data_dir, split)
            
            # Load cat images
            cat_dir = os.path.join(split_dir, 'cats')
            for img_name in os.listdir(cat_dir):
                img_path = os.path.join(cat_dir, img_name)
                img = cv2.imread(img_path)
                if img is not None:
                    features.append(self._extract_features(img))
                    labels.append(0)  # 0 for cats
                    
            # Load dog images
            dog_dir = os.path.join(split_dir, 'dogs')
            for img_name in os.listdir(dog_dir):
                img_path = os.path.join(dog_dir, img_name)
                img = cv2.imread(img_path)
                if img is not None:
                    features.append(self._extract_features(img))
                    labels.append(1)  # 1 for dogs
        
        # Convert to numpy arrays
        X = np.array(features)
        y = np.array(labels)
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
    def train(self):
        print("Training the model...")
        self.model.fit(self.X_train, self.y_train)
        
        # Calculate training and validation accuracy
        train_accuracy = self.model.score(self.X_train, self.y_train)
        val_accuracy = self.model.score(self.X_test, self.y_test)
        
        print(f'Training accuracy: {train_accuracy:.4f}')
        print(f'Validation accuracy: {val_accuracy:.4f}')
        
        return train_accuracy, val_accuracy
    
    def evaluate(self):
        y_pred = self.model.predict(self.X_test)
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred, target_names=['Cat', 'Dog']))
    
    def predict(self, img_path):
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Could not load image at {img_path}")
        
        features = self._extract_features(img)
        prediction = self.model.predict([features])[0]
        probability = self.model.predict_proba([features])[0]
        
        return 'Dog' if prediction == 1 else 'Cat', probability

def main():
    # Initialize the classifier
    classifier = CatDogClassifier()
    
    # Prepare data
    data_dir = 'dataset'
    if not os.path.exists(data_dir):
        print(f"Please create a 'dataset' directory with 'cats' and 'dogs' subdirectories containing respective images")
        return
    
    print("Preparing data...")
    classifier.prepare_data(data_dir)
    
    # Train the model
    print("\nTraining model...")
    classifier.train()
    
    # Evaluate the model
    print("\nEvaluating model...")
    classifier.evaluate()

if __name__ == '__main__':
    main()