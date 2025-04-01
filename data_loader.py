import os
import cv2
import numpy as np
from pathlib import Path
import requests
from urllib.parse import urljoin

class DataLoader:
    def __init__(self, data_dir="dataset"):
        self.data_dir = Path(data_dir)
        self.train_dir = self.data_dir / "train"
        self.test_dir = self.data_dir / "test"

    def create_sample_dataset(self):
        """Create a sample dataset with random images for testing"""
        try:
            # Create directories
            self.setup_directory_structure()
            
            # Create sample images
            for category in ['cats', 'dogs']:
                for split in ['train', 'test']:
                    dir_path = self.data_dir / split / category
                    
                    # Create 10 training images and 5 test images per category
                    num_images = 10 if split == 'train' else 5
                    
                    for i in range(num_images):
                        # Create a random image (128x128 RGB)
                        img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
                        
                        # Add some basic shapes to make it more realistic
                        if category == 'cats':
                            # Add triangular ears for cats
                            cv2.drawContours(img, [np.array([[64,30], [48,60], [80,60]])], 0, (200,200,200), -1)
                        else:
                            # Add rectangular ears for dogs
                            cv2.rectangle(img, (48,30), (80,60), (200,200,200), -1)
                        
                        # Add a circular face
                        cv2.circle(img, (64,64), 30, (180,180,180), -1)
                        
                        # Save the image
                        img_path = dir_path / f"{category[:-1]}_{i+1}.jpg"
                        cv2.imwrite(str(img_path), img)
            
            print("Sample dataset created successfully!")
            return True
        except Exception as e:
            print(f"Error creating sample dataset: {str(e)}")
            return False

    def setup_directory_structure(self):
        """Create necessary directories for organizing the dataset"""
        os.makedirs(self.train_dir, exist_ok=True)
        os.makedirs(self.test_dir, exist_ok=True)
        os.makedirs(self.train_dir / "cats", exist_ok=True)
        os.makedirs(self.train_dir / "dogs", exist_ok=True)
        os.makedirs(self.test_dir / "cats", exist_ok=True)
        os.makedirs(self.test_dir / "dogs", exist_ok=True)

    def validate_dataset(self):
        """Validate that the dataset was downloaded and organized correctly"""
        required_dirs = [
            self.train_dir / "cats",
            self.train_dir / "dogs",
            self.test_dir / "cats",
            self.test_dir / "dogs"
        ]
        
        for directory in required_dirs:
            if not directory.exists():
                print(f"Error: Required directory {directory} does not exist")
                return False
            
            # Check if directory contains images
            if len(list(directory.glob("*.jpg"))) == 0:
                print(f"Warning: No images found in {directory}")
                
        return True

    def download_dataset(self):
        """Create a sample dataset with random images for testing"""
        try:
            # Create directories
            self.setup_directory_structure()
            
            # Create sample images
            for category in ['cats', 'dogs']:
                for split in ['train', 'test']:
                    dir_path = self.data_dir / split / category
                    
                    # Create 6 training images and 4 test images per category
                    num_images = 6 if split == 'train' else 4
                    
                    for i in range(num_images):
                        # Create a random image (128x128 RGB)
                        img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
                        
                        # Add some basic shapes to make it more realistic
                        if category == 'cats':
                            # Add triangular ears for cats
                            cv2.drawContours(img, [np.array([[64,30], [48,60], [80,60]])], 0, (200,200,200), -1)
                            # Add pointy nose
                            cv2.drawContours(img, [np.array([[60,70], [64,80], [68,70]])], 0, (150,150,150), -1)
                            # Add whiskers
                            cv2.line(img, (64,75), (84,70), (100,100,100), 1)
                            cv2.line(img, (64,75), (84,75), (100,100,100), 1)
                            cv2.line(img, (64,75), (84,80), (100,100,100), 1)
                            cv2.line(img, (64,75), (44,70), (100,100,100), 1)
                            cv2.line(img, (64,75), (44,75), (100,100,100), 1)
                            cv2.line(img, (64,75), (44,80), (100,100,100), 1)
                        else:
                            # Add rectangular ears for dogs
                            cv2.rectangle(img, (48,30), (58,60), (200,200,200), -1)
                            cv2.rectangle(img, (70,30), (80,60), (200,200,200), -1)
                            # Add nose
                            cv2.circle(img, (64,75), 8, (100,100,100), -1)
                            # Add tongue
                            cv2.ellipse(img, (64,85), (8,12), 0, 0, 180, (200,100,100), -1)
                        
                        # Add a circular face
                        cv2.circle(img, (64,64), 30, (180,180,180), -1)
                        # Add eyes
                        cv2.circle(img, (54,60), 5, (50,50,50), -1)
                        cv2.circle(img, (74,60), 5, (50,50,50), -1)
                        
                        # Save the image
                        img_path = dir_path / f"{category[:-1]}_{i+1}.jpg"
                        cv2.imwrite(str(img_path), img)
                        print(f"Created {category[:-1]} image {i+1} in {split} set")
            
            print("Sample dataset created successfully!")
            return True
        except Exception as e:
            print(f"Error downloading dataset: {str(e)}")
            return False


def main():
    loader = DataLoader()
    
    # Download and setup dataset
    if loader.download_dataset():
        if loader.validate_dataset():
            print("Dataset is ready for use!")
        else:
            print("Dataset validation failed!")

if __name__ == "__main__":
    main()