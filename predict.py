import os
import cv2
import pickle
import argparse
from classifier import CatDogClassifier

def predict_image(image_path):
    # Initialize classifier with same parameters as training
    classifier = CatDogClassifier()
    
    try:
        # Make prediction
        label, probabilities = classifier.predict(image_path)
        confidence = probabilities[1] if label == 'Dog' else probabilities[0]
        
        return label, confidence
    except Exception as e:
        raise Exception(f"Error during prediction: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='Predict if an image contains a cat or a dog')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to the image to classify')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.image):
        print(f"Error: Image file '{args.image}' not found")
        return
    
    try:
        label, confidence = predict_image(args.image)
        print(f"\nPrediction: {label}")
        print(f"Confidence: {confidence:.2%}")
    except Exception as e:
        print(f"Error during prediction: {str(e)}")

if __name__ == '__main__':
    main()