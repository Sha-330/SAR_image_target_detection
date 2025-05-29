import os
import numpy as np
import cv2
import joblib
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from kymatio.numpy import Scattering2D
import pandas as pd
from pathlib import Path

class RadarModelTester:
    def __init__(self, model_path="radarmodel.h5", label_encoder_path="label.pkl", image_size=64):
        """
        Initialize the radar model tester
        
        Args:
            model_path: Path to the trained model (.h5 file)
            label_encoder_path: Path to the label encoder (.pkl file)
            image_size: Size to resize images to (should match training size)
        """
        self.image_size = image_size
        self.model = load_model(model_path)
        self.label_encoder = joblib.load(label_encoder_path)
        self.scattering = Scattering2D(J=2, shape=(image_size, image_size))
        
        print(f"Model loaded successfully!")
        print(f"Classes: {list(self.label_encoder.classes_)}")
        print(f"Model input shape: {self.model.input_shape}")
    
    def preprocess_image(self, image_path):
        """
        Preprocess a single image for classification
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Preprocessed feature vector
        """
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Resize and normalize
        img = cv2.resize(img, (self.image_size, self.image_size))
        img = img.astype(np.float32) / 255.0
        
        # Apply scattering transform
        features = self.scattering(img).flatten()
        return features
    
    def predict_single_image(self, image_path, show_probabilities=False):
        """
        Classify a single image
        
        Args:
            image_path: Path to the image file
            show_probabilities: Whether to return class probabilities
            
        Returns:
            Dictionary with prediction results
        """
        try:
            # Preprocess image
            features = self.preprocess_image(image_path)
            features = features.reshape(1, -1)  # Add batch dimension
            
            # Make prediction
            predictions = self.model.predict(features, verbose=0)
            predicted_class_idx = np.argmax(predictions[0])
            predicted_class = self.label_encoder.classes_[predicted_class_idx]
            confidence = predictions[0][predicted_class_idx]
            
            result = {
                'image_path': image_path,
                'predicted_class': predicted_class,
                'confidence': confidence,
                'success': True
            }
            
            if show_probabilities:
                class_probabilities = {}
                for i, class_name in enumerate(self.label_encoder.classes_):
                    class_probabilities[class_name] = predictions[0][i]
                result['class_probabilities'] = class_probabilities
            
            return result
            
        except Exception as e:
            return {
                'image_path': image_path,
                'error': str(e),
                'success': False
            }
    
    def predict_multiple_images(self, image_paths, show_probabilities=False):
        """
        Classify multiple images
        
        Args:
            image_paths: List of image file paths
            show_probabilities: Whether to return class probabilities
            
        Returns:
            List of prediction results
        """
        results = []
        
        print(f"Processing {len(image_paths)} images...")
        for i, image_path in enumerate(image_paths):
            print(f"Processing image {i+1}/{len(image_paths)}: {os.path.basename(image_path)}")
            result = self.predict_single_image(image_path, show_probabilities)
            results.append(result)
        
        return results
    
    def predict_from_directory(self, directory_path, show_probabilities=False, 
                             extensions=('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
        """
        Classify all images in a directory
        
        Args:
            directory_path: Path to directory containing images
            show_probabilities: Whether to return class probabilities
            extensions: Tuple of valid image extensions
            
        Returns:
            List of prediction results
        """
        image_paths = []
        
        # Find all image files in directory
        for ext in extensions:
            image_paths.extend(Path(directory_path).glob(f"*{ext}"))
            image_paths.extend(Path(directory_path).glob(f"*{ext.upper()}"))
        
        image_paths = [str(path) for path in image_paths]
        
        if not image_paths:
            print(f"No images found in directory: {directory_path}")
            return []
        
        print(f"Found {len(image_paths)} images in directory: {directory_path}")
        return self.predict_multiple_images(image_paths, show_probabilities)
    
    def display_results(self, results, save_to_csv=None):
        """
        Display and optionally save prediction results
        
        Args:
            results: List of prediction results
            save_to_csv: Path to save results as CSV (optional)
        """
        successful_results = [r for r in results if r['success']]
        failed_results = [r for r in results if not r['success']]
        
        print(f"\n{'='*60}")
        print(f"CLASSIFICATION RESULTS")
        print(f"{'='*60}")
        print(f"Total images processed: {len(results)}")
        print(f"Successful predictions: {len(successful_results)}")
        print(f"Failed predictions: {len(failed_results)}")
        
        if successful_results:
            print(f"\nSuccessful Predictions:")
            print(f"{'-'*60}")
            for result in successful_results:
                filename = os.path.basename(result['image_path'])
                print(f"{filename:<30} -> {result['predicted_class']:<15} (Confidence: {result['confidence']:.3f})")
        
        if failed_results:
            print(f"\nFailed Predictions:")
            print(f"{'-'*60}")
            for result in failed_results:
                filename = os.path.basename(result['image_path'])
                print(f"{filename:<30} -> Error: {result['error']}")
        
        # Class distribution
        if successful_results:
            class_counts = {}
            for result in successful_results:
                class_name = result['predicted_class']
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
            print(f"\nClass Distribution:")
            print(f"{'-'*30}")
            for class_name, count in sorted(class_counts.items()):
                percentage = (count / len(successful_results)) * 100
                print(f"{class_name:<20}: {count:>3} ({percentage:.1f}%)")
        
        # Save to CSV if requested
        if save_to_csv and successful_results:
            df_data = []
            for result in successful_results:
                row = {
                    'filename': os.path.basename(result['image_path']),
                    'full_path': result['image_path'],
                    'predicted_class': result['predicted_class'],
                    'confidence': result['confidence']
                }
                
                # Add class probabilities if available
                if 'class_probabilities' in result:
                    for class_name, prob in result['class_probabilities'].items():
                        row[f'prob_{class_name}'] = prob
                
                df_data.append(row)
            
            df = pd.DataFrame(df_data)
            df.to_csv(save_to_csv, index=False)
            print(f"\nResults saved to: {save_to_csv}")
    
    def visualize_predictions(self, results, max_images=12, figsize=(15, 10)):
        """
        Visualize prediction results with images
        
        Args:
            results: List of prediction results
            max_images: Maximum number of images to display
            figsize: Figure size for the plot
        """
        successful_results = [r for r in results if r['success']][:max_images]
        
        if not successful_results:
            print("No successful predictions to visualize")
            return
        
        n_images = len(successful_results)
        cols = min(4, n_images)
        rows = (n_images + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        if rows == 1:
            axes = axes.reshape(1, -1)
        if cols == 1:
            axes = axes.reshape(-1, 1)
        
        for i, result in enumerate(successful_results):
            row, col = i // cols, i % cols
            ax = axes[row, col] if rows > 1 else axes[col]
            
            # Load and display image
            img = cv2.imread(result['image_path'], cv2.IMREAD_GRAYSCALE)
            ax.imshow(img, cmap='gray')
            
            # Set title with prediction
            filename = os.path.basename(result['image_path'])
            title = f"{filename}\nPred: {result['predicted_class']}\nConf: {result['confidence']:.3f}"
            ax.set_title(title, fontsize=10)
            ax.axis('off')
        
        # Hide unused subplots
        for i in range(n_images, rows * cols):
            row, col = i // cols, i % cols
            ax = axes[row, col] if rows > 1 else axes[col]
            ax.axis('off')
        
        plt.tight_layout()
        plt.show()


# Example usage
def test_radar_model():
    """
    Example function showing how to use the RadarModelTester
    """
    # Initialize the tester
    tester = RadarModelTester(
        model_path="radarmodel.h5",
        label_encoder_path="label.pkl",
        image_size=64
    )
    
    # Method 1: Test single image
    print("Method 1: Testing single image")
    single_result = tester.predict_single_image("path/to/test/image.jpg", show_probabilities=True)
    if single_result['success']:
        print(f"Predicted class: {single_result['predicted_class']}")
        print(f"Confidence: {single_result['confidence']:.3f}")
        if 'class_probabilities' in single_result:
            print("All class probabilities:")
            for class_name, prob in single_result['class_probabilities'].items():
                print(f"  {class_name}: {prob:.3f}")
    else:
        print(f"Error: {single_result['error']}")
    
    # Method 2: Test multiple specific images
    print("\nMethod 2: Testing multiple specific images")
    image_paths = [
        "path/to/test/image1.jpg",
        "path/to/test/image2.jpg",
        "path/to/test/image3.jpg"
    ]
    multiple_results = tester.predict_multiple_images(image_paths, show_probabilities=True)
    tester.display_results(multiple_results, save_to_csv="test_results.csv")
    
    # Method 3: Test all images in a directory
    print("\nMethod 3: Testing all images in directory")
    directory_results = tester.predict_from_directory(
        "path/to/test/directory",
        show_probabilities=True
    )
    tester.display_results(directory_results, save_to_csv="directory_test_results.csv")
    
    # Method 4: Visualize results
    print("\nMethod 4: Visualizing results")
    tester.visualize_predictions(directory_results, max_images=8)


if __name__ == "__main__":
    # Example usage - modify paths as needed
    
    # Quick test example
    tester = RadarModelTester()
    
    # Replace with your actual test image paths
    test_images = [
        r"C:\Users\Ajmal\Documents\programs\SAR\Padded_imgs\BTR_60\HB03333 (another copy).JPG",
        r"C:\Users\Ajmal\Documents\programs\SAR\Padded_imgs\2S1\HB14937.JPG"
    ]
    
    # Or test a directory
    # results = tester.predict_from_directory("test_images_folder")
    
    # Test individual images
    results = []
    for img_path in test_images:
        if os.path.exists(img_path):
            result = tester.predict_single_image(img_path, show_probabilities=True)
            results.append(result)
    
    if results:
        tester.display_results(results, save_to_csv="my_test_results.csv")
        tester.visualize_predictions(results)
    else:
        print("No test images found. Please update the image paths in the script.")