# clustering_models.py
import boto3
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler # You might need this if your clustering models expect scaled data

class ClusteringModels:
    def __init__(self, seller_model_path="models/seller_clustering_model.pkl",
                 review_model_path="models/review_clustering_model.pkl",
                 customer_model_path="models/customer_clustering_model.pkl"):
        """
        Initializes ClusteringModels by loading pre-trained clustering models from local paths.
        """
        self.seller_model = self._load_model(seller_model_path)
        self.review_model = self._load_model(review_model_path)
        self.customer_model = self._load_model(customer_model_path)
        # You might also need to load a StandardScaler if your models were trained on scaled data
        # self.scaler = self._load_model("models/your_scaler.pkl")

    def _load_model(self, path):
        """Helper function to load a pickled model."""
        try:
            with open(path, 'rb') as f:
                model = pickle.load(f)
            print(f"Clustering model loaded from: {path}")
            return model
        except FileNotFoundError:
            print(f"Warning: Clustering model not found at {path}. Please ensure it's saved correctly.")
            return None
        except Exception as e:
            print(f"Error loading clustering model from {path}: {e}")
            return None

    def predict_seller_segment(self, features: list):
        """Predicts the seller segment based on a list of numerical features."""
        if not self.seller_model:
            return "Error: Seller clustering model not loaded."
        # Ensure input is a 2D array for single prediction
        input_data = np.array(features).reshape(1, -1)
        # If you have a scaler, apply it: input_data = self.scaler.transform(input_data)
        cluster = self.seller_model.predict(input_data)[0]
        return int(cluster)

    def predict_review_segment(self, features: list):
        """Predicts the review segment based on a list of numerical features (e.g., sentiment score, review length)."""
        if not self.review_model:
            return "Error: Review clustering model not loaded."
        input_data = np.array(features).reshape(1, -1)
        cluster = self.review_model.predict(input_data)[0]
        return int(cluster)

    def predict_customer_segment(self, features: list):
        """Predicts the customer segment based on a list of numerical features (e.g., purchase frequency, avg spend)."""
        if not self.customer_model:
            return "Error: Customer clustering model not loaded."
        input_data = np.array(features).reshape(1, -1)
        cluster = self.customer_model.predict(input_data)[0]
        return int(cluster)

# This block is for testing the class directly (optional)
if __name__ == "__main__":
    # For testing, you'll need dummy .pkl files in the 'models' folder
    # Create dummy models if you don't have them yet:
    from sklearn.cluster import KMeans
    import os
    if not os.path.exists('models'):
        os.makedirs('models')

    # Dummy Seller Clustering Model (e.g., 2 features)
    dummy_seller_model = KMeans(n_clusters=3, random_state=42, n_init=10)
    dummy_seller_model.fit(np.random.rand(100, 2)) # Fit with some random data
    with open('models/seller_clustering_model.pkl', 'wb') as f:
        pickle.dump(dummy_seller_model, f)

    # Dummy Review Clustering Model (e.g., 2 features)
    dummy_review_model = KMeans(n_clusters=4, random_state=42, n_init=10)
    dummy_review_model.fit(np.random.rand(100, 2))
    with open('models/review_clustering_model.pkl', 'wb') as f:
        pickle.dump(dummy_review_model, f)

    # Dummy Customer Clustering Model (e.g., 3 features)
    dummy_customer_model = KMeans(n_clusters=5, random_state=42, n_init=10)
    dummy_customer_model.fit(np.random.rand(100, 3))
    with open('models/customer_clustering_model.pkl', 'wb') as f:
        pickle.dump(dummy_customer_model, f)

    # Now test the ClusteringModels class
    cluster_analyzer = ClusteringModels(
        seller_model_path="models/seller_clustering_model.pkl",
        review_model_path="models/review_clustering_model.pkl",
        customer_model_path="models/customer_clustering_model.pkl"
    )

    print("\nTesting Seller Segmentation:")
    print(f"Prediction for [0.5, 0.8]: {cluster_analyzer.predict_seller_segment([0.5, 0.8])}")

    print("\nTesting Review-Based Segmentation:")
    print(f"Prediction for [0.2, 0.9]: {cluster_analyzer.predict_review_segment([0.2, 0.9])}")

    print("\nTesting Customer Segmentation:")
    print(f"Prediction for [0.1, 0.3, 0.7]: {cluster_analyzer.predict_customer_segment([0.1, 0.3, 0.7])}")