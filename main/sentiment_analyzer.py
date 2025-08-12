# sentiment_analyzer.py
import boto3
import os
import boto3 # AWS SDK for Python
import tempfile # To create temporary directories for downloaded models
import shutil # To clean up temporary directories

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# --- Helper function to download model directory from S3 ---
def download_model_from_s3(s3_bucket_name, s3_model_key_prefix, local_dir):
    """
    Downloads all files from a specific S3 prefix (which acts like a folder)
    to a local directory.
    """
    s3_client = boto3.client('s3')
    print(f"Downloading model files from s3://{s3_bucket_name}/{s3_model_key_prefix} to {local_dir}...")

    try:
        # List objects under the specified prefix
        response = s3_client.list_objects_v2(Bucket=s3_bucket_name, Prefix=s3_model_key_prefix)
        if 'Contents' not in response:
            print(f"ERROR: No objects found in S3 bucket '{s3_bucket_name}' with prefix '{s3_model_key_prefix}'.")
            return False

        for obj in response['Contents']:
            file_key = obj['Key']
            # Skip directories themselves, only download files
            if file_key.endswith('/'):
                continue

            # Create any necessary subdirectories locally
            local_file_path = os.path.join(local_dir, os.path.relpath(file_key, s3_model_key_prefix))
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

            print(f"  Downloading: {file_key} to {local_file_path}")
            s3_client.download_file(s3_bucket_name, file_key, local_file_path)
        print("All model files downloaded successfully from S3.")
        return True
    except Exception as e:
        print(f"ERROR during S3 download: {e}")
        return False


class SentimentAnalyzer:
    def __init__(self, s3_bucket_name=None, s3_model_key_prefix=None, model_name_or_path="distilbert-base-uncased-finetuned-sst-2-english"):
        """
        Initializes the SentimentAnalyzer.
        Prioritizes loading from S3 if s3_bucket_name and s3_model_key_prefix are provided.
        Otherwise, falls back to local path or HuggingFace Hub download.
        """
        self.tokenizer = None
        self.model = None
        self.labels = ["Negative", "Positive"] # For this specific DistilBERT model
        self.local_model_dir = None # To store temporary directory if downloaded from S3

        model_loaded = False

        # Attempt to load from S3 first if S3 details are provided
        if s3_bucket_name and s3_model_key_prefix:
            try:
                # Create a temporary directory to store the downloaded model
                self.local_model_dir = tempfile.mkdtemp()
                print(f"Attempting to download model from S3 to temporary directory: {self.local_model_dir}")

                if download_model_from_s3(s3_bucket_name, s3_model_key_prefix, self.local_model_dir):
                    model_name_or_path_to_load = self.local_model_dir # Load from the downloaded path
                    print(f"Attempting to load model from downloaded S3 path: {model_name_or_path_to_load}")
                    self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path_to_load)
                    self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path_to_load)
                    model_loaded = True
                    print(f"Sentiment model loaded successfully from S3 ({s3_bucket_name}/{s3_model_key_prefix}).")
            except Exception as e:
                print(f"ERROR: Failed to load sentiment model from S3: {e}")
                if self.local_model_dir and os.path.exists(self.local_model_dir):
                    shutil.rmtree(self.local_model_dir) # Clean up temp dir on error
                self.local_model_dir = None # Reset to indicate S3 load failed

        # Fallback to local path (like ./sentiment_model) or HuggingFace Hub if S3 load failed or not configured
        if not model_loaded:
            print(f"S3 load failed or not configured. Attempting to load model from: {model_name_or_path} (local path or HuggingFace Hub).")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
                self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
                model_loaded = True
                print(f"Sentiment model '{model_name_or_path}' loaded successfully (from local or HuggingFace Hub).")
            except Exception as e:
                print(f"ERROR: Failed to load sentiment model from {model_name_or_path}: {e}")
                self.tokenizer = None
                self.model = None

        if not model_loaded:
            print("CRITICAL: SentimentAnalyzer could not load any model. Check paths/internet/S3 config.")


    def analyze_sentiment(self, text: str):
        """
        Analyzes the sentiment of a given text string.
        """
        if not self.tokenizer or not self.model:
            return {"sentiment": "Error: Model not loaded", "confidence": "0.00%", "text": text}
        if not text or not isinstance(text, str) or text.strip() == "":
            return {"sentiment": "Neutral", "confidence": "N/A - No text provided", "text": text}

        inputs = self.tokenizer.encode(text, return_tensors="pt")
        with torch.no_grad():
            logits = self.model(inputs).logits

        probabilities = torch.softmax(logits, dim=1).tolist()[0]
        predicted_index = torch.argmax(logits).item()

        sentiment_label = self.labels[predicted_index]
        confidence_score = probabilities[predicted_index] * 100

        formatted_probabilities = {label: f"{prob:.2f}%" for label, prob in zip(self.labels, probabilities)}

        return {
            "text": text,
            "sentiment": sentiment_label,
            "confidence": f"{confidence_score:.2f}%",
            "probabilities": formatted_probabilities
        }

    def __del__(self):
        """Cleans up the temporary directory when the object is deleted."""
        if self.local_model_dir and os.path.exists(self.local_model_dir):
            try:
                shutil.rmtree(self.local_model_dir)
                print(f"Cleaned up temporary model directory: {self.local_model_dir}")
            except OSError as e:
                print(f"Error removing temporary directory {self.local_model_dir}: {e}")

if __name__ == "__main__":
    # For local testing, ensure AWS credentials are configured (e.g., via aws configure)
    # Replace 'your-ecom-insight-models-siddhant-2025' with your S3 bucket name
    # Replace 'sentiment_model/' with the folder path inside your S3 bucket
    analyzer = SentimentAnalyzer(s3_bucket_name='your-ecom-insight-models-siddhant-2025', s3_model_key_prefix='sentiment_model/')
    # OR, to test only from HuggingFace Hub directly (no S3):
    # analyzer = SentimentAnalyzer(model_name_or_path="distilbert-base-uncased-finetuned-sst-2-english")

    test_reviews = ["This is great!", "I am very disappointed.", "It is what it is."]
    for review in test_reviews:
        result = analyzer.analyze_sentiment(review)
        print(f"Review: '{result['text']}' -> Sentiment: {result['sentiment']} (Confidence: {result['confidence']})")