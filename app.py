# app.py (modify the forecasting_route function)

from flask import Flask, render_template, request     # jsonify, url_for
import os
import sys
# import boto3
import datetime

# Add the current directory to the Python path to import your modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


# Import your refactored ML model logic
from sentiment_analyzer import SentimentAnalyzer
from clustering_models import ClusteringModels # KEEP for later if needed
from delivery_analyzer import DeliveryAnalyzer # KEEP for later if needed
# from sales_forecaster import SalesForecaster # Uncomment when sales forecasting is ready

app = Flask(__name__)

# --- Jinja2 Custom Filter ---
@app.template_filter('format_currency')
def format_currency_filter(value, currency_symbol='₹'):
    """Formats a number as currency."""
    try:
        # Ensure value is convertible to float and format with 2 decimal places
        return f"{currency_symbol}{float(value):,.2f}"
    except (ValueError, TypeError):
        # Fallback for non-numeric or problematic values, return as is or a placeholder
        return f"{currency_symbol}N/A" # or just return value if you prefer

# --- Global variables to store loaded model instances ---
sentiment_analyzer_obj = None
clustering_models_obj = None # KEEP for later if needed
delivery_analyzer_obj = None # KEEP for later if needed
sales_forecaster_obj = None # KEEP for later if needed


# --- Function to load models/analyzers (executed once on app startup) ---
def load_all_models_on_startup():
    global sentiment_analyzer_obj, clustering_models_obj, delivery_analyzer_obj, sales_forecaster_obj

    print("Attempting to load all models and analyzers...")

    # Initialize Sentiment Analyzer (Primary focus: Loading from S3)
    try:
        # Load from S3: Using your S3 bucket 'ecom-models-007'
        # The s3_model_key_prefix is set to '' because your model files
        # (like model.safetensors, config.json, vocab.txt, etc.)
        # are located DIRECTLY IN THE ROOT of your S3 bucket.
        sentiment_analyzer_obj = SentimentAnalyzer(
            s3_bucket_name='ecom-models-007', # Your S3 bucket name
            s3_model_key_prefix='' # <--- THIS IS KEY: Empty string means look in the bucket's root
        )
        print("SentimentAnalyzer initialized (from S3).")
    except Exception as e:
        print(f"Failed to initialize SentimentAnalyzer from S3: {e}")
        sentiment_analyzer_obj = None
        print("Falling back to local model loading if S3 load failed...")
        try:
            # Fallback if S3 fails: Attempt to load from local ./sentiment_model directory
            # or download from HuggingFace Hub if not found locally.
            sentiment_analyzer_obj = SentimentAnalyzer(model_name_or_path="./sentiment_model")
            print("SentimentAnalyzer initialized (from local path/HuggingFace Hub).")
        except Exception as e_fallback:
            print(f"CRITICAL: SentimentAnalyzer could not load model from S3 OR local/Hub: {e_fallback}")
            sentiment_analyzer_obj = None


    # Initialize Clustering Models (Will show warnings if PKL files are missing, but won't crash)
    try:
        clustering_models_obj = ClusteringModels(
            seller_model_path="models/seller_clustering_model.pkl",
            review_model_path="models/review_clustering_model.pkl",
            customer_model_path="models/customer_clustering_model.pkl"
        )
        print("ClusteringModels initialized.")
    except Exception as e:
        print(f"Failed to initialize ClusteringModels: {e}")
        clustering_models_obj = None


    # Initialize Delivery Analyzer (Will use dummy rules if no precomputed data)
    try:
        delivery_analyzer_obj = DeliveryAnalyzer(precomputed_risk_data_path=None)
        print("DeliveryAnalyzer initialized.")
    except Exception as e:
        print(f"Failed to initialize DeliveryAnalyzer: {e}")
        delivery_analyzer_obj = None


    # Initialize Sales Forecaster (Uncomment when ready)
    # try:
    #     sales_forecaster_obj = SalesForecaster(model_path="models/sales_forecasting_model.pkl")
    #     print("SalesForecaster initialized.")
    # except Exception as e:
    #     print(f"Failed to initialize SalesForecaster: {e}")
    #     sales_forecaster_obj = None

    print("All model/analyzer initializations attempted.")


# --- Flask Routes ---

@app.route('/')
def index():
    """Renders the main homepage."""
    return render_template('index.html')

@app.route('/sentiment', methods=['GET', 'POST'])
def sentiment_route():
    """Handles requests for the Sentiment Analysis page."""
    result = None
    # If the request is a POST (form submission), process it
    if request.method == 'POST':
        if sentiment_analyzer_obj:
            text_input = request.form.get('review_text')
            result = sentiment_analyzer_obj.analyze_sentiment(text_input)
        else:
            result = {"error": "Sentiment analysis service is not available. Please check server logs."}
    # For GET requests or after POST, render the template
    return render_template('customer_emotion_analysis.html', result=result)


@app.route('/delivery', methods=['GET', 'POST'])
def delivery_route():
    """Handles requests for the Delivery Delay Analysis page."""
    result = None
    if request.method == 'POST':
        if delivery_analyzer_obj:
            region_input = request.form.get('region')
            seller_input = request.form.get('seller_id')
            result = delivery_analyzer_obj.analyze_delivery_risk(region_input, seller_input)
        else:
            result = {"error": "Delivery analysis service is not available. Please check server logs."}
    return render_template('delivery_delay_prevention.html', result=result)

@app.route('/clustering', methods=['GET', 'POST'])
def clustering_route():
    """Handles requests for the Product/User Clustering page."""
    segment_type = request.args.get('type', 'customer')
    result = None

    if request.method == 'POST':
        if clustering_models_obj:
            try:
                features_input = []
                if segment_type == 'seller':
                    features_input.append(float(request.form.get('seller_feature1', 0)))
                    features_input.append(float(request.form.get('seller_feature2', 0)))
                    predicted_cluster = clustering_models_obj.predict_seller_segment(features_input)
                    result = {"type": "Seller", "input_features": features_input, "cluster": predicted_cluster}
                elif segment_type == 'review':
                    features_input.append(float(request.form.get('review_sentiment_score', 0)))
                    features_input.append(float(request.form.get('review_length', 0)))
                    predicted_cluster = clustering_models_obj.predict_review_segment(features_input)
                    result = {"type": "Review", "input_features": features_input, "cluster": predicted_cluster}
                elif segment_type == 'customer':
                    features_input.append(float(request.form.get('customer_feature1', 0)))
                    features_input.append(float(request.form.get('customer_feature2', 0)))
                    features_input.append(float(request.form.get('customer_feature3', 0)))
                    predicted_cluster = clustering_models_obj.predict_customer_segment(features_input)
                    result = {"type": "Customer", "input_features": features_input, "cluster": predicted_cluster}
                else:
                    result = {"error": "Invalid clustering type selected."}
            except ValueError:
                result = {"error": "Invalid input for features. Please ensure all inputs are numbers."}
            except Exception as e:
                result = {"error": f"An unexpected error occurred during clustering: {e}"}
        else:
            result = {"error": "Clustering models service is not available. Please check server logs."}
    return render_template('buying_pattern_recognition.html', result=result, segment_type=segment_type)

@app.route('/forecasting', methods=['GET', 'POST'])
def forecasting_route():
    """Handles requests for the Sales Forecasting page."""
    result = None
    if request.method == 'POST':
        if sales_forecaster_obj: # This is still None, so it will hit the else block
            try:
                periods = int(request.form.get('periods'))
                
                # --- MODIFIED DUMMY DATA GENERATION ---
                # Generate realistic-looking dummy sales data for currency formatting
                # Start from a base value and add some fluctuation
                base_sales = 10000.0
                forecast_values_list = []
                current_date = datetime.date.today()
                
                for i in range(periods):
                    # Simulate some increasing trend and random fluctuation
                    sales_value = base_sales + (i * 500) + (i % 3 * 100) + (i % 2 * -50) + (i % 4 * 25)
                    forecast_values_list.append(sales_value)
                    
                    # Generate future dates (e.g., next month, next quarter, etc.)
                    # Simple example: just increment month for display
                    if i == 0:
                        future_date_str = "Next Month"
                    elif i == 1:
                        future_date_str = "Month after Next"
                    else:
                        future_date_str = f"Future Period {i+1}"
                    
                    # You can make this more sophisticated based on `periods` (weeks, months, quarters)
                    # For now, keeping it simple as "Period X"
                    
                predicted_sales = list(zip([f"Period {i+1}" for i in range(periods)], forecast_values_list))
                # --- END MODIFIED DUMMY DATA GENERATION ---

                result = {
                    "periods": periods,
                    "forecast": predicted_sales # Now contains (string, float) tuples
                }
            except ValueError:
                result = {"error": "Invalid input for periods. Please enter a valid integer."}
            except Exception as e:
                result = {"error": f"An unexpected error occurred during forecasting: {e}"}
        else:
            # This block will be hit until SalesForecaster is uncommented and functional
            try:
                periods = int(request.form.get('periods'))
                # Fallback dummy data generation if sales_forecaster_obj is None
                base_sales = 5000.0
                forecast_values_list = []
                for i in range(periods):
                    sales_value = base_sales + (i * 200) + (i % 4 * 50)
                    forecast_values_list.append(sales_value)
                
                predicted_sales = list(zip([f"Period {i+1}" for i in range(periods)], forecast_values_list))
                
                result = {
                    "periods": periods,
                    "forecast": predicted_sales
                }
            except ValueError:
                result = {"error": "Invalid input for periods. Please enter a valid integer."}
            except Exception as e:
                result = {"error": "Sales forecasting service is not fully configured or available. Using simple dummy data. Error: " + str(e)}

    return render_template('accurate_sales_forecasting.html', result=result)

@app.route('/team')
def team_route():
    """Renders the Team page."""
    return render_template('team.html')

@app.route('/support')
def support_route():
    """Renders the Support page."""
    return render_template('support.html')

if __name__ == '__main__':
    if not os.path.exists('models'):
        os.makedirs('models')
    
    load_all_models_on_startup()

    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=port)