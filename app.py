from flask import Flask, request, jsonify, Response
import pickle
import numpy as np
from numpy.linalg import norm
import urllib.request
import os
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
import pandas as pd
from transformers import pipeline
from flask_cors import CORS # No need for cross_origin decorator if CORS is global

# --- Global Initialization (Loaded once at app startup) ---

# Ensure 'uploads' directory exists
UPLOAD_FOLDER = "./uploads/"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load Deep Learning Model
print("Loading ResNet50 model...")
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False
feature_extractor_model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])
print("ResNet50 model loaded.")

# Load pre-computed features and filenames
print("Loading embeddings and filenames...")
try:
    with open('embeddings.pkl', 'rb') as f:
        feature_list = pickle.load(f)
    with open('filenames.pkl', 'rb') as f:
        filenames = pickle.load(f)
    print("Embeddings and filenames loaded.")
except FileNotFoundError:
    print("Error: embeddings.pkl or filenames.pkl not found. Please ensure they exist.")
    feature_list = []
    filenames = []
    # Handle this more gracefully, perhaps exit or disable image features if critical
except Exception as e:
    print(f"Error loading pickle files: {e}")
    feature_list = []
    filenames = []


# Initialize Nearest Neighbors
if feature_list: # Only fit if data is available
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)
    print("NearestNeighbors model fitted.")
else:
    print("Warning: No feature list loaded, NearestNeighbors not fitted.")
    neighbors = None # Or handle this case where neighbor search won't work

# Load CSV for search function (once at startup)
print("Loading grandfinaleX.csv...")
try:
    df_products = pd.read_csv('./grandfinaleX.csv', on_bad_lines='skip')
    # Pre-process search columns once
    SEARCH_COLUMNS = ['gender', 'masterCategory', 'subCategory', 'articleType', 'productDisplayName']
    for col in SEARCH_COLUMNS:
        if col in df_products.columns:
            df_products[col] = df_products[col].astype(str).str.lower()
        else:
            print(f"Warning: Search column '{col}' not found in CSV.")
            # Remove from SEARCH_COLUMNS if not present
            SEARCH_COLUMNS.remove(col)
    print("CSV loaded and pre-processed for search.")
except FileNotFoundError:
    print("Error: grandfinaleX.csv not found. Search functionality will be limited.")
    df_products = pd.DataFrame() # Empty DataFrame if file not found
except Exception as e:
    print(f"Error loading grandfinaleX.csv: {e}")
    df_products = pd.DataFrame()


# Load Sentiment Analysis Pipeline (once at startup)
print("Loading sentiment analysis pipeline...")
try:
    sentiment_pipeline = pipeline("sentiment-analysis")
    print("Sentiment analysis pipeline loaded.")
except Exception as e:
    print(f"Error loading sentiment pipeline: {e}")
    sentiment_pipeline = None # Handle cases where pipeline might not be available


# --- Flask App Configuration ---
app = Flask(__name__)
CORS(app) # Enable CORS for all origins by default. Adjust `origins` if needed for production.
app.config['CORS_HEADERS'] = 'Content-Type'

# --- Utility Functions ---
def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result

# --- Routes ---

@app.route('/', methods=['GET'])
def helloworld():
    return jsonify({'response': "Hello World! Your API is running."})

@app.route('/recommend', methods=['GET'])
def recommend():
    # Use .get() for safer parameter access
    img_url = request.args.get('name')
    img_id = request.args.get('id')

    if not img_url or not img_id:
        return jsonify({'error': 'Missing "name" (image URL) or "id" parameters'}), 400

    if not feature_extractor_model or not neighbors or not filenames:
        return jsonify({'error': 'Image recommendation services are not initialized.'}), 503

    temp_img_name = f"{img_id}.jpg"
    temp_img_path = os.path.join(UPLOAD_FOLDER, temp_img_name)

    try:
        # Download image
        print(f"Downloading {img_url} to {temp_img_path}")
        urllib.request.urlretrieve(img_url, temp_img_path)

        # Extract features
        result_features = extract_features(temp_img_path, feature_extractor_model)

        # Find nearest neighbors
        distances, indices = neighbors.kneighbors([result_features])
        final_ids = []
        for i in indices[0]:
            if 0 <= i < len(filenames) and isinstance(filenames[i], str):
                try:
                    # Safely extract ID from filename (e.g., 'path/to/12345.jpg' -> 12345)
                    file_id_str = os.path.splitext(os.path.basename(filenames[i]))[0]
                    final_ids.append(int(file_id_str))
                except ValueError:
                    print(f"Warning: Could not convert filename ID to int: {filenames[i]}")
            else:
                print(f"Warning: Invalid index or non-string filename at index {i}: {filenames[i]}")

        return jsonify({'result': final_ids})

    except Exception as e:
        print(f"Error in /recommend: {e}")
        return jsonify({'error': f'An error occurred during recommendation: {str(e)}'}), 500
    finally:
        # Ensure temporary file is removed
        if os.path.exists(temp_img_path):
            os.remove(temp_img_path)
            print(f"Cleaned up {temp_img_path}")


@app.route('/image_search', methods=['GET'])
def imagesearch():
    img_url = request.args.get('url')

    if not img_url:
        return jsonify({'error': 'Missing "url" parameter'}), 400

    if not feature_extractor_model or not neighbors or not filenames:
        return jsonify({'error': 'Image search services are not initialized.'}), 503

    temp_img_name = "test_search.jpg" # Use a unique name if multiple concurrent requests are expected
    temp_img_path = os.path.join(UPLOAD_FOLDER, temp_img_name)

    try:
        print(f"Downloading {img_url} for search to {temp_img_path}")
        urllib.request.urlretrieve(img_url, temp_img_path)

        result_features = extract_features(temp_img_path, feature_extractor_model)
        distances, indices = neighbors.kneighbors([result_features])
        final_ids = []
        for i in indices[0]:
            if 0 <= i < len(filenames) and isinstance(filenames[i], str):
                try:
                    file_id_str = os.path.splitext(os.path.basename(filenames[i]))[0]
                    final_ids.append(int(file_id_str))
                except ValueError:
                    print(f"Warning: Could not convert filename ID to int: {filenames[i]}")
            else:
                print(f"Warning: Invalid index or non-string filename at index {i}: {filenames[i]}")

        return jsonify({'result': final_ids})

    except Exception as e:
        print(f"Error in /image_search: {e}")
        return jsonify({'error': f'An error occurred during image search: {str(e)}'}), 500
    finally:
        if os.path.exists(temp_img_path):
            os.remove(temp_img_path)
            print(f"Cleaned up {temp_img_path}")


@app.route('/search', methods=['GET'])
def search():
    print("Search API called")
    query = request.args.get('query')

    if not query:
        return jsonify({'error': 'Missing "query" parameter'}), 400
    
    if df_products.empty:
        return jsonify({'error': 'Product data for search is not loaded.'}), 503

    query_keywords = query.lower().split()
    
    # Create a copy to avoid SettingWithCopyWarning if not needed, or just work on a new column
    temp_df = df_products.copy()
    temp_df['match_score'] = 0

    # Count how many columns contain each keyword
    for word in query_keywords:
        match_mask = temp_df[SEARCH_COLUMNS].apply(lambda row: row.str.contains(word, na=False), axis=1).any(axis=1)
        temp_df.loc[match_mask, 'match_score'] += 1

    result_df = temp_df[temp_df['match_score'] > 0]

    if 'id' in result_df.columns:
        result_df = result_df.sort_values(by='match_score', ascending=False)
        # Ensure 'id' column is numeric for conversion to int
        top_ids = result_df['id'].dropna().astype(int).drop_duplicates().head(20).tolist()
    else:
        top_ids = []
        print("Warning: 'id' column not found in search results after filtering.")

    print("Matched IDs:", top_ids)
    return jsonify({'searchResult': top_ids})


@app.route('/sentiment', methods=['GET'])
def sentiment_analysis(): # Renamed function to avoid conflict with imported pipeline variable
    text = request.args.get('text') # Get text from query parameter
    if not text:
        return jsonify({'error': 'Missing "text" parameter for sentiment analysis'}), 400

    if not sentiment_pipeline:
        return jsonify({'error': 'Sentiment analysis service is not initialized.'}), 503

    try:
        result = sentiment_pipeline(text) # Pass the actual text
        print(f"Sentiment for '{text}': {result}")
        return jsonify({'sentiment_result': result})
    except Exception as e:
        print(f"Error during sentiment analysis: {e}")
        return jsonify({'error': f'An error occurred during sentiment analysis: {str(e)}'}), 500


if __name__ == '__main__':
    # In a production environment, set debug=False and use a production WSGI server
    # e.g., gunicorn -w 4 app:app
    app.run(debug=True, host='0.0.0.0') # host='0.0.0.0' to make it accessible from other devices on network