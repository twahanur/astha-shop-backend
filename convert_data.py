# convert_data.py
import pickle
import json

print("Starting conversion of .pkl files to .json...")

# Convert embeddings.pkl
try:
    with open('embeddings.pkl', 'rb') as f_pkl:
        embeddings_data = pickle.load(f_pkl)
    with open('embeddings.json', 'w') as f_json:
        json.dump(embeddings_data, f_json)
    print("✅ embeddings.pkl successfully converted to embeddings.json")
except FileNotFoundError:
    print("Error: embeddings.pkl not found.")
except Exception as e:
    print(f"An error occurred: {e}")

# Convert filenames.pkl
try:
    with open('filenames.pkl', 'rb') as f_pkl:
        filenames_data = pickle.load(f_pkl)
    with open('filenames.json', 'w') as f_json:
        json.dump(filenames_data, f_json)
    print("✅ filenames.pkl successfully converted to filenames.json")
except FileNotFoundError:
    print("Error: filenames.pkl not found.")
except Exception as e:
    print(f"An error occurred: {e}")

print("\nConversion complete.")