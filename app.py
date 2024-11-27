
from flask import Flask, render_template, request, jsonify
import os
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from PIL import Image
import sqlite3

app = Flask(__name__)

# Load the pre-trained ResNet50 model
model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

# Define the path to save uploaded images
UPLOAD_FOLDER = 'uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Make sure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Helper function to load vendor features
def get_vendor_features():
    vendor_features = {}
    conn = sqlite3.connect('vendor_images.db')
    cursor = conn.cursor()
    cursor.execute("SELECT image_name, image_path FROM vendor_images")
    for image_name, image_path in cursor.fetchall():
        try:
            img = Image.open(image_path).resize((224, 224))
            img_array = np.array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)
            features = model.predict(img_array)
            vendor_features[image_name] = features
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
    conn.close()
    print(f"Loaded vendor features: {vendor_features}")  # Debugging line
    # Debug: Check vendor features
    print(f"Vendor features (keys and sample values): {[(k, v.shape) for k, v in vendor_features.items()]}")
    # Debug: Print vendor feature dimensions
    print(f"Vendor feature dimensions for {image_name}: {features.shape}")

    return vendor_features


# Precompute vendor features at the start
vendor_features = get_vendor_features()

@app.route('/')
def index():
    return render_template('index.html')  # This assumes you have an index.html file

@app.route('/api/search', methods=['POST'])
def search_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        # Save the uploaded image
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(image_path)
        # Debug: Confirm the image path
        print(f"Uploaded image saved at: {image_path}") 
        # Debug: Print uploaded feature dimensions
        print(f"Uploaded features shape: {uploaded_features.shape}")

        # Preprocess the uploaded image
        img = Image.open(image_path).resize((224, 224))
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        
        # Extract features of the uploaded image
        uploaded_features = model.predict(img_array)
        
        # Debugging: Print uploaded image features
        print("Uploaded Image Features:", uploaded_features)
        
        # Compare with vendor features
        similarities = {}
        for vendor_name, vendor_feature in vendor_features.items():
            # Debugging: Print vendor image features
            print(f"Vendor Image Features for {vendor_name}:", vendor_feature)
            
            sim = cosine_similarity(uploaded_features, vendor_feature)
            print(f"Similarity with {vendor_name}: {sim[0][0]}")
            similarities[vendor_name] = sim[0][0]
        
        # If no similarities found, return error
        if not similarities:
            print("No similarities found.") 
            return jsonify({'error': 'No similar images found'})
        
        # Debugging: Print similarities
        print("Similarities:", similarities)
        
        # Find the most similar image
        most_similar = max(similarities, key=similarities.get)
        highest_similarity = similarities[most_similar]
        print(f"Most similar image: {most_similar} with similarity {highest_similarity}")
        return jsonify({
            'message': 'Image uploaded successfully',
            'most_similar_image': most_similar,
            'similarity_score': highest_similarity
        })

# Debug: Check contents of the database
conn = sqlite3.connect('vendor_images.db')
cursor = conn.cursor()
cursor.execute("SELECT * FROM vendor_images")
rows = cursor.fetchall()
print(f"Database contents: {rows}")  # Add this line
conn.close()


if __name__ == '__main__':
    app.run(debug=True)
