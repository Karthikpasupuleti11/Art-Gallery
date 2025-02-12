import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify
import os
import base64

# ---------------------- Database Connection ----------------------

from pymongo import MongoClient

# MongoDB connection string
new_mongo_client = MongoClient("mongodb+srv://Karthik:karthik@cluster0.0fbrm.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
new_db = new_mongo_client.get_database("Cluster0")  # Replace with your database name if needed
app = Flask(__name__)

# ---------------------- Helper Functions ----------------------
def preprocess_image(image_path):
    """Read and preprocess an image."""
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (224, 224))  # Resize to a fixed size
    return image

def extract_color_histogram(image):
    """Extract color histogram features."""
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)  # Normalize the histogram
    return hist.flatten()

def find_most_similar_image(uploaded_image_data, dataset_image_paths):
    """Find the most similar image based on color histograms."""
    # Preprocess the uploaded image
    np_array = np.frombuffer(uploaded_image_data, np.uint8)
    uploaded_image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    uploaded_image = cv2.resize(uploaded_image, (224, 224))
    uploaded_hist = extract_color_histogram(uploaded_image)

    # Compare against dataset images
    similarities = []
    for path in dataset_image_paths:
        dataset_image = preprocess_image(path)
        dataset_hist = extract_color_histogram(dataset_image)

        # Compute similarity (Cosine Similarity)
        similarity = np.dot(uploaded_hist, dataset_hist) / (
            np.linalg.norm(uploaded_hist) * np.linalg.norm(dataset_hist)
        )
        similarities.append(similarity)

    most_similar_index = np.argmax(similarities)
    return dataset_image_paths[most_similar_index], similarities[most_similar_index]

def image_to_base64(image_path):
    """Convert image to Base64 for embedding in HTML."""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

# ---------------------- Routes ----------------------

@app.route("/test_db")
def test_db():
    try:
        new_db.command("ping")  # This will ping the database to check the connection
        return "MongoDB connection successful!"
    except Exception as e:
        return f"Error connecting to MongoDB: {str(e)}"

@app.route("/")
def signup():
    """Render the Sign-Up Page."""
    return render_template("signup.html")

@app.route("/3d")
def Threed():
    """Render the Sign-Up Page."""
    return render_template("3d.html")

@app.route("/home")
def home():
    """Render the Home Page."""
    return render_template("home.html")

@app.route('/second')
def second():
    return render_template('second.html')

@app.route("/index", methods=["GET", "POST"])
def upload_image():
    """Handle image upload and similarity search."""
    if request.method == "POST":
        uploaded_file = request.files.get("file")
        if uploaded_file:
            uploaded_image_data = uploaded_file.read()
            
            BASE_DIR = os.path.dirname(os.path.abspath(__file__))

            # Build the path to the images folder
            dataset_image_dir = os.path.join(BASE_DIR, 'static', 'images')
            dataset_image_paths = [
                os.path.join(dataset_image_dir, img)
                for img in os.listdir(dataset_image_dir)
                if os.path.isfile(os.path.join(dataset_image_dir, img))
            ]
            
            # Find the most similar image
            most_similar_image_path, similarity_score = find_most_similar_image(
                uploaded_image_data, dataset_image_paths
            )
            
            # Check similarity threshold
            similarity_threshold = 0.5  # Cosine similarity threshold (0 to 1)
            if similarity_score < similarity_threshold:
                message = f"Oops! No similar image found. Similarity score: {similarity_score:.2f}"
                return jsonify({
                    "below_threshold": True,
                    "similarity_score": float(similarity_score)
                })

            # Convert the matched image to Base64
            similar_image_base64 = image_to_base64(most_similar_image_path)
            return jsonify({
                "most_similar_image": f"data:image/jpeg;base64,{similar_image_base64}",
                "similarity_score": float(similarity_score)
            })

    return render_template("index.html")

@app.route("/clean")
def clean_interface():
    """Render the Clean Interface Page."""
    return render_template("clean_interface.html")

if __name__ == "__main__":
    app.run(debug=True)
