import cv2
import numpy as np
import pymongo
import base64
import os
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# ---------------------- MongoDB Connection ----------------------
MONGO_URI = "mongodb+srv://Karthik:karthik@cluster0.0fbrm.mongodb.net/"

# Connect to MongoDB Atlas
client = pymongo.MongoClient(MONGO_URI)
db = client["art_gallery"]
art_collection = db["artworks"]

print("Connected to MongoDB Atlas!")

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
    np_array = np.frombuffer(uploaded_image_data, np.uint8)
    uploaded_image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    uploaded_image = cv2.resize(uploaded_image, (224, 224))
    uploaded_hist = extract_color_histogram(uploaded_image)

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

@app.route("/")
def signup():
    """Render the Sign-Up Page."""
    return render_template("signup.html")

@app.route("/3d")
def Threed():
    """Render the 3D Page."""
    return render_template("3d.html")

@app.route("/home")
def home():
    """Render the Home Page."""
    return render_template("home.html")

@app.route('/second')
def second():
    return render_template('second.html')

@app.route("/upload", methods=["GET", "POST"])
def upload_image():
    """Handle image upload, similarity search, and store metadata in MongoDB."""
    if request.method == "POST":
        uploaded_file = request.files.get("file")
        if uploaded_file:
            uploaded_image_data = uploaded_file.read()
            
            BASE_DIR = os.path.dirname(os.path.abspath(__file__))
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

            # Store uploaded image metadata in MongoDB
            uploaded_image_base64 = base64.b64encode(uploaded_image_data).decode('utf-8')
            image_metadata = {
                "filename": uploaded_file.filename,
                "image_data": uploaded_image_base64,
                "similarity_score": similarity_score
            }
            art_collection.insert_one(image_metadata)
            print("Image metadata stored in MongoDB!")

            # Convert the matched image to Base64
            similar_image_base64 = image_to_base64(most_similar_image_path)

            return jsonify({
                "most_similar_image": f"data:image/jpeg;base64,{similar_image_base64}",
                "similarity_score": float(similarity_score)
            })

    return render_template("index.html")

@app.route("/get_images", methods=["GET"])
def get_images():
    """Fetch stored artworks from MongoDB."""
    artworks = list(art_collection.find({}, {"_id": 0}))  # Exclude MongoDB ID
    return jsonify({"artworks": artworks})

if __name__ == "__main__":
    app.run(debug=True)
