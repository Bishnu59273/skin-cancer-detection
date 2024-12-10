import os
import numpy as np
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Ensure the upload directory exists
UPLOAD_FOLDER = 'static/uploaded_images'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load pre-trained EfficientNetB0 model with ImageNet weights
base_model = EfficientNetB0(
    weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

# Custom model for multi-class classification
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(1024, activation='relu'),
    Dropout(0.2),
    Dense(4, activation='softmax')  # Multi-class classification
])

# Define class names
class_names = ['Melanoma', 'Basal Cell Carcinoma',
               'Squamous Cell Carcinoma', 'Benign']

# Set allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

# Function to check allowed file extensions


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Route for the homepage


@app.route('/')
def index():
    return render_template('index.html')

# Route for image upload and prediction


@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image file found"}), 400

    image_file = request.files['image']

    if image_file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if image_file and allowed_file(image_file.filename):
        # Secure the file name
        filename = secure_filename(image_file.filename)

        # Save the image to the server
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image_file.save(image_path)

        try:
            # Prepare the image for prediction
            img = image.load_img(
                image_path, target_size=(224, 224))  # Resize image
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(
                img_array, axis=0)  # Add batch dimension
            img_array = preprocess_input(img_array)

            # Make the prediction
            prediction = model.predict(img_array)

            # Get index of highest probability
            predicted_class_idx = np.argmax(prediction, axis=1)
            # Get the class name
            predicted_class = class_names[predicted_class_idx[0]]

            # Confidence as the probability of the predicted class
            # Convert the max probability to percentage
            confidence = float(np.max(prediction)) * 100

            # Send back the result
            response = {
                "prediction": predicted_class,
                "confidence": f"{confidence:.2f}%",
                "image_url": image_path
            }
            return jsonify(response)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return jsonify({"error": "Invalid file type"}), 400


if __name__ == '__main__':
    app.run(debug=True)
