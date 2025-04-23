from flask import Flask, render_template, request, jsonify
import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename

app = Flask(__name__)
model = load_model("model.h5")

UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Class names (Edit as per your dataset labels)
class_names = [
    "Speed limit (20km/h)", "Speed limit (30km/h)", "Speed limit (50km/h)", "Speed limit (60km/h)",
    "Speed limit (70km/h)", "Speed limit (80km/h)", "End of speed limit (80km/h)",
    "Speed limit (100km/h)", "Speed limit (120km/h)", "No passing", "No passing for vehicles over 3.5 metric tons",
    "Right-of-way at the next intersection", "Priority road", "Yield", "Stop",
    "No vehicles", "Vehicles over 3.5 metric tons prohibited", "No entry", "General caution",
    "Dangerous curve to the left", "Dangerous curve to the right", "Double curve",
    "Bumpy road", "Slippery road", "Road narrows on the right", "Road work", "Traffic signals",
    "Pedestrians", "Children crossing", "Bicycles crossing", "Beware of ice/snow",
    "Wild animals crossing", "End of all speed and passing limits", "Turn right ahead",
    "Turn left ahead", "Ahead only", "Go straight or right", "Go straight or left",
    "Keep right", "Keep left", "Roundabout mandatory", "End of no passing",
    "End of no passing by vehicles over 3.5 metric tons"
]

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (32, 32))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255.0
    img = img.reshape(1, 32, 32, 1)
    return img

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    img = preprocess_image(filepath)
    prediction = model.predict(img)
    class_index = np.argmax(prediction)
    confidence = float(np.max(prediction)) * 100
    label = class_names[class_index]

    return jsonify({
        'label': label,
        'confidence': f"{confidence:.2f}%",
        'filepath': filepath
    })

if __name__ == '__main__':
    app.run(debug=True, port=5001)
