from flask import Flask, request, send_from_directory, jsonify
from flask_cors import CORS  # Import the CORS module
import os
import threading
import signal
import sys
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

app.config['UPLOAD_FOLDER'] = 'static/uploads'
model = tf.keras.models.load_model("tea_dust_model_V3.h5")

class_names = ["Class1", "Class2", "Class3", "Class4", "Class5"]

def process_image(image_path):
    img = load_img(image_path, target_size=(512, 512))
    img_array = img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    return img_array

def predict_class(img_array):
    predictions = model.predict(img_array)
    predicted_class_index = tf.argmax(predictions[0]).numpy()
    return class_names[predicted_class_index]

def create_upload_folder():
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])

def shutdown_server():
    print("Shutting down the application...")
    os.kill(os.getpid(), signal.SIGINT)

def shutdown():
    try:
        thread = threading.Thread(target=shutdown_server)
        thread.start()
    except SystemExit:
        sys.exit()

@app.route('/', methods=['POST'])
def index():
    predicted_class = None
    uploaded_image = None

    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        create_upload_folder()

        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)

        img_array = process_image(filename)
        predicted_class = predict_class(img_array)
        uploaded_image = file.filename

    return jsonify(predicted_class = predicted_class, uploaded_image  =uploaded_image)

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

@app.route('/shutdown', methods=['POST'])
def shutdown_request():
    shutdown()
    return 'Server shutting down...'

if __name__ == '__main__':
    app.run(debug=True)

