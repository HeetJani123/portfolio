from flask import Flask, request, redirect, url_for, send_file, render_template
from fer import FER
import cv2
import os
import tempfile

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect-emotion', methods=['POST'])
def detect_emotion():
    if 'image' not in request.files:
        return redirect(url_for('index'))
    
    image_file = request.files['image']
    if image_file.filename == '':
        return redirect(url_for('index'))
    
    if image_file:
        # Save the image file temporarily
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
        image_path = temp_file.name
        image_file.save(image_path)

        # Load the image and perform emotion detection
        image = cv2.imread(image_path)
        detector = FER()
        result = detector.detect_emotions(image)

        # Determine the most likely emotion
        if result:
            (bounding_box, emotions) = result[0]
            emotion, score = detector.top_emotion(image)
            message = f"Detected Emotion: {emotion} with confidence {score:.2f}"
        else:
            message = "No face detected."

        # Clean up the temporary file
        os.remove(image_path)

        return message

if __name__ == "__main__":
    app.run(debug=True)
