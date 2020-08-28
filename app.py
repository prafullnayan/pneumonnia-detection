from __future__ import division, print_function
import sys
import os
import glob
import re
import numpy as np

from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask, redirect, url_for, request, render_template

# Define a flask app
app = Flask(__name__)


model = load_model("model2.h5")         # Necessary



def model_predict(img_path,model):
    img = image.load_img(img_path, target_size=(150, 150))

    # Preprocessing the image
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    x = x / 255.0 

    preds = model.predict(x)
    #prediction = preds > 0.5
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

@app.route('/base.html',methods=['GET'])
def base():
    return render_template('base.html')

@app.route('/index.html',methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)

        # Process your result for human
        # pred_class = preds.argmax(axis=-1)            # Simple argmax
          # ImageNet Decode
        if(preds >0.5):
            answer = "Pneumonia"
        else:
            answer = "Normal"
        result = answer               # Convert to string
        return result
    return None


if __name__ == '__main__':
    app.run(debug=True)
