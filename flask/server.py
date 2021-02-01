import cv2
import os
import numpy as np
import tensorflow as tf
from flask_cors import CORS
from tensorflow.keras.models import load_model
from flask import Flask, request, render_template, make_response, jsonify

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.InteractiveSession(config=config)

app = Flask(__name__)
CORS(app)

HOST = "0.0.0.0" 
PORT_NUMBER = int(os.getenv('PORT_NUMBER', 8080)) 
APP_ROOT_1 = os.getenv('APP_ROOT', '/infer1')
APP_ROOT_2 = os.getenv('APP_ROOT', '/infer2')
model = load_model('animal_model_classification.h5')
image_width = 300
image_height = 300
classes = ['cat', 'dog', 'pandas']

# render default webpage
@app.route('/')
def home():
    return render_template('home.html')

@app.route(APP_ROOT_1, methods = ['POST', 'GET'])
def classify_image():
    if request.method == 'POST':
    # geting data from html form
        img_name = request.form["file"]
        # call funtion to classify image and receive result
        result = classify_animal(img_name)
        # return result to client
        response = {'result': result, 'image': img_name}        
        return make_response(jsonify(response), 200)
        
@app.route(APP_ROOT_2, methods=["POST"]) 
def infer(): 
	data = request.json 
	img_path = data['img_path'] 
	return classify_animal(img_path)
        
def classify_animal(img_path):
    # read image
    image = cv2.imread(img_path)
    image = image/255
    image = cv2.resize(image, (image_width,image_height))   
    image = np.reshape(image, [1,image_width,image_height,3])

    # pass the image through the network to obtain our predictions
    preds = model.predict(image)
    label = classes[np.argmax(preds)]

    return label
  
if __name__ == '__main__':
    app.run(host=HOST, port=PORT_NUMBER)
