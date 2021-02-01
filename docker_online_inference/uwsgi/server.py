import cv2
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from flask import Flask, request, render_template, make_response, jsonify

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.InteractiveSession(config=config)

app = Flask(__name__)
APP_ROOT = os.getenv('APP_ROOT', '/infer')
HOST = "0.0.0.0"
PORT_NUMBER = int(os.getenv('PORT_NUMBER', 8080))

image_width = 300
image_height = 300
classes = ['cat', 'dog', 'pandas']
model = load_model('animal_model_classification.h5')

        
@app.route(APP_ROOT, methods=["POST"]) 
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
