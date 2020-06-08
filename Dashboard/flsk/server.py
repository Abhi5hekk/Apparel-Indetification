import json
from flask import Flask,jsonify,request
from flask_cors import CORS

import numpy as np
from PIL import Image
import requests
from io import BytesIO

from tagger import tag

app = Flask(__name__)
CORS(app)

@app.route("/tag",methods=['GET'])
def return_price():
  img = request.args.get('image_url')

  if img is None:
    return jsonify({"error": "no image provided"})
  
  response = requests.get(img)
  pil_img = Image.open(BytesIO(response.content))
  pil_img = pil_img.convert("RGB")

  return jsonify(tag(pil_img))

@app.route("/",methods=['GET'])
def default():
  return "API AVAILABLE"

if __name__ == "__main__":
    app.run("0.0.0.0", 5000)