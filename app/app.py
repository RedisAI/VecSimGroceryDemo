import base64
import io
import json
import time

from flask import Flask, request, send_file
from img2vec_pytorch import Img2Vec
from PIL import Image
from redis import Redis
from pathlib import Path

FILE = Path(__file__).resolve()

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

redis = Redis(decode_responses=True)
img2vec = Img2Vec()

g_threshold = 0.4
g_max_detections = 10


@app.route('/')
def index():
    return send_file('index.html')


@app.route('/search', methods=['POST'])
def search():
    if 'image' not in request.files:
        return 'no image found', 400
    start = time.time()
    file = request.files['image']
    image = Image.open(file)
    # BytesIO is a fake file stored in memory
    img_byte_arr = io.BytesIO()
    # image.save expects a file as an argument, passing a bytes io ins
    image.save(img_byte_arr, format=image.format)
    # Turn the BytesIO object back into a bytes object (use base 64)
    img_base_64 = base64.encodebytes(img_byte_arr.getvalue())

    # Run the entire flow in RedisGears:
    # 1. detect every object in the image with the detection model in RedisAI.
    # 2. for every detected object:
    # 2.1 crop the image and run the encoding model to get the relevant embedding
    # 2.2 search for the top 4 similar products stored in the database, and return them
    res = redis.execute_command("RG.TRIGGER", 'RunSearchFlow', img_base_64, g_threshold, g_max_detections)[0]
    print(f"Total flow took: {time.time()-start}")
    return json.loads(res)
