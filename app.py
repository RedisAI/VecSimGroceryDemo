from flask import Flask, request, send_file

from img2vec_pytorch import Img2Vec
from PIL import Image
from redis import Redis

from redis.commands.search.query import Query
import time

import torch
from torchvision import transforms

from pathlib import Path

import torch

FILE = Path(__file__).resolve()

from yolov5.utils.general import non_max_suppression
from yolov5.models.experimental import attempt_load


def load_models_to_redis(redis, detection_model, img2vec):
    # Convert the encoding model from torch to onnx and store it in RedisAI
    model_with_pruned_last_layer = img2vec.model
    model_with_pruned_last_layer.fc = torch.nn.Identity()
    model_with_pruned_last_layer.eval()
    dummy_input = torch.randn(1, 3, 224, 224)
    input_names = ['image']
    output_name = ['embedding']
    encoding_model_file_name = "resnet-18.onnx"
    torch.onnx.export(model_with_pruned_last_layer, dummy_input, encoding_model_file_name, verbose=False,
                      input_names=input_names, output_names=output_name, dynamic_axes={'image': [0]})
    with open(encoding_model_file_name, 'rb') as f:
        model_blob = f.read()
        redis.execute_command('AI.MODELSTORE', 'encoding_model', 'ONNX', 'CPU', 'BLOB', model_blob)

    # Do the same with the other detection model.
    detection_model_file_name = "retina-net.onnx"
    input_names = ['image']
    output_name = ['bounding_boxes']
    torch.onnx.export(detection_model, dummy_input, detection_model_file_name, verbose=False,
                      input_names=input_names, output_names=output_name, dynamic_axes={'image': [2, 3]})

    with open(detection_model_file_name, 'rb') as f:
        model_blob = f.read()
        redis.execute_command('AI.MODELSTORE', 'detection_model', 'ONNX', 'CPU', 'BLOB', model_blob)


app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

redis = Redis()
img2vec = Img2Vec()

weights = 'best.pt'
g_threshold = 0.4
g_max_det = 10
g_device = torch.device('cpu')
g_model = attempt_load(weights, device=g_device)
load_models_to_redis(redis, g_model, img2vec)  # Use RedisAI

transform = transforms.Compose([
    transforms.ToTensor(),
])


# Get the model
def get_boxes(image, model, threshold, max_det):
    img = transform(image).to(g_device)[None]
    predictions = model(img)
    detections = non_max_suppression(predictions, conf_thres=threshold, max_det=max_det)
    return detections[0][:, :4].tolist()

#box_points = [xmin,ymin, xmax,ymax]
def search_product(image, box_points):
    product = image.crop(box_points)
    query_vector = img2vec.get_vec(product)

    res, search_time = getTopK(query_vector, 4)
    print('KNN search time\t', search_time)
    return res

@app.route('/')
def index():
    return send_file('index.html')

@app.route('/search', methods=['POST'])
def search():
    if 'image' not in request.files:
        return 'no image found', 400

    file = request.files['image']
    image = Image.open(file)

    start = time.time()
    boxes = get_boxes(image, g_model, g_threshold, g_max_det)
    print('boxing time\t', time.time() - start)

    return {
        'results': [
            {
                'box': box,
                'products': search_product(image, box)
            } for box in boxes
        ]
    }

def getTopK(query_vector, k = 10, filter = '*'):
    q = Query(f'({filter})=>[KNN {k} @vectors $vec_param AS distance]').sort_by('distance').paging(0, k)\
        .return_fields('id', 'brand', 'name', 'family', 'distance').return_field('$.images[0].url', 'image')
    start = time.time()
    res = redis.ft().search(q, query_params={'vec_param': query_vector.tobytes()})
    search_time = time.time() - start

    return [doc.__dict__ for doc in res.docs], search_time
