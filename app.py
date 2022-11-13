from flask import Flask, request, send_file

from redis import Redis
from redis.commands.search.query import Query

import torch
from torchvision import transforms
from img2vec_pytorch import Img2Vec
from PIL import Image
from yolov5.utils.general import non_max_suppression
from yolov5.models.experimental import attempt_load

from pathlib import Path
import time

FILE = Path(__file__).resolve()

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

redis = Redis()
img2vec = Img2Vec()

weights = 'best.pt'
g_conf_threshold = 0.2
g_overlap_threshold = 0.7
g_max_det = 10
g_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
g_model = attempt_load(weights, device=g_device)

transform = transforms.Compose([
    transforms.ToTensor(),
])

# Get the model
def get_boxes(image, model, conf_threshold, overlap_threshold, max_det):
    # Prepare image
    img = transform(image).to(g_device)[None]
    # Run model
    predictions = model(img)
    # Remove overlapping (overlap_threshold), low confidence (conf_threshold) and worst (max_det) squares.
    detections = non_max_suppression(predictions, conf_thres=conf_threshold, max_det=max_det, iou_thres=overlap_threshold)
    # Return list of coordinations of the boxes
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
    boxes = get_boxes(image, g_model, g_conf_threshold, g_overlap_threshold, g_max_det)
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
