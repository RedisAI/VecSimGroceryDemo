from flask import Flask, request, send_file

from img2vec_pytorch import Img2Vec
from PIL import Image
from redis import Redis

from redis.commands.search.query import Query
import time

import torch
from torchvision import transforms

import os
import platform
import sys
from pathlib import Path

import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT) + "/deps")  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
import yolov5.models

#from deps.yolov5.models.experimental import attempt_download, attempt_load  # scoped to avoid circular import
#from deps.yolov5.utils.downloads import attempt_download

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

redis = Redis()
img2vec = Img2Vec()



weights = 'best.pt'
threshold = 0.4
max_det = 1000
g_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


transform = transforms.Compose([
    transforms.ToTensor(),
])

#dnn and fp16 false, we dont use data
def get_detection_model(weights, device=torch.device('cpu'), fp16=True, fuse=True):

    from deps.yolov5.models.experimental import attempt_load

    w = str(weights)
    model = attempt_load(w, device=device, inplace=True, fuse=fuse)
    stride = max(int(model.stride.max()), 32)  # model stride
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    model.half() # fp16
    return model

# Get the model
def get_boxes(image, model, threshold):
    
    model = get_detection_model(w, device = g_device)

    im = transform(image).to(device)
    
    pred = model(im)

    # Remove overlapping squares
    pred = non_max_suppression(pred, conf_thres=threshold,  max_det=max_det)

    boxes = List()
    #[x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    for det in pred:
        boxes.append(det[:, :4])

    return boxes.astype(np.int32)


g_model = get_detection_model(weights)

#box_points = [xmin,ymin, xmax,ymax]
def search_product(i, image, box_points):
    product = image.crop(box_points)
    product.save(f'meow_{i}.jpg')
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
    boxes = get_boxes(image, g_model, g_device)
    print('boxing time\t', time.time() - start)

    return {
        'results': [
            {
                'box': box.tolist(),
                'products': search_product(i, image, box)
            } for i, box in enumerate(boxes)
        ]
    }

def getTopK(query_vector, k = 10, filter = '*'):
    q = Query(f'({filter})=>[KNN {k} @vectors $vec_param AS distance]').sort_by('distance').paging(0, k)\
        .return_fields('id', 'brand', 'name', 'family', 'distance').return_field('$.images[0].url', 'image')
    start = time.time()
    res = redis.ft().search(q, query_params={'vec_param': query_vector.tobytes()})
    search_time = time.time() - start

    return [doc.__dict__ for doc in res.docs], search_time
