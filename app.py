from flask import Flask, request, send_file
from img2vec_pytorch import Img2Vec
from PIL import Image
from redis import Redis
import numpy as np
from redis.commands.json.path import Path
from redis.commands.search.field import VectorField, TagField, NumericField
from redis.commands.search.query import Query
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
import os
import time
import json

app = Flask(__name__)
redis = Redis()
img2vec = Img2Vec()
vector_field_name = "vectors"

@app.route('/')
def index():
    return send_file('index.html')

@app.route('/search', methods=['POST'])
def search():
    if 'image' not in request.files:
        return 'no image found', 400

    file = request.files['image']
    image = Image.open(file)
    query_vector = img2vec.get_vec(image)

    res, total_time = getTopK(query_vector, 4)

    print(total_time)

    return res

def getTopK(query_vector, k = 10, filter = '*'):
    q = Query(f'({filter})=>[KNN {k} @{vector_field_name} $vec_param]=>{{$yield_distance_as: dist}}').sort_by(f'dist').paging(0, k)
    start = time.time()
    res = redis.ft().search(q, query_params={'vec_param': query_vector.tobytes()})
    search_time = time.time() - start

    products = [json.loads(doc.json) for doc in res.docs]
    for prod, doc in zip(products, res.docs):
        prod['distance'] = doc.dist

    return {'products': products}, search_time
