from img2vec_pytorch import Img2Vec
from PIL import Image
from redis import Redis
import numpy as np
from redis.commands.search.field import VectorField, TagField, NumericField
from redis.commands.search.query import Query
import os
import time

host = "localhost"
port = 6379


def create_index_in_redis(dim, vector_field_name):
    redis = Redis(host=host, port=port)
    redis.flushall()
    schema = (VectorField(vector_field_name, "FLAT", {"TYPE": "FLOAT32", "DIM": dim, "DISTANCE_METRIC": "COSINE"}))
    redis.ft().create_index(schema)
    redis.ft().config_set("default_dialect", 2)


def load_data(client, img2vec, vector_field_name):
    count = 0
    total_time = 0
    first_id = 3738649
    last_id = 5478851
    for i in range(first_id, last_id+1):
        # Read in an image (rgb format)
        for j in range(4):
            fname = f'data/{i}-{j}.jpg'
            if not os.path.isfile(fname):
                continue
            try:
                img = Image.open(fname)
                start = time.time()
                vec = img2vec.get_vec(img)
                total_time += time.time() - start
                client.hset(f"{i}-{j}", mapping={vector_field_name: vec.tobytes()})
                count += 1
            except RuntimeError as e:
                print(f"Runtime error {e} for {fname}")

    print("Index size: ", client.ft().info()['num_docs'])
    print(f"Avg inference time is: {total_time/count}")


def main():
    dim = 512
    vector_field_name = "vector"
    client = Redis(host=host, port=port)
    create_index_in_redis(dim, vector_field_name)

    # Initialize Img2Vec
    img2vec = Img2Vec()

    # Load images data, convert it to vector embeddings and store it in Redis.
    load_data(client, img2vec, vector_field_name)

    # Run KNN for sime image
    fname = f'data/5166398-0.jpg'
    img = Image.open(fname)
    query_vector = img2vec.get_vec(img)
    start = time.time()
    q = Query(f'*=>[KNN 10 @{vector_field_name} $vec_param]=>{{$yield_distance_as: dist}}').sort_by(f'dist')
    res = client.ft().search(q, query_params={'vec_param': query_vector.tobytes()})
    end = time.time()

    print(f"search took {end-start} seconds")

    docs = [doc.id for doc in res.docs]
    print(docs)
    dists = [float(doc.dist) if hasattr(doc, 'dist') else '-' for doc in res.docs]
    print(dists)


if __name__ == '__main__':
    main()
