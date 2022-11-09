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

host = "localhost"
port = 6379
prod_prefix = 'product:'

def create_index_in_redis(dim, vector_field_name):
    redis = Redis(host=host, port=port)
    redis.flushall()
    schema = (
        VectorField(f'$.{vector_field_name}[*]', "FLAT", {"TYPE": "FLOAT32", "DIM": dim, "DISTANCE_METRIC": "COSINE"}, as_name=vector_field_name),
        TagField('$.brand', as_name='brand'),
        NumericField('$.id', as_name='id'),
        TagField("$.names.short", as_name='name'),
        TagField("$.names.long", as_name='long_name'),
        TagField("$.family.name", as_name='family'),
        )
    redis.ft().create_index(schema, definition=IndexDefinition(prefix=[prod_prefix], index_type=IndexType.JSON))
    redis.ft().config_set("default_dialect", 2)


def load_data(client, img2vec, vector_field_name):
    count = 0
    total_time = 0

    map = {}
    for file in os.listdir('out'):
        split = file.split('.')

        id = split[0].split('-')[0]
        ext = split[-1]
        
        if (not id in map):
            map[id] = {
                'json': None,
                'images': []
            }

        if (ext == 'json'):
            map[id]['json'] = file
        else:
            map[id]['images'].append(file)
    
    for id in map:
        # Read in an image (rgb format

        if (map[id]['json'] == None):
            print(id)
            continue


        try:
            imgs = [Image.open(f'out/{img}') for img in map[id]['images']]
            start = time.time()
            vectors = img2vec.get_vec(imgs)
            total_time += time.time() - start
            # try:
            with open(f'out/{map[id]["json"]}', 'r') as jfile: json_doc = json.load(jfile)
            # except FileNotFoundError:
            #     json_doc = {"id": i, "names": {"short": "not Found", "long": "File Not Found"}}
            json_doc[vector_field_name] = [vec.tolist() for vec in vectors]
            client.json().set(f"{prod_prefix}{id}", Path.root_path(), json_doc)
            count += 1
        except Exception as e:
            print(f"Runtime error {e} for {id}")
        finally:
            [img.close() for img in imgs]

    info = client.ft().info()
    print("Index size: ", info['num_docs'])
    print("Number of Records: ", info['num_records'])
    print(f"Avg inference time is: {total_time/count}")


def main():
    dim = 512
    vector_field_name = "vectors"
    client = Redis(host=host, port=port)
    create_index_in_redis(dim, vector_field_name)

    # Initialize Img2Vec
    img2vec = Img2Vec()

    # Load images data, convert it to vector embeddings and store it in Redis.
    load_data(client, img2vec, vector_field_name)

    # Run KNN for some image
    for fname in [img for img in os.listdir() if img.split('.')[-1] == 'jpg']:
        print(fname)
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
