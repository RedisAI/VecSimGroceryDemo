import argparse
import os
import time
from urllib.parse import urlparse

from redis import Redis, exceptions
import torch
from img2vec_pytorch import Img2Vec
from yolov5.models.experimental import attempt_load


# Busy wait for Redis until it finishes loading the RDB in memory upon loading
def wait_for_redis(redis):
    while True:
        try:
            redis.ping()
            return
        except exceptions.BusyLoadingError:
            time.sleep(5)


def load_models_to_redis(redis, detection_model, encoding_model, tag='v0'):
    # Override the last fully-connected layer and evaluate the updated model.
    encoding_model.fc = torch.nn.Identity()
    encoding_model.eval()

    # Create a dummy input which is required to trace the model and export it to onnx.
    dummy_input = torch.randn(1, 3, 224, 224)
    input_names = ['image']
    output_name = ['embedding']
    encoding_model_file_name = "resnet-18-encoder.onnx"
    torch.onnx.export(encoding_model, dummy_input, encoding_model_file_name, input_names=input_names,
                      output_names=output_name, dynamic_axes={'image': [0]})  # first axe is dynamic (batch size)
    # Save the model in RedisAI (https://oss.redis.com/redisai/commands/#aimodelstore)
    with open(encoding_model_file_name, 'rb') as f:
        model_blob = f.read()
        redis.execute_command('AI.MODELSTORE', 'encoding_model', 'ONNX', 'CPU', 'TAG', tag, 'BLOB', model_blob)

    # Trace, export and store the detection model as well. Note that
    detection_model_file_name = "retina-net.onnx"
    input_names = ['image']
    output_name = ['bounding_boxes']
    # Image input size is variable, hence axes 2 and 3 are dynamic as well.
    torch.onnx.export(detection_model, dummy_input, detection_model_file_name, do_constant_folding=True,
                      input_names=input_names, output_names=output_name, dynamic_axes={'image': [0, 2, 3]})
    with open(detection_model_file_name, 'rb') as f:
        model_blob = f.read()
        redis.execute_command('AI.MODELSTORE', 'detection_model', 'ONNX', 'CPU', 'TAG', tag, 'BLOB', model_blob)


def main():
    print("Loading pretrained detection and encoding models into Redis (using RedisAI)...")

    # Connect to redis-server, wait for it to be ready to accept connections.
    parser = argparse.ArgumentParser()
    parser.add_argument('-rs', '--redis_server', help='Redis URL', type=str, default='redis://127.0.0.1:6379')
    args = parser.parse_args()
    url = urlparse(args.redis_server)
    redis = Redis(host=url.hostname, port=url.port, decode_responses=True)
    wait_for_redis(redis)

    # Use a pretrained object detection model based on yolov5 (https://github.com/ultralytics/yolov5),
    # after performing fine-tuning ('best.pt' are the model weights after fine-tuning).
    weights = os.path.dirname(os.path.abspath(__file__)) + '/models/best.pt'
    device = torch.device('cpu')  # can change to 'cuda'
    detection_model = attempt_load(weights, device=device)

    # Use resnet-18 pretrained model as the encoding model, whose second last layer's output will be the embedding
    encoding_model = Img2Vec(model='resnet-18', layer='default').model

    load_models_to_redis(redis, detection_model, encoding_model)  # Use RedisAI
    print("Done\n")

    # Load gear that runs the flow in RedisGears
    print("Loading application flow recipe into RedisGears (using internal Python interpreter)...")
    gear_path = os.path.dirname(os.path.abspath(__file__)) + '/../app_redisai/old_gear.py'
    with open(gear_path) as f:
        redis.execute_command("RG.PYEXECUTE", f.read(), 'ID', 'flow', 'UPGRADE', 'REQUIREMENTS',
                              "torchvision")
    print("Done\n")


if __name__ == '__main__':
    main()
