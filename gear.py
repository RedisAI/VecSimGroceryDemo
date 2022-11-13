import redisAI
from yolov5.utils.general import non_max_suppression
from torchvision import transforms
from img2vec_pytorch import Img2Vec
import time


def getTopK(query_vector, k=10, filter='*'):
    start = time.time()
    res = execute('FT.SEARCH', 'idx', f'({filter})=>[KNN {k} @vectors $vec_param AS distance]', 'SORTBY', 'distance',
            'RETURN', 6, 'id', 'brand', 'name', 'family', 'distance', '$.images[0].url', 'AS', 'image', 'PARAMS', 2,
            'vec_param', query_vector.tobytes())
    search_time = time.time() - start

    return res, search_time

# Get the model
async def get_boxes(image, device, threshold, max_det):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    img = transform(image).to(device)[None]
    modelRunner = redisAI.createModelRunner('detection_model')
    redisAI.modelRunnerAddInput(modelRunner, 'image', img)
    for i in range(4):
        redisAI.modelRunnerAddOutput(modelRunner, f'bounding_boxes_{i}')
    predictions = await redisAI.modelRunnerRunAsync(modelRunner)
    detections = non_max_suppression(predictions, conf_thres=threshold, max_det=max_det)
    return detections[0][:, :4].tolist()


#box_points = [xmin,ymin, xmax,ymax]
async def search_product(image, box_points):
    product = image.crop(box_points)
    img2vec = Img2Vec()
    product = img2vec.normalize(img2vec.to_tensor(img2vec.scaler(product))).unsqueeze(0).to(img2vec.device).numpy()

    modelRunner = redisAI.createModelRunner('prediction_model')
    redisAI.modelRunnerAddInput(modelRunner, 'image', product)
    redisAI.modelRunnerAddOutput(modelRunner, 'embedding')
    query_vector = await redisAI.modelRunnerRunAsync(modelRunner)

    res, search_time = getTopK(query_vector, 4)
    print('KNN search time\t', search_time)
    return res


async def run_flow(image, device, threshold, max_det):
    boxes = get_boxes(image, device, threshold, max_det)
    return {
        'results': [
            {
                'box': box,
                'products': search_product(image, box)
            } for box in boxes
        ]
    }

gb = GB('CommandReader')
gb.map(run_flow)
gb.register(trigger='RunSearchFlow')
