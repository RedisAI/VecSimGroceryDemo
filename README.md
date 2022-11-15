# Online Grocery Application - Vector Similarity Demo

An application that uses vector similarity search for finding similar grocery items based on an image, and allows adding them to user's cart.

## How to run
To run the demo app and load the data, run the following commands:
```
# Clone the repository
$ git clone https://github.com/RedisAI/VecSimGroceryDemo.git
$ cd VecSimGroceryDemo

# Launch the demo with docker-compose (may require sudo in linux)
$ docker-compose up
```

## Usage

Upon lunching the demo, a Redis instance is created in a container with grocery items data loaded into it. Each product is represented by a JSON document that was preloaded using [RedisJSON](/docs/stack/json), and contain the following properties: id, brand, name, family, and **a list of vector embeddings that represent images of the product** from different angels. These documents are indexed using [RediSearch](/docs/stack/json).

### The flow
Upon sending a `search` request to the app server, the following steps will take place:
1. An image detection model (yolov5) will run and yield all the items that are recognized in the image.
2. For every detected product: 
   1. An encoding model (the output of the second last layer of resnet-18) will generate a vector embedding of the product image (note that this is the same model that generated the vector embeddings for the stored products images).
   2. Search for the top 4 similar products using *RediSearch*, and return them to the user.

### Basic app
For searching similar products, go to `localhost:5000` in your browser, and take a picture of the desired products. Then, the application will detect the products in the image and will search for the 4 most similar products that are available in the database. These should appear upon clicking the appropriate rectangle, and clicking the "+" sign will allow you to add the product to your cart.

### Advanced - orchestrate flow with RedisAI and RedisGears
You can run an "improved" application that runs the entire flow within Redis, by using `localhost:5001`. While the basic app is responsible for running the entire flow within the app, the advanced app is leaner and only responsible for triggering the flow over the given image in [RedisGears](https://oss.redis.com/redisgears/index.html). Hence, the entire flow is being executed with a **single server command**, that in turn, use [RedisAI](https://oss.redis.com/redisai/) to run the models' inferences and RediSearch to perform the vector similarity top K search.

Note that with the advance application, you can replace the underline detection and encoding models without changing the app itself! By updating `data/models_loader.py` and `app_redisai/gear.py` to use your new models, you can run `data/models_loader.py` while the demo is running and update the models stored in RedisAI. This will work, as long as you don't change the gear's entrypoint (currently `RunSearchFlow` is registered with 4 predefined inputs - see `app_redisai/app.py`)
