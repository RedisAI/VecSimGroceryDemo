from flask import Flask, request, send_file

app = Flask(__name__)

@app.route('/')
def index():
    return send_file('index.html')

@app.route('/search', methods=['POST'])
def search():
    if 'image' not in request.files:
        return 'no image found', 400

    file = request.files['image']
    # TODO
    file.save('test.jpg')

    return {
        'products': [
            {"score": 0.1, "id":4916118,"names":{"short":"Candy Mentchees","long":"Candy Mentchees"},"brand":"Mitzvah Kinder","isWeighable":False,"family":{"name":"Toys"},"images":["https://storage.googleapis.com/sp-public/product-images/global/4916118/4927867/large.jpg"]},
            {"score": 0.2, "id":5082713,"names":{"short":"Gourmet Red Wine Sauce","long":"Gourmet Red Wine Sauce"},"brand":"Jasmine Gourmet","isWeighable":False,"family":{"name":"Red Wine Vinegar"},"images":["https://storage.googleapis.com/sp-public/product-images/global/5082713/2736019/large.jpg"]}
        ]
    }
