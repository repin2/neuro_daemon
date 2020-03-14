from flask import Flask
from flask import request
from ansamble import predict_one_pic

app = Flask(__name__)
@app.route('/', methods=['POST'])
def process_picture():
    if request.method == 'POST':
        pic_bytes = request.json
        calories = predict_one_pic(pic_bytes['img'])
        return {'result': calories}


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)

