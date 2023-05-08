# TODO: move this file into its own 'backend' folder

from flask import Flask, jsonify, request
from flask_restful import Resource, Api
import re
from preprocessing import Preprocessing
from generator import RegexGenerator
import ast
import json
import numpy as np

debug_mode = True
# with open("./model/build_config.json", "r", encoding="utf8") as file:
#     config = json.load(file)
model = RegexGenerator()
p = Preprocessing(database_path="./model/data.db")
matches = list(('abAB12', 'cdCD34', 'efEF56', 'ghGH78'))
rejections = list(('abc', 'def', 'ghi', 'jkl'))
matches = p.encode_texts(matches, 100, 10)
rejections = p.encode_texts(rejections, 100, 10)

matches = matches.reshape((1, 10, 100))

#FIXME: this is so convoluted, fix it to load like normal
model.compile(optimizer="adam", loss="mse")
rejections = rejections.reshape((1, 10, 100))
model.fit([matches, rejections], np.array([0.]))

model.load_weights("./model/model.h5")

allowed_origins = [r"http(s?)://127\.0\.0\.1.*",
                   r"http(s?)://lucasrpatten\.github\.io/RegexGenerator/"] if not debug_mode else [r".*"]


def allow_origin(origin):
    for pattern in allowed_origins:
        if re.fullmatch(pattern, origin) is not None:
            return True
    return False


def get_prediction(matches, rejections):
    matches = list(matches)
    rejections = list(rejections)
    matches = p.encode_texts(matches, 100, 10)
    rejections = p.encode_texts(rejections, 100, 10)

    matches = matches.reshape((1, 10, 100))
    rejections = rejections.reshape((1, 10, 100))

    response = model([matches, rejections])

    response = [chr(int(abs(i)*128)) for i in response[0]]
    return response


class GetRegex(Resource):
    def post(self):
        data = request.get_json()
        headers = request.headers
        origin = headers["Origin"]

        if not allow_origin(origin):
            return "Access denied", 403

        matches = data["matches"]
        rejections = data["rejections"]
        response = get_prediction(matches, rejections)

        return response, 200, {
            'Access-Control-Allow-Origin': f'{origin}',
            "Content-Type": "application/json"
        }

    def options(self):
        headers = request.headers
        origin = headers["Origin"]

        if not allow_origin(origin):
            return "Access denied", 403

        return None, 200, {
            'Access-Control-Allow-Origin': f'{origin}',
            'Access-Control-Allow-Methods': 'POST',
            'Access-Control-Allow-Headers': f'{headers["Access-Control-Request-Headers"]}',
        }


app = Flask(__name__)
api = Api(app)
api.add_resource(GetRegex, '/get-regex')
app.run(debug=debug_mode)
