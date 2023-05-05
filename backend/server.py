from flask import Flask, jsonify, request
from flask_restful import Resource, Api
import ast
import json


class GetRegex(Resource):
    def post(self):
        data = request.get_json()
        print(data)
        return jsonify(["hi"]), 200, {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Expose-Headers': '*',
            "Content-Type": "application/json"
        }

    def get(self):
        return{"a": "hi"}, 200, {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Expose-Headers': '*',
            "Content-Type": "application/json"
        }


app = Flask(__name__)
api = Api(app)
api.add_resource(GetRegex, '/get-regex')
app.run(debug=True)


if __name__ == "__main__":
    main()
