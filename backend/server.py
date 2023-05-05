from flask import Flask, jsonify, request
from flask_restful import Resource, Api
import ast
import json


class GetRegex(Resource):
    def post(self):
        data = request.get_json()
        print(data)
        headers = request.headers
        return "hi", 200, {
            'Access-Control-Allow-Origin': f'{headers["Origin"]}',
            "Content-Type": "application/json"
        }

    def options(self):
        headers = request.headers
        print(headers)
        return None, 200, {
            'Access-Control-Allow-Origin': f'{headers["Origin"]}',
            'Access-Control-Allow-Methods': 'POST',
            'Access-Control-Allow-Headers': f'{headers["Access-Control-Request-Headers"]}',
        }


app = Flask(__name__)
api = Api(app)
api.add_resource(GetRegex, '/get-regex')
app.run(debug=True)


if __name__ == "__main__":
    main()
