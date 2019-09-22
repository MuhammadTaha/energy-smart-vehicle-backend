from flask import Flask, request, jsonify, make_response
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/route-based', methods=['GET', 'POST'])
def route_based():
    return request.data

@app.route('/charging-based', methods=['GET', 'POST'])
def charging_based():
    return request.data

if __name__ == '__main__':
    app.run()