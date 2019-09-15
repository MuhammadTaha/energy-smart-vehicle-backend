from flask import Flask

app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Hello World!'

@app.route('/min-energy')
def min_energy():
    return 'Hello World!'

@app.route('/min-cost')
def min_cost():
    return 'minimum cost time!'

@app.route('/min-drive-time')
def min_drive_time():
    return 'minimum driving time'


if __name__ == '__main__':
    app.run()
