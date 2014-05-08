from flask import Flask
flask_politeness = Flask(__name__)

@flask_politeness.route('/')
def hello_world():
    return 'Hello World!'

if __name__ == '__main__':
    flask_politeness.run()
