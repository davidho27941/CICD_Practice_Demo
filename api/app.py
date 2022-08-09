import numpy as np
import jsonpickle
import cv2

from flask import Flask, request, Response

app = Flask(__name__)

@app.route("/api/test", methods=['GET'])
def test():
    req = request
    array = np.fromstring(req.data, np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)

    response = {
        "message": f"Image received. Image size: {image.shape[0]}, {image.shape[1]}",
    }

    response_encoded = jsonpickle.encode(response)
    return Response(response=response_encoded, status=200, mimetype="application/json")

if __name__ == "__main__":
    app.run(host="192.168.0.53", port="5555")