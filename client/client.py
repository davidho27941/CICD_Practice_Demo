import requests
import os 
import yaml
import cv2

def read_image(path):
    return cv2.imread(path)

def encode_image(image):
    _, endcoded_image = cv2.imencode(".jpg", image)
    return endcoded_image

def send_request(url, data, header):
    response = requests.get(url, data=data, headers=header)
    print(response, response.status_code, response.content)

if __name__ == "__main__":
    with open(f"{os.getcwd()}/config/config.yml", 'r') as stream:
        config = yaml.load(stream, Loader=yaml.CLoader)
    base_url = "http://192.168.0.53:5555"
    api_route = "/api/test"
    local_image_path = config['client']['TEST_DATASET_PATH']
    header = {'content-type': 'image/png'}

    image = read_image(local_image_path)
    endcoded_image = encode_image(image)

    send_request(base_url+api_route, data=endcoded_image.tobytes(), header=header)