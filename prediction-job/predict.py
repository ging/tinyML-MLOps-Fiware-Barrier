import os
from flask import Flask, request, jsonify
from datetime import datetime, timedelta
import time

import requests


def make_prediction(weekday, day, hour, temperature, wind_speed):
    current_file_path = os.path.abspath(__file__)
    here = os.path.dirname(current_file_path)
    out_dir = os.path.join(here, 'classifiers/latest')
    result_filename = os.path.join(out_dir, 'result')
    code_filename = os.path.join(out_dir, 'code')

    import subprocess
    errors = None
    try:
        subprocess.check_output([f'{code_filename}', str(weekday), str(day), str(hour), str(temperature), str(wind_speed)])
        errors = 0
    except subprocess.CalledProcessError as e:
        print(e)
        errors = e.returncode

    with open(result_filename, 'r') as f:
        result = int(f.read().strip())

    return result, errors

def get_weekday_day_hour_in_two_hours():
    current_time = datetime.now()
    new_time = current_time + timedelta(hours=2)
    weekday = new_time.weekday() + 1
    day = new_time.day
    hour = new_time.hour
    timestamp = round(time.time() * 1000)

    return weekday, day, hour, timestamp

def update_prediction(prediction, timestamp):
    entity_url = "http://orion:1026/ngsi-ld/v1/entities/urn:ngsi-ld:DensityDevice:1:Prediction:1/attrs"

    headers = {"Content-Type": "application/ld+json"}
    
    payload = {
        "value": {"type": "Property", "value": prediction},
        "timestamp": {"type": "Property", "value": timestamp},
        "@context": [
            "https://raw.githubusercontent.com/smart-data-models/dataModel.Device/master/context.jsonld"
        ]
    }

    response = requests.patch(entity_url, json=payload, headers=headers)
    if response.status_code == 204:
        print("Prediction updated successfully!")
    else:
        print("Failed to update prediction:", response.text)

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    temperature = data['data'][0]['temperature']['value']
    wind_speed = data['data'][0]['wind_speed']['value']
    weekday, day, hour, timestamp = get_weekday_day_hour_in_two_hours()
    prediction, _ = make_prediction(weekday, day, hour, temperature, wind_speed)
    print("prediction result:", prediction)
    update_prediction(prediction, timestamp)
    return jsonify({'message': 'Prediction made', 'prediction': prediction})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3001, debug=True)


