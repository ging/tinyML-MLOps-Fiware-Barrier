

curl orion:1026/ngsi-ld/v1/entities -s -S -H 'Content-Type: application/ld+json' -d @- <<EOF
{
    "id": "urn:ngsi-ld:DensityDevice:1:Measurement:1",
    "type": "DeviceMeasurement",
    "name":{
      "type":"Property",
      "value": "Sensor1"
    },
    "value":{
      "type":"Property",
      "value": 12
    },
    "temperature":{
      "type":"Property",
      "value": 10
    },    
    "wind_speed":{
      "type":"Property",
      "value": 0.5
    },
    "@context": [
        "https://raw.githubusercontent.com/smart-data-models/dataModel.Device/master/context.jsonld"
    ]
}
EOF





curl orion:1026/ngsi-ld/v1/entities -s -S -H 'Content-Type: application/ld+json' -d @- <<EOF
{
    "id": "urn:ngsi-ld:DensityDevice:1:Prediction:1",
    "type": "DeviceMeasurement",
    "name":{
      "type":"Property",
      "value": "Sensor1Prediction"
    },
    "value":{
      "type":"Property",
      "value": 0
    },
    "timestamp":{
      "type":"Property",
      "value": 0
    },
    "@context": [
        "https://raw.githubusercontent.com/smart-data-models/dataModel.Device/master/context.jsonld"
    ]
}
EOF