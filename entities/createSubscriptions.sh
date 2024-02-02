curl -v  orion:1026/ngsi-ld/v1/subscriptions/ -s -S -H 'Content-Type: application/ld+json' -d @- <<EOF
{
  "description": "A subscription to get updates from the device",
  "type": "Subscription",
  "entities": [{
    "id": "urn:ngsi-ld:DensityDevice:1:Measurement:1",
    "type": "DeviceMeasurement"
    }],
  "watchedAttributes": [
    "value",
    "temperature",
    "wind_speed"
    ],
  "notification": {
    "endpoint": {
      "uri": "http://predict:3001/predict",
      "accept": "application/json"
    }
  },
    "@context": [
        "https://uri.etsi.org/ngsi-ld/v1/ngsi-ld-core-context.jsonld",
        "https://raw.githubusercontent.com/smart-data-models/dataModel.Device/master/context.jsonld"
    ] 
}
EOF
