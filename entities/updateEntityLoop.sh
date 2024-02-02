#!/bin/bash

while true; do
    temperature=$(shuf -i 0-40 -n 1)
    wind_speed=$(shuf -i 0-20 -n 1)

      curl --location --request PATCH 'http://orion:1026/ngsi-ld/v1/entities/urn:ngsi-ld:DensityDevice:1:Measurement:1/attrs' \
      --header 'Content-Type: application/ld+json' \
      --data-raw '{
         "value":{
            "type":"Property",
            "value": 6
         },
         "temperature":{
            "type":"Property",
            "value": '"$temperature"'
         },
         "wind_speed":{
            "type":"Property",
            "value": '"$wind_speed"'
         },
         "@context": [
            "https://raw.githubusercontent.com/smart-data-models/dataModel.Device/master/context.jsonld"
         ]
      }'
    sleep 20
done
