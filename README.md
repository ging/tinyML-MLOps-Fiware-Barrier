# FIWARE Machine Learning TinyML and MLOps - Barrier use case

Start base infraestructure
```
docker compose up -d
```

```
cd airflow
docker compose up -d
```

## With Airflow as orchestrator

- Access http://localhost:5000 to access MLFlow client

- Access http://localhost:8080 to access the Airflow Web UI (user: airflow, password: airflow)

- Initialize the dags:
  - 1. `create_connection_dag` to create the connection to train server
  - 2. `train_model` to train the model


Every 20 seconds the `urn:ngsi-ld:DensityDevice:1:Measurement:1` entity is updates, orion sends a notification to the predict system, who updates the `urn:ngsi-ld:DensityDevice:1:Prediction:1`

To get the entities:

```
curl localhost:1026/ngsi-ld/v1/entities/urn:ngsi-ld:DensityDevice:1:Measurement:1
```

```
curl localhost:1026/ngsi-ld/v1/entities/urn:ngsi-ld:DensityDevice:1:Prediction:1
```

```
curl localhost:1026/ngsi-ld/v1/subscriptions
```
