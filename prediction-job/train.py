import os
import warnings
import sys

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import mlflow
import mlflow.sklearn

import time
import emlearn
import logging
import threading

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

from flask import Flask, request, jsonify

current_file_path = os.path.abspath(__file__)
#here = os.path.dirname(current_file_path)
here = '/prediction-job'

def eval_metrics(actual, pred, average='micro'):
    accuracy = accuracy_score(actual, pred)
    precision = precision_score(actual, pred, average=average)
    recall = recall_score(actual, pred, average=average)
    f1 = f1_score(actual, pred, average=average)
    return accuracy, precision, recall, f1


def generateCcode(out_dir, model_filename, output_filename):
    print(out_dir)
    code = f'''
    #include "{model_filename}" // emlearn generated model

    #include <stdio.h> // printf

    int
    predict(float weekday, float day, float hour, float temperature, float wind_speed) {{
        const int n_features = 5;
        const float features[5] = {{weekday, day, hour, temperature, wind_speed}};
        const int32_t out = model_predict(features, n_features);

        printf(\"Predicted label: %d\\n\", out);

        FILE *fptr;
        fptr = fopen("{output_filename}", "w");
        fprintf(fptr, "%d", out);
        fclose(fptr);

        return 0;
    }}

    int
    main(int argc, const char *argv[])
    {{
    
        float weekday = atof(argv[1]);
        float day = atof(argv[2]);
        float hour = atof(argv[3]);
        float temperature = atof(argv[4]);
        float wind_speed = atof(argv[5]);

        return predict(weekday, day, hour, temperature, wind_speed);
    }}'''

    code_file = os.path.join(out_dir,f'code.c')
    with open(code_file, 'w') as f:
        f.write(code)

    print('Generated', code_file)

    include_dirs = [emlearn.includedir]
    path = emlearn.common.compile_executable(
            code_file,
            out_dir,
            name=f'code',
            include_dirs=include_dirs
    )
    print(path)


def train():
    warnings.filterwarnings("ignore")
    randmos_state = 40
    np.random.seed(randmos_state)

    csv_file = os.path.join(here, "barrier.csv")
    
    data = pd.read_csv(csv_file, sep=",")
    
    train, test = train_test_split(data)
    target_column = "density"
    feature_columns = ["weekday","day","hour","temperature","wind_speed"]

    n_estimators = int(sys.argv[1]) if len(sys.argv) > 1 else 50

    experiment_name = 'FIWARE_MLOPS_TINYML'
    mlflow.set_tracking_uri('http://mlflow-server:5000')
    mlflow.set_experiment(experiment_name=experiment_name)

    with mlflow.start_run():
        name ="random_forest"
        timestamp = round(time.time() * 1000)
        run_name = f'{name}_{timestamp}'

        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=10, random_state=randmos_state)
        model.fit(train[feature_columns], train[target_column])

        test_pred = model.predict(test[feature_columns])

        (accuracy, precision, recall, f1) = eval_metrics(test[target_column], test_pred)


        print("RandomForestClassifier model (n_estimators={:f}:".format(n_estimators))
        print("  accuracy: %s" % accuracy)
        print("  precision: %s" % precision)
        print("  recall: %s" % recall)
        print("  f1: %s" % f1)

        mlflow.set_tag("mlflow.runName", run_name)
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1", f1)
    

        out_dir = os.path.join(here, 'classifiers/latest')
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        
        model_filename = os.path.join(out_dir, f'{name}_model.h')
        cmodel = emlearn.convert(model)
        cmodel.save(file=model_filename, name='model')
        
        test_pred_tiny = cmodel.predict(test[feature_columns])
        test_data = np.array(test[feature_columns]).flatten()

        (accuracy_tiny, precision_tiny, recall_tiny, f1_tiny) = eval_metrics(test[target_column], test_pred_tiny)


        print("RandomForestClassifier cmodel (n_estimators={:f}:".format(n_estimators))
        print("  accuracy_tiny: %s" % accuracy_tiny)
        print("  precision_tiny: %s" % precision_tiny)
        print("  recall_tiny: %s" % recall_tiny)
        print("  f1_tiny: %s" % f1_tiny)

        mlflow.log_param("n_estimators_tiny", n_estimators)
        mlflow.log_metric("accuracy_tiny", accuracy_tiny)
        mlflow.log_metric("precision_tiny", precision_tiny)
        mlflow.log_metric("recall_tiny", recall_tiny)
        mlflow.log_metric("f1_tiny", f1_tiny)
        
        
        mlflow.sklearn.log_model(model, name)
        
        result_filename = os.path.join(out_dir, 'result')
        generateCcode(out_dir, model_filename, result_filename)



app = Flask(__name__)

@app.route('/train', methods=['POST'])
def train_model():
    train_thread = threading.Thread(target=train)
    train_thread.start()
    return jsonify({'message': 'Training in progress...'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)
