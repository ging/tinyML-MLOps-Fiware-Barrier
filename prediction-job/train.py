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

from joblib import dump, load

import time
import emlearn
import logging
import threading

from emlearn.evaluate.trees import model_size_bytes, compute_cost_estimate

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

from flask import Flask, request, jsonify

current_file_path = os.path.abspath(__file__)
here = os.path.dirname(current_file_path)
# here = '/prediction-job'

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
        const int out = model_predict(features, n_features);

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

    csv_file = os.path.join(here, "barrier_half.csv")
    
    data = pd.read_csv(csv_file, sep=",")

    print(f'Dataset size in rows: {data.shape[0]}')
    
    train, test = train_test_split(data, test_size=0.2, random_state=randmos_state)
    print(f'Test dataset size in rows: 20%')
    target_column = "density_binary"
    feature_columns = ["weekday","day","hour","temperature","wind_speed"]

    n_estimators_big = 50
    max_depth_big = 10

    experiment_name = 'FIWARE_MLOPS_TINYML'
    mlflow.set_tracking_uri('http://localhost:5001')
    mlflow.set_experiment(experiment_name=experiment_name)

    with mlflow.start_run():
        name_big ="random_forest_big"
        timestamp_train_big_start = time.time_ns() / (10**6)
        name = "random_forest"
        run_name = f'{name}_{timestamp_train_big_start}'

        model_big = RandomForestClassifier(n_estimators=n_estimators_big, max_depth=max_depth_big, random_state=randmos_state)
        model_big.fit(train[feature_columns], train[target_column])
        timestamp_train_big_end = time.time_ns() / (10**6)

        timestamp_predict_big_start = time.time_ns() / (10**6)
        test_pred = model_big.predict(test[feature_columns])
        timestamp_predict_big_end = time.time_ns() / (10**6)

        (accuracy, precision, recall, f1) = eval_metrics(test[target_column], test_pred)

        train_big_time = timestamp_train_big_end - timestamp_train_big_start
        predict_base_time = timestamp_predict_big_end - timestamp_predict_big_start

        out_dir_base = os.path.join(here, 'classifiers/latest_big')
        if not os.path.exists(out_dir_base):
            os.makedirs(out_dir_base)
        
        fname = os.path.join(here, 'classifiers/latest_big', f'{name_big}_model.joblib')
        dump(model_big, fname)
        model_big_size_MB = os.path.getsize(fname) / 1024 / 1024


        print("RandomForestClassifier model (n_estimators={:f}:".format(n_estimators_big))
        print("  max_depth_big: %s" % max_depth_big)
        print("  accuracy_big: %s" % accuracy)
        print("  precision_big: %s" % precision)
        print("  recall_big: %s" % recall)
        print("  f1_big: %s" % f1)
        print("  time_big_train ms: %s" % train_big_time)
        print("  time_big_predict ms: %s" % predict_base_time)
        print("  model_big_size_MB: %s" % model_big_size_MB)

        mlflow.set_tag("mlflow.runName", run_name)
        mlflow.log_param("n_estimators_big", n_estimators_big)
        mlflow.log_param("max_depth_big", max_depth_big)
        mlflow.log_metric("accuracy_big", accuracy)
        mlflow.log_metric("precision_big", precision)
        mlflow.log_metric("recall_big", recall)
        mlflow.log_metric("f1_big", f1)
        mlflow.log_metric("time_train_big ms", timestamp_train_big_end - timestamp_train_big_start)
        mlflow.log_metric("time_predict_big ms", timestamp_predict_big_end - timestamp_predict_big_start)
        mlflow.log_metric("model_big_size_MB", model_big_size_MB)


        n_estimators = 10
        max_depth = 8
        timestamp_train_base_start = time.time_ns() / (10**6)
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=randmos_state)
        model.fit(train[feature_columns], train[target_column])
        timestamp_train_base_end = time.time_ns() / (10**6)

        train_base_time = timestamp_train_base_end - timestamp_train_base_start        
    
        out_dir = os.path.join(here, 'classifiers/latest')
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        
        timestamp_convert_c_start = time.time_ns() / (10**6)
        model_filename = os.path.join(out_dir, f'{name}_model.h')
        cmodel = emlearn.convert(model)
        timestamp_convert_c_end = time.time_ns() / (10**6)
        cmodel.save(file=model_filename, name='model')
        compiled_model_filename = os.path.join(out_dir, f'code')

        result_filename = os.path.join(out_dir, 'result')
        generateCcode(out_dir, model_filename, result_filename)
        
        timestamp_predict_c_start = time.time_ns() / (10**6)
        test_pred_tiny = cmodel.predict(test[feature_columns])
        timestamp_predict_c_end = time.time_ns() / (10**6)
        test_data = np.array(test[feature_columns]).flatten()

        (accuracy_tiny, precision_tiny, recall_tiny, f1_tiny) = eval_metrics(test[target_column], test_pred_tiny)

        cmodel_size_MB = os.path.getsize(compiled_model_filename) / 1024 / 1024
      
        convert_c_time = timestamp_convert_c_end - timestamp_convert_c_start
        predict_c_time = timestamp_predict_c_end - timestamp_predict_c_start

        print("RandomForestClassifier cmodel (n_estimators={:f}:".format(n_estimators))
        print("  max_depth_tiny: %s" % max_depth)
        print("  accuracy_tiny: %s" % accuracy_tiny)
        print("  precision_tiny: %s" % precision_tiny)
        print("  recall_tiny: %s" % recall_tiny)
        print("  f1_tiny: %s" % f1_tiny)
        print("  time_train_tiny ms %s" % train_base_time)
        print("  time_convert_c_tiny ms %s" % convert_c_time)
        print("  time_train_and_convert_c_tiny ms %s" % (train_base_time + convert_c_time))
        print("  time_predict_c_tiny ms %s" % predict_c_time)
        print("  model_tiny_size_MB: %s" % cmodel_size_MB)


        mlflow.log_param("n_estimators_tiny", n_estimators)
        mlflow.log_param("max_depth_tiny", max_depth)
        mlflow.log_metric("accuracy_tiny", accuracy_tiny)
        mlflow.log_metric("precision_tiny", precision_tiny)
        mlflow.log_metric("recall_tiny", recall_tiny)
        mlflow.log_metric("f1_tiny", f1_tiny)
        mlflow.log_metric("time_train_tiny ms", train_base_time)
        mlflow.log_metric("time_convert_c_tiny ms", convert_c_time)
        mlflow.log_metric("time_train_and_convert_c_tiny ms", train_base_time + convert_c_time)
        mlflow.log_metric("time_predict_c_tiny ms", predict_c_time)
        mlflow.log_metric("model_tiny_size_kb", cmodel_size_MB)       
        mlflow.sklearn.log_model(model, name)
        



app = Flask(__name__)

@app.route('/train', methods=['GET', 'POST'])
def train_model():
    train_thread = threading.Thread(target=train)
    train_thread.start()
    return jsonify({'message': 'Training in progress...'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)
