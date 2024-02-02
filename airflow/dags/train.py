from airflow import DAG
from airflow.providers.http.operators.http import SimpleHttpOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'train_model',
    default_args=default_args,
    description='Make a cURL request in Apache Airflow DAG',
    schedule_interval=None,
)

curl_request_predict = SimpleHttpOperator(
    task_id='train_model',
    http_conn_id="http_train",
    method='POST',
    endpoint='/train',
    dag=dag,
)

curl_request_predict
