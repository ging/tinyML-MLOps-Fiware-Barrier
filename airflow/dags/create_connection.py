from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime

from airflow import settings
from airflow.models import Connection

def create_connection_function():
    # Define the connection parameters
    conn_id = "http_train"
    conn_type = "http"
    host = "http://train"
    port = 3000
    schema = "http"

    # Create a new connection object
    new_conn = Connection(
        conn_id=conn_id,
        conn_type=conn_type,
        host=host,
        port=port,
        schema=schema
    )

    # Add the new connection to the Airflow metadata database
    session = settings.Session()
    session.add(new_conn)
    session.commit()
    session.close()


default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
}

dag = DAG(
    'create_connection_dag',
    default_args=default_args,
    description='Automatically create a connection in Airflow',
    schedule_interval='@once',
)

create_connection_operator = PythonOperator(
    task_id='create_connection_task',
    python_callable=create_connection_function,  
    dag=dag,
)

create_connection_operator
