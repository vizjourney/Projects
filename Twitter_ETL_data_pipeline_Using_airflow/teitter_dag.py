from datetime import timedelta
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime
from twitter_etl import run_twitter_etl

# Default configuration for the DAG
default_args = {
    'owner': 'airflow',
    'start_date': datetime(2020, 11, 8),
    'retries': 1,
    'retry_delay': timedelta(minutes=1)
}

# Create a new Airflow DAG
dag = DAG(
    'twitter_dag',
    default_args=default_args,
    description='DAG for running Twitter ETL process',
    schedule_interval=timedelta(days=1),
)

# Define a PythonOperator to run the Twitter ETL script
run_etl = PythonOperator(
    task_id='run_twitter_etl',
    python_callable=run_twitter_etl,
    dag=dag,
)

# Set task dependencies
run_etl

if __name__ == "__main__":
    dag.cli()
