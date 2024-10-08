import os
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from preprocess import sample_data, handle_initial_data, split_data
import yaml
from airflow.models.variable import Variable
from airflow.utils import timezone
from datetime import timedelta
import sys
sys.path.append('./code/models')
from train_model import train

# Initialize Airflow variable
with open("./configs/main.yml", 'r') as file:
    config = yaml.safe_load(file)

def update_current_version():
    current_version = int(Variable.get(key='current_version', default_var=0))
    
    # Read the version from the YAML file
    num_samples = config['data']['num_samples']
    
    # Calculate the current version
    current_version = (current_version + 1) % num_samples
    
    # Store the version as an Airflow variable
    Variable.set(key='current_version', value=current_version)

with DAG(
    dag_id="data_extract",
    schedule_interval="*/15 * * * *",
    catchup=True,
    start_date = timezone.utcnow() - timedelta(minutes=20),
    max_active_runs=1
) as dag:
    update_sample_number_task = PythonOperator(
        task_id="update_current_version",
        python_callable=update_current_version
    )

    extract_task = PythonOperator(
        task_id="extract_data_sample",
        python_callable=sample_data,
        op_args=[
            os.path.join(config['BASE_DIR'], config['RAW_DATA']), 
            os.path.join(config['BASE_DIR'], config['PROCESSED_PATH']),
            int(Variable.get(key='current_version', default_var=0)),
            config['data']['num_samples']
        ]
    )

    preprocess_task = PythonOperator(
        task_id="preprocess_data_sample",
        python_callable=handle_initial_data,
        op_args=[
            os.path.join(config['BASE_DIR'], config['PROCESSED_PATH'])
        ]
    )

    train_test_split_task = PythonOperator(
        task_id="train_test_split",
        python_callable=split_data,
        op_args=[
            os.path.join(config['BASE_DIR'], config['PROCESSED_PATH']), 
            os.path.join(config['BASE_DIR'], config['TRAIN_TEST_PATH']), 
            config['data']['test_size'],
            config['data']['random_state']
        ]
    )

    train_model_task = PythonOperator(
        task_id="learning_models",
        python_callable=train
    )

    move_model_for_docker_task = BashOperator(
        task_id="copy-model",
        bash_command=(
            f"rsync -a {os.path.join(config['BASE_DIR'], './models/best.pth')} "
            f"{os.path.join(config['BASE_DIR'], './code/deployment/api/model_dir/model.pth')} && "
            f"rsync -a {os.path.join(config['BASE_DIR'], './code/models/models.py')} "
            f"{os.path.join(config['BASE_DIR'], './code/deployment/api/models.py')}"
        ),
    )

    build_docker_image = BashOperator(
        task_id='build_docker_image',
        bash_command=(
            f"docker-compose -f {os.path.join(config['BASE_DIR'], './code/deployment/docker-compose.yml')} build"
        ),
        dag=dag,
    )

    extract_task >> preprocess_task >> train_test_split_task >> train_model_task >> move_model_for_docker_task >> build_docker_image >> update_sample_number_task