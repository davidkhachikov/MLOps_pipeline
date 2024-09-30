#!/bin/bash

# Activate virtual environment
source .venv/bin/activate

# Install required packages
pip install -r requirements.txt

# Update .bashrc with AIRFLOW_HOME
sed -i '/^export AIRFLOW_HOME=/d' ~/.bashrc
NEW_AIRFLOW_HOME="$PWD/services/airflow"
echo "export AIRFLOW_HOME=\"$NEW_AIRFLOW_HOME\"" >> ~/.bashrc

unset AIRFLOW_HOME

# Print the value of AIRFLOW_HOME
echo "AIRFLOW_HOME has been set to: $NEW_AIRFLOW_HOME. Please execute the following command to apply the changes:"
echo "source ~/.bashrc && source .venv/bin/activate"
echo "After running this command, the AIRFLOW_HOME environment variable will be properly configured."