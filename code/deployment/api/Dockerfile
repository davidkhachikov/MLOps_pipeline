# Build an image that can serve mlflow models.
FROM python:3.10.12-slim

RUN apt-get -y update && apt-get install -y --no-install-recommends nginx

WORKDIR /

COPY . /code

RUN pip install --no-cache-dir --upgrade -r /code/model_dir/requirements.txt

CMD ["python", "/code/api.py", "--host=0.0.0.0", "--port=5001"]