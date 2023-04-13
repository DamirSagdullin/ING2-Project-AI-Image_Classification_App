FROM tiangolo/uvicorn-gunicorn-fastapi:python3.9

RUN pip install --upgrade pip

COPY ./app/requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

COPY ./app /app