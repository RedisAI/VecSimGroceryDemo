# syntax=docker/dockerfile:1

FROM python:3.8

ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0

COPY app_redisai .
COPY data/models models
COPY data/models_loader.py .

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

RUN python3 models_loader.py

EXPOSE 5001
CMD ["flask", "run"]
