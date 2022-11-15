# syntax=docker/dockerfile:1

FROM python:3.8

ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_RUN_PORT=5001

COPY app_redisai app_redisai
COPY data/models data/models
COPY data/models_loader.py data/

WORKDIR app_redisai

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

EXPOSE 5001

RUN chmod +x start.sh

CMD ["/bin/bash","-c","./start.sh"]
