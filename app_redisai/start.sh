#!/bin/bash

cd /data
python3 models_loader.py --redis_server redis://redis:6379
cd /app_redisai
flask run
