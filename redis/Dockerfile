# syntax=docker/dockerfile:1

FROM redislabs/redisai:1.2.7-cpu-bionic as redisai
FROM redislabs/redisearch:2.6.1 as redisearch
FROM redislabs/redisgears:1.2.5

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -qqy

COPY . /data

ENV LD_LIBRARY_PATH /usr/lib/redis/modules

COPY --from=redisai ${LD_LIBRARY_PATH}/redisai.so ${LD_LIBRARY_PATH}/
COPY --from=redisai ${LD_LIBRARY_PATH}/backends ${LD_LIBRARY_PATH}/backends
COPY --from=redisearch ${LD_LIBRARY_PATH}/redisearch.so ${LD_LIBRARY_PATH}/
COPY --from=redisearch ${LD_LIBRARY_PATH}/rejson.so ${LD_LIBRARY_PATH}/

EXPOSE 6379

CMD [ "redis-server", "/data/redis.conf" ]
