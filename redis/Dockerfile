# syntax=docker/dockerfile:1

FROM redislabs/redisai:1.2.7-cpu-bionic as redisai
FROM redislabs/redisgears:1.2.5 as redisgears
FROM redislabs/redisearch:2.6.1

COPY . /data

ENV LD_LIBRARY_PATH /usr/lib/redis/modules

COPY --from=redisai ${LD_LIBRARY_PATH}/redisai.so ${LD_LIBRARY_PATH}/
COPY --from=redisai ${LD_LIBRARY_PATH}/backends ${LD_LIBRARY_PATH}/backends
COPY --from=redisgears /var/opt/redislabs/lib/modules/redisgears.so ${LD_LIBRARY_PATH}/

EXPOSE 6379

CMD [ "redis-server", "/data/redis.conf" ]
