# Apache Superset Test

Here, the openDD dataset is analysed with Apache Superset.

## Setup

We will run Superset via Docker, refer to [Superset Dockerhub](https://hub.docker.com/r/apache/superset).

```shell
cd superset_test

docker build -t superset-test .

# run docker
docker run -d -p 8181:8088 --name superset superset-test

# create admin user
docker exec -it superset superset fab create-admin --username admin  --firstname Superset  --lastname Admin  --email admin@superset.com  --password admin

# upgrade database
docker exec -it superset superset db upgrade

# init
docker exec -it superset superset init
```

Now open the browser to  http://localhost:8181/login/ and log in with `admin` at both user and password.

Add a new database by providing the following sqlite paths:

```
# map data
sqlite:////app/example_data/map_rdb1/map_rdb1.sqlite

# object tracker data
sqlite:////app/example_data/rdb1_4.sqlite
```

Have fun!
