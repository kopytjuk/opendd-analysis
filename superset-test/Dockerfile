FROM apache/superset

# Switching to root to install the required packages
USER root

RUN apt-get install unzip
RUN wget https://data.l3pilot.eu/OpenDD/opendd_v3-example_data.zip
RUN unzip opendd_v3-example_data.zip

# allow sqlite
RUN echo "PREVENT_UNSAFE_DB_CONNECTIONS = False" >> /app/superset/config.py

# Switching back to using the `superset` user
USER superset
