FROM apache/airflow:2.7.3-python3.10

USER root

# Install system dependencies
RUN apt-get update \
  && apt-get install -y --no-install-recommends \
         vim \
  && apt-get autoremove -yqq --purge \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

USER airflow

# Copy requirements file
COPY requirements.txt /opt/airflow/

# Install Python dependencies
RUN pip install --no-cache-dir -r /opt/airflow/requirements.txt
