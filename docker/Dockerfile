FROM apache/airflow:2.7.1-python3.10

USER root

# Install system packages needed for Prophet & scientific libs
RUN apt-get update && apt-get install -y \
    build-essential \
    libatlas-base-dev \
    gfortran \
    libgomp1 \
    && apt-get clean

# Create a writable scratch directory for Prophet/CmdStanpy and set permissions
RUN mkdir -p /opt/airflow/tmp && chmod -R 777 /opt/airflow/tmp

# Copy requirements inside container
COPY docker/requirements.txt /requirements.txt

# Switch back to airflow user
USER airflow

# Install Python dependencies
RUN pip install --no-cache-dir -r /requirements.txt
