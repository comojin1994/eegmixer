FROM continuumio/miniconda3:latest

RUN apt-get update && apt-get install -y git
RUN pip install mlflow==2.12.1 psycopg2-binary==2.9.9 pymysql==1.1.0 boto3==1.34.82