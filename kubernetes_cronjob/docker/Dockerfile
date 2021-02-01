FROM jupyter/scipy-notebook
USER root
WORKDIR /docker
ADD . /docker

RUN pip install awscli joblib boto3
RUN mkdir /docker/model

# Env variables
ENV MODEL_DIR=/docker/model
ENV MODEL_FILE=clf.joblib
ENV METADATA_FILE=metadata.json
ENV BUCKET_NAME=kubernetes-job
