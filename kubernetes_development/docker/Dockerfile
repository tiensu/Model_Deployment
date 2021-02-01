FROM jupyter/scipy-notebook
USER root

WORKDIR /docker
ADD . /docker

RUN pip install flask flask-restful joblib

RUN mkdir /docker/model
ENV MODEL_DIR=/docker/model
ENV MODEL_FILE=clf.joblib
ENV METADATA_FILE=metadata.json

RUN python3 train.py