FROM jupyter/scipy-notebook
USER root
WORKDIR /code
ADD . /code

RUN pip install joblib
RUN mkdir /code/model

# Env variables
ENV MODEL_DIR=/code/model
ENV MODEL_FILE=clf.joblib
ENV METADATA_FILE=metadata.json

# COPY train.py ./train.py
# COPY inference.py ./inference.py

RUN python3 train.py