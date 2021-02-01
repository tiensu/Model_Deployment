FROM tensorflow/tensorflow:2.3.0-gpu
WORKDIR /uwsgi
ADD . /uwsgi
RUN pip install -r requirements.txt
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
CMD ["uwsgi", "app.ini"]
