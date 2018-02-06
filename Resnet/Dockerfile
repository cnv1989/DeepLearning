FROM python:3-onbuild

RUN python ./resnet50_train.py
COPY resnet50.json .
COPY resnet50.h5 .
