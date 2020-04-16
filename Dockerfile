FROM sinzlab/pytorch:latest

# WORKDIR /notebooks
WORKDIR /src
ADD . /src/notebooks

RUN pip3 install -e src/notebooks
RUN pip3 install -e src/notebooks/ml-utils

WORKDIR /notebooks