FROM sinzlab/pytorch:latest

ADD . /src/nnfabrik
WORKDIR /src

RUN pip3 install -e nnfabrik

WORKDIR /notebooks