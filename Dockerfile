FROM sinzlab/pytorch:latest

RUN pip3 install -e /src/nnfabrik

WORKDIR /notebooks