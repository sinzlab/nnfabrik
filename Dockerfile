FROM sinzlab/pytorch:latest

ADD . /src/nnfabrik
RUN pip3 install sphinx-rtd-theme
RUN pip3 install -e /src/nnfabrik

WORKDIR /notebooks
