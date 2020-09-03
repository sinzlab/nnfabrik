FROM sinzlab/pytorch:latest

ADD . /src/nnfabrik
WORKDIR /src
RUN pip3 install sphinx-rtd-theme

RUN pip3 install -e nnfabrik

WORKDIR /notebooks
ENTRYPOINT ["/bin/bash"]