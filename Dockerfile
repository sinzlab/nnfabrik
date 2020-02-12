FROM sinzlab/pytorch:latest
    
WORKDIR /src


# Add editable installation of nnfabrik
ADD . /src/nnfabrik
RUN pip3 install -e /src/nnfabrik/ml-utils
RUN pip3 install -e /src/nnfabrik

WORKDIR /notebooks


