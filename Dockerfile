FROM sinzlab/pytorch:v1.2.0
    
WORKDIR /src


# Add editable installation of nnfabrik
ADD . /src/nnfabrik
RUN pip3 install -e /src/nnfabrik/ml-utils
RUN pip3 install -e /src/nnfabrik

WORKDIR /notebooks


