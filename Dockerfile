FROM sinzlab/pytorch:v3.8-torch1.4.0-cuda10.1-dj0.12.4
    
WORKDIR /src


# Add editable installation of nnfabrik
ADD . /src/nnfabrik
RUN pip3 install -e /src/nnfabrik/ml-utils
RUN pip3 install -e /src/nnfabrik

WORKDIR /notebooks


