FROM sinzlab/pytorch:latest

# WORKDIR /notebooks
WORKDIR /src
ADD . /src/notebooks

RUN pip3 install -e src/notebooks
RUN pip3 install -e src/notebooks/ml-utils

WORKDIR /notebooks

#RUN python setup.py install
#RUN pip3 install -e mei/mei
#RUN pip3 install -e ml-utils
#RUN pip3 install -e nnvision/nnvision

# Add editable installation of nnfabrik
#ADD . /src/projects
#RUN pip3 install -e /src/projects/ml-utils
#RUN pip3 install -e /src/projects
#RUN pip3 install -e /src/projects/mei/mei
#RUN pip3 install -e /src/projects/nnvision/nnvision
