FROM ubuntu:14.04

RUN apt-get update && \
    apt-get -y install python3-dev python3-pip python3-numpy llvm-3.6-dev libedit-dev zlib1g-dev && \
    ln -s /usr/bin/llvm-config-3.6 /usr/local/bin/llvm-config
COPY requirements.txt /opt/src/requirements.txt
RUN pip3 install -r /opt/src/requirements.txt
COPY nbody /opt/src/nbody
CMD ["/usr/bin/python3", "/opt/src/nbody/simulation.py"]