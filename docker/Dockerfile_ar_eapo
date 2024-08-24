From ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install wget -y && \
    apt-get install unzip -y && \
    apt-get install git -y && \
    apt-get install vim -y && \
    apt-get install python3-pip -y && \
    apt-get install libyaml-cpp-dev -y && \
    #apt install libeigen3-dev -y && \
    apt-get install ffmpeg -y && \
    apt-get install libpython3.10 -y && \
    apt-get install libx11-6 -y && \
    apt-get install libsm6 -y && \
    apt-get install libxt6 -y && \
    apt-get install libglib2.0-0 -y && \
    apt-get install python3-sphinx -y && \
    apt-get install python3-numpydoc -y && \
    apt-get install python3-sphinx-rtd-theme -y && \
    apt-get install python-is-python3

# libeigen3-dev install does not work with apt
RUN wget -O Eigen.zip https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip
RUN unzip Eigen.zip
RUN cp -r eigen-3.4.0/Eigen /usr/local/include

RUN pip install -U pip

COPY . ./double_pendulum/

WORKDIR "/double_pendulum"

RUN make install
RUN make pythonfull

RUN python -m pip install stable-baselines3==2.3.2 cloudpickle==3.0.0
