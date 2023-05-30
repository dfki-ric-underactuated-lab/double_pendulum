# syntax=docker/dockerfile:1

FROM ubuntu:20.04
WORKDIR /home/

RUN apt update
RUN apt-get install -y python3-pip
RUN pip install --upgrade pip
RUN pip3 install numpy==1.23.1 dill
RUN python3 -m pip install drake
RUN apt install -y wget unzip python-is-python3

# Copy everything
COPY . ./double_pendulum/

# Works using the fix in https://dfki-ric-underactuated-lab.github.io/double_pendulum/installation.html
# RUN apt-get install -y --no-install-recommends ffmpeg
# RUN apt-get install -y libyaml-cpp-dev libeigen3-dev libpython3.8 libx11-6 libsm6 libxt6 libglib2.0-0 python3-sphinx python3-numpydoc python3-sphinx-rtd-theme
