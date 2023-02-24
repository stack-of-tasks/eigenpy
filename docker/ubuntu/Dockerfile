FROM ubuntu:22.04

RUN apt-get update -y && DEBIAN_FRONTEND=noninteractive apt-get install -qqy \
    build-essential \
    cmake \
    cmake-curses-gui \
    vim \
    gdb \
    git \
    libboost-all-dev \
    libeigen3-dev \
    liburdfdom-dev \
    python3-numpy

WORKDIR /src
RUN git clone --recursive -j2 -b devel https://github.com/stack-of-tasks/eigenpy

ENV CTEST_OUTPUT_ON_FAILURE=ON
ENV CTEST_PROGRESS_OUTPUT=ON
ENV CTEST_PARALLEL_LEVEL=2

WORKDIR /src/eigenpy/build
RUN cmake -DPYTHON_EXECUTABLE=$(which python3) -DCMAKE_BUILD_TYPE=Debug -DBUILD_TESTING=ON ..
RUN make -sj2
