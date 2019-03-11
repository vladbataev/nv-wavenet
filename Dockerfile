FROM nvidia/cuda:9.0-cudnn7-devel
ENV http_proxy http://proxy:8080
ENV HTTP_PROXY http://proxy:8080
ENV https_proxy https://proxy:8080
ENV HTTPS_PROXY https://proxy:8080

RUN apt-get update && apt-get install -y --no-install-recommends \
        vim \
        gfortran \
        libblas-dev \
        liblapack-dev \
        libffi-dev \
        libssl-dev \
        ca-certificates \
        curl \
        wget \
        libsqlite3-dev \
        libbz2-dev \
        libncurses-dev \
        sox \
        ffmpeg \
        git \
        locales \
        screen

RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

RUN wget https://www.python.org/ftp/python/3.6.0/Python-3.6.0.tgz
RUN tar xvf Python-3.6.0.tgz
WORKDIR Python-3.6.0
RUN ./configure
RUN make -j
RUN make install
RUN curl --silent --show-error --retry 5 https://bootstrap.pypa.io/get-pip.py | python3

RUN pip3 install https://download.pytorch.org/whl/cu90/torch-0.4.1-cp36-cp36m-linux_x86_64.whl \
                 torchvision numpy scipy matplotlib librosa \
                 tensorboardX inflect Unidecode jupyter keras cffi tqdm nltk \
                 sklearn seaborn pandas lws \
                 https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.12.0-cp36-cp36m-linux_x86_64.whl
