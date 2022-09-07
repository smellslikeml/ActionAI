# Built using nvidia jetpack 4.6.1 

ARG BASE_IMAGE=nvcr.io/nvidia/l4t-base:r32.6.1
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /workspace

# change the locale from POSIX to UTF-8
RUN locale-gen en_US en_US.UTF-8 && update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
ENV LANG=en_US.UTF-8

WORKDIR /

# run updates
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
          git \
                cmake \
                build-essential \
                curl \
                ca-certificates \
                wget \
                gnupg2 \
                lsb-release \
    && rm -rf /var/lib/apt/lists/*


# compile yaml-cpp-0.6
RUN git clone --branch yaml-cpp-0.6.0 https://github.com/jbeder/yaml-cpp yaml-cpp-0.6 && \
    cd yaml-cpp-0.6 && \
    mkdir build && \
    cd build && \
    cmake -DBUILD_SHARED_LIBS=ON .. && \
    make -j$(nproc) && \
    cp libyaml-cpp.so.0.6.0 /usr/lib/aarch64-linux-gnu/ && \
    ln -s /usr/lib/aarch64-linux-gnu/libyaml-cpp.so.0.6.0 /usr/lib/aarch64-linux-gnu/libyaml-cpp.so.0.6

#
# install OpenCV (with GStreamer support)
#
COPY jetson-ota-public.asc /etc/apt/trusted.gpg.d/jetson-ota-public.asc

RUN echo "deb https://repo.download.nvidia.com/jetson/common r32.6 main" > /etc/apt/sources.list.d/nvidia-l4t-apt-source.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
            libopencv-python \
    && rm /etc/apt/sources.list.d/nvidia-l4t-apt-source.list \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------
# PyTorch Installations
# ----------------------------
#
# install prerequisites (many of these are for numpy)
#
ENV PATH="/usr/local/cuda-10.2/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda-10.2/lib64:/usr/local/cuda-10.2/extras/CUPTI/lib64:${LD_LIBRARY_PATH}"

RUN apt-get update && \
    ldconfig && \
    apt-get install -y --no-install-recommends \
            python3-pip \
		  python3-dev \
		  libopenblas-dev \
		  libopenmpi2 \
            openmpi-bin \
            openmpi-common \
		  gfortran \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install setuptools Cython wheel
RUN pip3 install --ignore-installed numpy==1.19.4 --verbose

ARG PYTORCH_URL=https://nvidia.box.com/shared/static/9eptse6jyly1ggt9axbja2yrmj6pbarc.whl
ARG PYTORCH_WHL=torch-1.6.0-cp36-cp36m-linux_aarch64.whl

RUN wget --quiet --show-progress --progress=bar:force:noscroll --no-check-certificate ${PYTORCH_URL} -O ${PYTORCH_WHL} && \
    pip3 install ${PYTORCH_WHL} --verbose && \
    rm ${PYTORCH_WHL}


#
# torchvision 0.7
#
ARG TORCHVISION_VERSION=v0.7.0
ARG TORCH_CUDA_ARCH_LIST="7.2"

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
		  git \
		  build-essential \
            libjpeg-dev \
		  zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

RUN git clone -b ${TORCHVISION_VERSION} https://github.com/pytorch/vision torchvision && \
    cd torchvision && \
    python3 setup.py install && \
    cd ../ && \
    rm -rf torchvision

# 
# PyCUDA
#
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"
RUN echo "$PATH" && echo "$LD_LIBRARY_PATH"

RUN apt-get install -y python3-dev 
RUN pip3 install cython

RUN apt-get update && apt-get install -y pkg-config libhdf5-100 libhdf5-serial-dev hdf5-tools libhdf5-dev zlib1g-dev zip libjpeg8-dev liblapack-dev libblas-dev gfortran
RUN ln -s /usr/lib/aarch64-linux-gnu/libhdf5_serial.so.10 /usr/lib/aarch64-linux-gnu/libhdf5.so
RUN ln -s /usr/lib/aarch64-linux-gnu/libhdf5_serial_hl.so.10 /usr/lib/aarch64-linux-gnu/libhdf5_hl.so

RUN pip3 install --upgrade pip setuptools wheel testresources && pip3 install markupsafe packaging future==0.18.2 mock==3.0.5 keras_preprocessing==1.1.2 keras_applications==1.0.8 gast==0.4.0 protobuf pybind11 pkgconfig
RUN env H5PY_SETUP_REQUIRES=0 pip3 install -U h5py==3.1.0

RUN pip3 install scipy

RUN pip3 install pycuda --verbose

RUN export OPENBLAS_CORETYPE=ARMV8
RUN pip3 uninstall -y numpy && pip3 install numpy==1.19.4

# -------------------
# torch2trt installations
# -------------------
RUN git clone https://github.com/NVIDIA-AI-IOT/torch2trt && \
    cd torch2trt && \
    python3 setup.py install --plugins
# -------------------
# trt_pose installation
# -------------------
RUN apt-get update && \
    apt-get install -y python3-matplotlib
RUN pip3 install tqdm cython pycocotools

RUN git clone https://github.com/NVIDIA-AI-IOT/trt_pose && \
    cd trt_pose && \
    python3 setup.py install 

# Additional installs
RUN cd / && git clone -b legacy_py3.6 https://github.com/QUVA-Lab/e2cnn.git
RUN cd /e2cnn/ && python3 setup.py install

# Install Tensorflow
RUN pip3 install --pre --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v46 tensorflow>=2
