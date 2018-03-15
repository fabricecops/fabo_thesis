FROM nvidia/cuda:9.0-cudnn7-runtime-ubuntu16.04

ARG https_proxy
ARG http_proxy

########################
###  OPENCV INSTALL  ###
########################

ARG OPENCV_VERSION=3.4.0
ARG OPENCV_INSTALL_PATH=/opencv/usr

RUN apt-get update && \
        apt-get install -y \
	python-pip \
        build-essential \
        cmake \
        git \
        wget \
        unzip \
        yasm \
        pkg-config \
        libswscale-dev \
        libtbb2 \
        libtbb-dev \
        libjpeg-dev \
        libpng-dev \
        libtiff-dev \
        libjasper-dev \
        libavformat-dev \
        libpq-dev \

	libxine2-dev \
	libglew-dev \
	libtiff5-dev \
	zlib1g-dev \
	libjpeg-dev \
	libpng12-dev \
	libjasper-dev \
	libavcodec-dev \
	libavformat-dev \
	libavutil-dev \
	libpostproc-dev \
	libswscale-dev \
	libeigen3-dev \
	libtbb-dev \
	libgtk2.0-dev \
	pkg-config \
        ##
        ## Python
        python-dev \
        python-numpy \
        python3-dev \
        python3-numpy \
	## VTK
	#libvtk6-dev \
    ## Cleanup
    && rm -rf /var/lib/apt/lists/*

#RUN pip install numpy

## Create install directory
## Force success as the only reason for a fail is if it exist

RUN mkdir -p $OPENCV_INSTALL_PATH; exit 0

WORKDIR /

## Single command to reduce image size
## Build opencv
RUN wget https://github.com/opencv/opencv/archive/$OPENCV_VERSION.zip \
    && unzip $OPENCV_VERSION.zip \
    && mkdir /opencv-$OPENCV_VERSION/cmake_binary \
    && cd /opencv-$OPENCV_VERSION/cmake_binary \
    && cmake -DBUILD_TIFF=ON \
       -DBUILD_opencv_java=OFF \
       -DBUILD_SHARED_LIBS=OFF \
       -DWITH_CUDA=ON \
       # -DENABLE_FAST_MATH=1 \
       # -DCUDA_FAST_MATH=1 \
       # -DWITH_CUBLAS=1 \
       -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-8.0 \
       ##
       ## Should compile for most card
       ## 3.5 binary code for devices with compute capability 3.5 and 3.7,
       ## 5.0 binary code for devices with compute capability 5.0 and 5.2,
       ## 6.0 binary code for devices with compute capability 6.0 and 6.1,
       -DCUDA_ARCH_BIN='3.0 3.5 5.0 6.0 6.2' \
       -DCUDA_ARCH_PTX="" \
       ##
       ## AVX in dispatch because not all machines have it
       -DCPU_DISPATCH=AVX,AVX2 \
       -DENABLE_PRECOMPILED_HEADERS=OFF \
       -DWITH_OPENGL=OFF \
       -DWITH_OPENCL=OFF \
       -DWITH_QT=OFF \
       -DWITH_IPP=ON \
       -DWITH_TBB=ON \
       -DFORCE_VTK=ON \
       -DWITH_EIGEN=ON \
       -DWITH_V4L=ON \
       -DWITH_XINE=ON \
       -DWITH_GDAL=ON \
       -DWITH_1394=OFF \
       -DWITH_FFMPEG=OFF \
       -DBUILD_PROTOBUF=OFF \
       -DBUILD_TESTS=OFF \
       -DBUILD_PERF_TESTS=OFF \
       -DCMAKE_BUILD_TYPE=RELEASE \
       -DCMAKE_INSTALL_PREFIX=$OPENCV_INSTALL_PATH \
    .. \
    ##
    ## Add variable to enable make to use all cores
    && export NUMPROC=$(nproc --all) \
    && make -j$NUMPROC install \
    ## Remove following lines if you need to move openCv
    && rm /$OPENCV_VERSION.zip \
    && rm -r /opencv-$OPENCV_VERSION

## Compress the openCV files so you can extract them from the docker easily
RUN tar cvzf opencv-$OPENCV_VERSION.tar.gz --directory=$OPENCV_INSTALL_PATH .





#------------------------
# General setup
#------------------------
RUN apt-get update \
    && apt-get install -y lsb-release \
    wget


# To complete this step, please visit 'http://milq.github.io/install-opencv-ubuntu-debian'.
# #------------------------
# # Install python dependecies
# #------------------------

RUN apt-get update && apt-get install -q -y \
    python-pip \
    python-dev \
    build-essential

RUN pip install psutil
RUN pip install keras
RUN pip install keras-rl
RUN pip install tensorflow==1.5
RUN pip install pandas
RUN pip install sklearn
RUN pip install matplotlib
RUN pip install imageio
RUN pip install h5py

# #--------------------------
# #Set env
# #--------------------------
ENV PATH_P  '/app/'


# #--------------------------
# #Expose ports
# #--------------------------
# #notebook
EXPOSE 8888




# #--------------------------
# #Copy framework
# #--------------------------
# Set the working directory to /app
WORKDIR /app


