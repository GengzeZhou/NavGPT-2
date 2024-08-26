# Matterport3DSimulator
# 2024, by gengze.zhou@adelaide.edu.au
# Requires nvidia gpu with driver 396.37 or higher

FROM ubuntu:22.04

# timezone
RUN apt update && apt install -y tzdata; \
    apt clean;

# sshd
RUN mkdir /run/sshd; \
    apt install -y openssh-server; \
    sed -i 's/^#\(PermitRootLogin\) .*/\1 yes/' /etc/ssh/sshd_config; \
    sed -i 's/^\(UsePAM yes\)/# \1/' /etc/ssh/sshd_config; \
    apt clean;

# entrypoint
RUN { \
    echo '#!/bin/bash -eu'; \
    echo 'ln -fs /usr/share/zoneinfo/${TZ} /etc/localtime'; \
    echo 'echo "root:${ROOT_PASSWORD}" | chpasswd'; \
    echo 'exec "$@"'; \
    } > /usr/local/bin/entry_point.sh; \
    chmod +x /usr/local/bin/entry_point.sh;

# Set the default working directory inside the container
WORKDIR /root

# Install a few libraries to support both EGL and OSMESA options
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y \
    git \
    wget \
    curl \
    unzip \
    doxygen \
    libjsoncpp-dev \
    libepoxy-dev \
    libglm-dev \
    libosmesa6 \
    libosmesa6-dev \
    libglew-dev \
    libgl1-mesa-dev \
    build-essential

#install cmake
ADD https://cmake.org/files/v3.12/cmake-3.12.2-Linux-x86_64.sh /cmake-3.12.2-Linux-x86_64.sh
RUN mkdir /opt/cmake
RUN sh /cmake-3.12.2-Linux-x86_64.sh --prefix=/opt/cmake --skip-license
RUN ln -s /opt/cmake/bin/cmake /usr/local/bin/cmake
RUN cmake --version

# Install opencv
# Download and unpack sources
RUN wget -O opencv.zip https://github.com/opencv/opencv/archive/3.4.15.zip && \
    unzip opencv.zip && \
    rm opencv.zip && \
    cd opencv-3.4.15 && \
    mkdir build && cd build && \
    cmake .. && \
    cmake --build .
ENV OpenCV_DIR /root/opencv-3.4.15/build
RUN echo "export OpenCV_DIR=/root/opencv-3.4.15/build" >> ~/.bashrc

# Download Miniconda installer script, run it, and clean up afterwards
# This installs Miniconda under /root/miniconda3
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
RUN /bin/bash ~/miniconda.sh -b -p /root/miniconda3
RUN rm ~/miniconda.sh

# Add the Conda installation to PATH for easier command execution
ENV PATH /root/miniconda3/bin:$PATH
RUN echo ' # >>> conda initialize >>>' >> ~/.bashrc
RUN echo ' # !! Contents within this block are managed by 'conda init' !!' >> ~/.bashrc
RUN echo ' __conda_setup="$('/root/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"' >> ~/.bashrc
RUN echo ' if [ $? -eq 0 ]; then' >> ~/.bashrc
RUN echo '     eval "$__conda_setup"' >> ~/.bashrc
RUN echo ' else' >> ~/.bashrc
RUN echo '     if [ -f "/root/miniconda3/etc/profile.d/conda.sh" ]; then' >> ~/.bashrc
RUN echo '         . "/root/miniconda3/etc/profile.d/conda.sh"' >> ~/.bashrc
RUN echo '     else' >> ~/.bashrc
RUN echo '         export PATH="/root/miniconda3/bin:$PATH"' >> ~/.bashrc
RUN echo '     fi' >> ~/.bashrc
RUN echo ' fi' >> ~/.bashrc
RUN echo ' unset __conda_setup' >> ~/.bashrc
RUN echo ' # <<< conda initialize <<<' >> ~/.bashrc

# Create a new Conda environment named "navgpt" with Python 3.8
RUN conda create -n navgpt python=3.8
RUN /root/miniconda3/bin/conda run -n navgpt python --version

# Configure shell to make RUN commands use the new environment by default
SHELL ["conda", "run", "-n", "navgpt", "/bin/bash", "-c"]

# Activate the Conda environment by default when starting a shell session
RUN echo "source activate navgpt" >> ~/.bashrc

# Install the required Python packages
RUN conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=11.8 -c pytorch -c nvidia
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

RUN apt-get install -y pkg-config
# Clone the Matterport3DSimulator repository
RUN cd /root && \
    git clone --recursive https://github.com/GengzeZhou/Matterport3DSimulator-revised.git && \
    mv Matterport3DSimulator-revised Matterport3DSimulator && \
    cd Matterport3DSimulator && \
    git checkout opencv3.x-ubuntu20.04+ && \
    mkdir build && cd build && \
    cmake -DEGL_RENDERING=ON .. && \
    make -j8

# Add the Matterport3DSimulator Python API to the Python path
ENV PYTHONPATH /root/Matterport3DSimulator/build
RUN echo "export PYTHONPATH=/root/Matterport3DSimulator/build" >> ~/.bashrc

ENV TZ Australia/Adelaide

ENV ROOT_PASSWORD root

EXPOSE 22

ENTRYPOINT ["entry_point.sh"]
CMD    ["/usr/sbin/sshd", "-D", "-e"]