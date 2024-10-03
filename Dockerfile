FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y \
        libglib2.0-0 \
        libgl1-mesa-glx \
        git \
        vim \
        g++ \
        sudo \
        xvfb \
        libxrender-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ARG USERNAME=whikwon
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    #
    # [Optional] Add sudo support. Omit if you don't need to install software after connecting.
    && apt-get update \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# ********************************************************
# * Anything else you want to do like clean up goes here *
# ********************************************************

RUN groupmod --gid $USER_GID $USERNAME \
    && usermod --uid $USER_UID --gid $USER_GID $USERNAME \
    && chown -R $USER_UID:$USER_GID /opt/conda

# [Optional] Set the default user. Omit if you want to keep the default as root.
USER $USERNAME

RUN echo 'export PATH="$HOME/.local/bin:$PATH"' > $HOME/.bashrc

RUN pip install omegaconf torchmetrics==0.10.3 fvcore iopath xformers==0.0.18 submitit 

WORKDIR /workspaces