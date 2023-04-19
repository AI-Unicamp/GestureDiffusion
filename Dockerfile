FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

RUN rm /etc/apt/sources.list.d/cuda.list

RUN apt-get update
RUN apt-get install -y wget git nano ffmpeg

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-py37_4.8.3-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-py37_4.8.3-Linux-x86_64.sh -b \
    && rm -f Miniconda3-py37_4.8.3-Linux-x86_64.sh

RUN conda --version

WORKDIR /root
COPY environment.yml /root

RUN conda install tqdm -f
RUN conda update conda
RUN conda install pip
RUN conda --version
RUN conda env create -f environment.yml
RUN pip install blobfile
RUN pip install PyYAML

SHELL ["conda", "run", "-n", "mdm", "/bin/bash", "-c"]
RUN python -m spacy download en_core_web_sm
RUN pip install git+https://github.com/openai/CLIP.git
