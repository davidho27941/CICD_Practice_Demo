FROM nvcr.io/nvidia/tensorflow:22.04-tf2-py3 as BASE
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y && \
    apt-get install -y ffmpeg libsm6 libxext6 && \
    apt-get clean -qy && \
    useradd -ms /bin/bash user 

EXPOSE 5555

USER user 
RUN python -m pip install -r requirements.txt
RUN python main.py --stage deploy