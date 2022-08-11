FROM nvcr.io/nvidia/tensorflow:22.04-tf2-py3 as BASE
ENV DEBIAN_FRONTEND=noninteractive

RUN mkdir /app && mkdir /app/source 
COPY ./* /app/source/

RUN apt-get update -y && \
    apt-get install -y ffmpeg libsm6 libxext6 git python3 python3-dev python3-pip && \
    apt-get clean -qy && \
    useradd -ms /bin/bash user

USER user 
WORKDIR /app/source
RUN ls -al && python3 -m pip install -r requirements.txt

EXPOSE 5555
CMD ["flask", "--app", "api/app.py", "run", "-b", "192.168.0.53:5555"]
