FROM tensorflow/tensorflow:2.19.0

ARG MODEL_VERSION="0.1"


RUN apt-get update && apt-get install ffmpeg -y

RUN apt-get update && \
  apt-get install -qyy \
    -o APT::Install-Recommends=false -o APT::Install-Suggests=false \
    python3 python3-distutils curl ca-certificates libsndfile1
#for opencv
RUN apt-get update
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Pacific/Auckland
RUN apt-get install -y python3-opencv
RUN curl -q https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
  python3 get-pip.py --quiet --no-cache-dir

RUN apt-get purge -y curl && apt-get autoremove -y && apt-get clean

COPY requirements.txt .
#using tf image dont need to install it
RUN sed "s/tensorflow~=*/#tensorflow~=/" requirements.txt -i

RUN pip3 install --no-cache-dir -r requirements.txt

WORKDIR /
RUN rm -rf /tmp/workdir
RUN mkdir /models/bird-model -p
RUN wget "https://github.com/TheCacophonyProject/AI-Model/releases/download/audio-v$MODEL_VERSION/audiomodel.tar"
RUN tar xzvf audiomodel.tar -C /models/bird-model --strip-components=1

COPY Melt /Melt
ENTRYPOINT ["python3","/Melt/chain.py"]
