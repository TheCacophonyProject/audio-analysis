FROM tensorflow/tensorflow:2.11.0

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


RUN pip install gdown
RUN mkdir /models/bird-model -p
RUN gdown --fuzzy "https://drive.google.com/file/d/1vx_KARUfboUHn95JngZRFT_wFupvCabj/view?usp=sharing" -O bird-model.tar
RUN tar xzvf bird-model.tar -C /models/bird-model --strip-components=1
RUN rm bird-model.tar

RUN mkdir /models/morepork-model -p

RUN gdown --fuzzy "https://drive.google.com/file/d/1M3rb49f-yIWxCchZtX5QYhbN4tZ7qkD9/view?usp=sharing" -O morepork-model.tar
RUN tar xzvf morepork-model.tar -C /models/morepork-model --strip-components=1
RUN rm morepork-model.tar

COPY Melt /Melt
ENTRYPOINT ["python3","/Melt/chain.py"]
