# Start from minimal Ubuntu image that already includes ffmpeg.
# Ubuntu's ffmpeg package pulls in over 400MB of dependencies.
FROM jrottenberg/ffmpeg:3.4-ubuntu

RUN apt-get update && \
  apt-get install -qyy \
    -o APT::Install-Recommends=false -o APT::Install-Suggests=false \
    python3 python3-distutils curl ca-certificates libsndfile1

#for opencv
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

RUN curl -q https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
  python3 get-pip.py --quiet --no-cache-dir

RUN apt-get purge -y curl && apt-get autoremove -y && apt-get clean

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

WORKDIR /
RUN rm -rf /tmp/workdir

COPY Melt /Melt
ENTRYPOINT ["python3","/Melt/chain.py"]
