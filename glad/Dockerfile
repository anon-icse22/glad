FROM nvidia/cuda:10.2-cudnn8-devel

# install python3.9
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update
RUN apt-get install -y software-properties-common
RUN add-apt-repository -y ppa:deadsnakes/ppa
RUN apt-get update
RUN apt-get install -y python3.9

# install defects4j
RUN apt-get install -y openjdk-8-jdk
ENV JAVA_HOME /usr/lib/jvm/java-8-openjdk-amd64
RUN apt-get install -y build-essential git cpanminus make curl unzip perl subversion
RUN cd /usr/local/src && git clone https://github.com/rjust/defects4j && cd defects4j && cpanm --installdeps . && ./init.sh
ENV PATH="${PATH}:/usr/local/src/defects4j/framework/bin"

# install pip3.9
RUN cd /tmp && apt-get install -y python3.9-distutils && curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && python3.9 get-pip.py
RUN mkdir -p /home/java-pred-synth
COPY ./requirements.txt /home/java-pred-synth/requirements.txt
RUN pip3.9 install -r /home/java-pred-synth/requirements.txt

# load whole directory
COPY . /home/java-pred-synth
RUN mkdir -p /home/java-pred-synth/data/defects4j-buggy-projects
RUN mkdir -p /home/java-pred-synth/models/weights/
RUN mkdir -p /home/java-pred-synth/data/finetune
RUN mkdir /home/java-pred-synth/logs

WORKDIR /home/java-pred-synth

