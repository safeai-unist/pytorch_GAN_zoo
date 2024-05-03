# 기본 이미지 설정
FROM continuumio/miniconda3:latest

# 파이썬 버전 설정
ARG PYTHON_VERSION=3.9

# 환경변수 설정
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

# Miniconda 설치
RUN apt-get update --fix-missing && \
    apt-get install -y wget bzip2 ca-certificates curl git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    /bin/bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

# MindSpore 설치
RUN /opt/conda/bin/conda install -y python=${PYTHON_VERSION} && \
    /opt/conda/bin/pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.3.0rc1/MindSpore/unified/x86_64/mindspore-2.3.0rc1-cp${PYTHON_VERSION}-cp${PYTHON_VERSION}-linux_x86_64.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple

# 작업 디렉토리 설정
WORKDIR /app

# 호스트 디렉토리 마운트
VOLUME /home/safeai24/safe24/pytorch_GAN_zoo:/app/pytorch_GAN_zoo

# 필요한 추가 작업이 있으면 여기에 추가하세요.

# 컨테이너 실행 시 실행될 명령 설정
# CMD ["/bin/bash"]
