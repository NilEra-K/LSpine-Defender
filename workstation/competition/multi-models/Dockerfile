# 避免被弃用 
# 1. nvidia/cuda:12.2.2-cudnn8-devel-ubuntu20.04 被弃用
# 2. nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04 未测试
# 3. nvidia/cuda:12.3.1-cudnn8-devel-ubuntu22.04 未被弃用
# FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu20.04
FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04
# FROM nvidia/cuda:12.3.1-cudnn8-devel-ubuntu22.04

RUN ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime

RUN apt-get update && apt-get install -y --no-install-recommends wget build-essential libreadline-dev \ 
    libncursesw5-dev libssl-dev libsqlite3-dev libgdbm-dev libbz2-dev liblzma-dev zlib1g-dev uuid-dev libffi-dev libdb-dev \
    libglib2.0-0 libgl1-mesa-glx git ffmpeg unzip

RUN wget --no-check-certificate https://mirrors.huaweicloud.com/python/3.10.13/Python-3.10.13.tgz \
    && tar -xf Python-3.10.13.tgz \
    && cd Python-3.10.13 \
    && ./configure --enable-optimizations \
    && make -j$(($(nproc) - 1)) \
    && make install

RUN apt-get autoremove -y

# [可选] 配置pip镜像源
# 不配置时需要使用 -i https://pypi.tuna.tsinghua.edu.cn/simple 参数
# RUN mkdir -p /root/.pip && \
#     echo "[global]" > /root/.pip/pip.conf && \
#     echo "index-url = https://pypi.tuna.tsinghua.edu.cn/simple" >> /root/.pip/pip.conf && \
#     echo "trusted-host = pypi.tuna.tsinghua.edu.cn" >> /root/.pip/pip.conf

WORKDIR /kaggle
ENV PYTHONPATH=/kaggle

COPY requirements.txt .

# 不配置镜像源时使用
# RUN python3 -m pip install --upgrade pip
# RUN pip3 install -r ./requirements.txt
RUN python3 -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade pip
RUN pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple -r ./requirements.txt