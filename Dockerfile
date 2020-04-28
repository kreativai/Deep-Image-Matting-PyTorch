FROM nvidia/cuda:10.1-cudnn7-devel AS production

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y \
	python3-opencv ca-certificates python3-dev git wget sudo && \
  rm -rf /var/lib/apt/lists/*

# create a non-root user
ARG USER_ID=1000
RUN useradd -m --no-log-init --system  --uid ${USER_ID} appuser -g sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER appuser
WORKDIR /home/appuser

ENV PATH="/home/appuser/.local/bin:${PATH}"
RUN wget https://bootstrap.pypa.io/get-pip.py && \
	python3 get-pip.py --user && \
	rm get-pip.py

# install dependencies
# See https://pytorch.org/ for other options if you use a different version of CUDA
RUN pip install --user torch torchvision tensorboard cython

ENV FORCE_CUDA="1"
# This will build detectron2 for all common cuda architectures and take a lot more time,
# because inside `docker build`, there is no way to tell which architecture will be used.
ENV TORCH_CUDA_ARCH_LIST="Kepler;Kepler+Tesla;Maxwell;Maxwell+Tegra;Pascal;Volta;Turing"

RUN pip install --user pillow gunicorn flask gevent scipy jupyter matplotlib jupyter_http_over_ws
RUN jupyter serverextension enable --py jupyter_http_over_ws

COPY ./submodules /submodules
COPY ./apilib /customlibs/apilib

WORKDIR /src

#CMD sh -c "ls /weights && ls /src && ls /submodules && cd /src && export CUDA_DEVICE=0 && gunicorn app:app -b 0.0.0.0:5000 -k gevent --worker-connections 1000 --timeout 90"
CMD sh -c "export CUDA_DEVICE=0 && export FLASK_ENV=development && export LANG=C.UTF-8 && export LC_ALL=C.UTF-8 && flask run --host=0.0.0.0 --without-threads"

FROM production AS dev

#CMD ["jupyter-notebook", "--allow-root" ,"--port=8888" ,"--no-browser" ,"--ip=0.0.0.0"]
CMD sh -c "export CUDA_DEVICE=0 && export FLASK_ENV=development && export LANG=C.UTF-8 && export LC_ALL=C.UTF-8 && flask run --host=0.0.0.0 --without-threads"
