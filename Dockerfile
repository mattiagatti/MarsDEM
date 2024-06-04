FROM python:3.11-slim-buster
LABEL org.opencontainers.image.source=https://github.com/mattiagatti/MarsDEM
WORKDIR /python-docker
COPY . /python-docker/
RUN apt-get update && apt-get install --no-install-recommends -y \
    libegl1 \
    libgl1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*
RUN pip install -r app_requirements.txt
EXPOSE 7860
ENV GRADIO_SERVER_NAME="0.0.0.0"
CMD [ "python", "app.py"]