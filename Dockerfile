FROM ubuntu:20.04

RUN apt-get update &&   \
  apt-get install -y    \
    python3-pip         \
  && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --shell /bin/bash explorer --uid 1001
USER explorer
WORKDIR /home/explorer/

COPY --chown=explorer:explorer ./requirements.txt .
RUN pip3 install --user -r requirements.txt

RUN mkdir /home/explorer/server
WORKDIR /home/explorer/server
COPY --chown=explorer:explorer python_server/static static
COPY --chown=explorer:explorer python_server/templates templates
COPY --chown=explorer:explorer python_server/dbfiles dbfiles
COPY --chown=explorer:explorer python_server/server.py python_server/client.py ./

CMD [ "python3", "server.py" ]
