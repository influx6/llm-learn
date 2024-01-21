FROM mcr.microsoft.com/devcontainers/python:1-3.10-bookworm

COPY ./.vscode/git.alias /home/vscode/.git.alias
COPY ./.vscode/keybindings.json /home/vscode/.vscode-server/data/Machine/keybindings.json
COPY ./.vscode/keybindings.json /home/vscode/.vscode-server/data/keybindings.json
COPY ./.vscode/keybindings.json /home/vscode/.vscode-server/data/User/keybindings.json

RUN echo 'source $HOME/.git.alias' >> /home/vscode/.bashrc \
    && chown -R vscode:1000 /home/vscode/.vscode-server