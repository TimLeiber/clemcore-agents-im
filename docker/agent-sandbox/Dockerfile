FROM python:3.11-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV CHROME_BIN=/usr/bin/chromium
ENV PUPPETEER_EXECUTABLE_PATH=/usr/bin/chromium
ENV PLAYWRIGHT_BROWSERS_PATH=/ms-playwright

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    zsh \
    coreutils \
    findutils \
    grep \
    sed \
    gawk \
    diffutils \
    patch \
    tar \
    gzip \
    unzip \
    zip \
    xz-utils \
    bzip2 \
    curl \
    wget \
    git \
    git-lfs \
    build-essential \
    make \
    cmake \
    pkg-config \
    ca-certificates \
    openssl \
    gnupg \
    dnsutils \
    iputils-ping \
    netcat-openbsd \
    procps \
    psmisc \
    lsof \
    jq \
    sqlite3 \
    ripgrep \
    fd-find \
    file \
    less \
    tree \
    vim-tiny \
    nano \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    chromium \
    xvfb \
    wamerican \
    wordnet \
    aspell \
    aspell-en \
    hunspell \
    hunspell-en-us \
    && ln -sf /usr/bin/fdfind /usr/local/bin/fd \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /etc/apt/keyrings && \
    curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key \
      | gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg && \
    echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_22.x nodistro main" \
      > /etc/apt/sources.list.d/nodesource.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends nodejs && \
    rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel && \
    python -m pip install --no-cache-dir \
        requests \
        httpx \
        beautifulsoup4 \
        lxml \
        html5lib \
        trafilatura \
        readability-lxml \
        pandas \
        numpy \
        scipy \
        sympy \
        networkx \
        python-dateutil \
        rapidfuzz \
        duckdb \
        pyarrow \
        regex \
        tqdm \
        pyyaml \
        matplotlib \
        pillow \
        nltk \
        wordfreq \
        wordninja

RUN npm install -g @anthropic-ai/claude-code @openai/codex openclaw@latest

COPY pyproject.toml /tmp/clemcore-pyproject.toml

RUN mkdir -p /tmp/clemcore-build && \
    cp /tmp/clemcore-pyproject.toml /tmp/clemcore-build/pyproject.toml && \
    touch /tmp/clemcore-build/README.md && \
    mkdir -p /tmp/clemcore-build/clemcore && \
    touch /tmp/clemcore-build/clemcore/__init__.py && \
    pip install --no-cache-dir /tmp/clemcore-build && \
    pip install --no-cache-dir \
        claude-agent-sdk \
        fastmcp \
        openai-codex \
        "hermes-agent[cli,mcp]"

RUN useradd -m -u 1000 agent && \
    mkdir -p /workspace /ms-playwright && \
    chown -R agent:agent /workspace /ms-playwright

USER agent
WORKDIR /workspace

ENV PYTHONPATH=/opt/clemcore

CMD ["bash"]