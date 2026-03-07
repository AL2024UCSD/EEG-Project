FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1     PIP_NO_CACHE_DIR=1     PIP_DISABLE_PIP_VERSION_CHECK=1     MPLBACKEND=Agg

# System packages commonly needed for MNE / SciPy / scikit-learn / matplotlib
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    gcc \
    g++ \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    libhdf5-dev \
    libffi-dev \
    libglib2.0-0 \
    libgl1 \
    libxext6 \
    libxrender1 \
    libfreetype6-dev \
    libpng-dev \
    # enable Qt-backed interactive plots (safe to keep)
    libqt6gui6 \
    libqt6widgets6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Install Python deps first for better Docker layer caching
COPY requirements.txt ./
RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install -r requirements.txt

# Safety net: install common packages used across the project if they were missed in requirements.txt
# (keeps the container usable if requirements.txt lags behind code changes)
RUN python -m pip install \
    mne \
    mne-connectivity \
    numpy \
    scipy \
    pandas \
    matplotlib \
    seaborn \
    scikit-learn \
    statsmodels \
    tqdm \
    joblib \
    python-picard \
    pyedflib \
    jupyterlab \
    ipykernel \
    specparam \
    --no-deps

COPY . .

EXPOSE 8888

CMD ["bash"]