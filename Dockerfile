FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

# System packages required for SciPy, MNE, Matplotlib, h5py, other dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    gcc \
    g++ \
    libhdf5-dev \
    libglib2.0-0 \
    libxkbcommon-x11-0 \
    libxcb-cursor0 \
    libfreetype6-dev \
    libpng-dev \
    libqt6gui6 \
    libqt6widgets6 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /EEG-Project

# Install Python requirements
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Default command (interactive shell)
CMD ["bash"]