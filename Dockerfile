FROM continuumio/miniconda3:latest
LABEL authors="grant"
# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    ninja-build \
    ffmpeg \
    vim \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set up the working directory
WORKDIR /biotransport

# Create conda environment
COPY environment.yml /biotransport/
RUN conda env create -f environment.yml

# Make RUN commands use the new environment by default
SHELL ["conda", "run", "-n", "biotransport", "/bin/bash", "-c"]

# Initialize the environment for interactive use
RUN echo "conda activate biotransport" >> ~/.bashrc

# Entry point to start in bash with conda environment active
ENTRYPOINT ["/bin/bash", "-c", "source ~/.bashrc && exec bash"]
