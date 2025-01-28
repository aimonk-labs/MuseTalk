FROM musetalk_aimonk:v1

# Set the working directory inside the container to /musetalk
WORKDIR /musetalk

# Copy the current directory's contents into /musetalk
COPY . /musetalk

# Install system dependencies (add rclone if not already in the base image)
RUN apt-get update && apt-get install -y rclone && \
    rm -rf /var/lib/apt/lists/*

# Initialize conda for non-interactive shells and install required Python packages
RUN /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && conda activate musev && \
    pip3 install diffusers fastapi"

# Configure rclone (optional: if you have a pre-configured rclone.conf file)
# Replace 'rclone.conf' with your actual configuration file
# COPY rclone.conf /root/.config/rclone/rclone.conf

# Expose port 8080 for FastAPI
EXPOSE 8080

# Add a health check to ensure the server is running
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 CMD curl -f http://localhost:8080/health || exit 1

# Command to activate conda environment and run the FastAPI server on port 8080
CMD ["/bin/bash", "-c", "source /opt/conda/etc/profile.d/conda.sh && conda activate musev && uvicorn main_server:app --reload --host 0.0.0.0 --port 8080"]