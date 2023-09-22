# Use the base image from Intel dlstreamer
FROM intel/dlstreamer:devel

USER root

# Set environment variables for display and user
ENV DISPLAY=$DISPLAY

# Install packages
RUN apt-get update -y && \
    apt-get install vim -y && \
    apt-get install git -y && \
    pip install scikit-learn && \
    python3 -m pip install --upgrade pip

# Download models (This is optional if you need it)
WORKDIR /opt/intel/dlstreamer/samples
COPY models.lst .
RUN ./download_models.sh

# Clone and install ActionAI from the cli branch
RUN git clone -b actionai-cli https://github.com/smellslikeml/ActionAI.git /home/dlstreamer/ActionAI
RUN pip install /home/dlstreamer/ActionAI

WORKDIR /home/dlstreamer
COPY entrypoint.sh /entrypoint.sh
# Make the entrypoint script executable
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
