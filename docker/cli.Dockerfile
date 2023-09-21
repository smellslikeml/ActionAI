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


WORKDIR /home/dlstreamer
# Copy your Python script to the Docker image
COPY run_train.py /home/dlstreamer/


# Add a default command for this image
CMD ["python3", "run_train.py"]

