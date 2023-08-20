FROM ubuntu:latest

COPY . /TorcsBot

# Set environment variables
ENV DISPLAY=:99
ENV PYTHONPATH="${PYTHONPATH}:/TorcsBot"

# Install dependencies
RUN apt-get update && \
    apt-get install -y torcs python3-pip libglib2.0-dev libgl1-mesa-dev libglu1-mesa-dev freeglut3-dev libplib-dev libopenal-dev libalut-dev libxi-dev libxmu-dev libxrender-dev libxrandr-dev libpng-dev xautomation wget xvfb && \
    wget "http://nz.archive.ubuntu.com/ubuntu/pool/main/libx/libxxf86vm/libxxf86vm-dev_1.1.4-1build3_amd64.deb" && \
    dpkg -i libxxf86vm-dev_1.1.4-1build3_amd64.deb 

# Set working directory
WORKDIR /TorcsBot

# configure gym_torcs
RUN cd gym_torcs/vtorcs-RL-color/ && \
    chmod +x ./configure && \
    ./configure --x-libraries=/usr/lib/ && \
    make && \
    make install && \
    make datainstall

# # Install Python dependencies
# RUN pip3 install numpy torch torchvision torchaudio gym



# # Command to run the script
# CMD ["python", "TORCS_DDPG/test.py", "--device", "cpu"]
