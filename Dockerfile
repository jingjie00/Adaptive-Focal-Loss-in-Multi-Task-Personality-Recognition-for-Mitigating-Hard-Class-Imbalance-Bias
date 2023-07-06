# Use a base image with Conda pre-installed
FROM continuumio/anaconda3

# Set the working directory inside the container
WORKDIR /app

# Copy the environment.yml file to the container
COPY requirements.txt .

# Create a Conda environment and activate it
RUN python -m pip install -r requirements.txt
RUN echo "source activate phdwork" > ~/.bashrc
ENV PATH /opt/conda/envs/phdwork/bin:$PATH


