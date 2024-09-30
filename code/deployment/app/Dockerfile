# Use Python slim image
FROM python:3.10.12-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the current directory into the container
COPY . /app

# Install required dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir gradio requests pyyaml numpy scipy

# Expose the port where Gradio app will run
EXPOSE 5155

# Command to run the Gradio app
CMD ["python", "-u", "/app/app.py"]
