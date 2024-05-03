IMAGE_NAME="pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime"

# Container name
CONTAINER_NAME="style2405010748"

VOLUME_DIR="/home/safeai24/safe24/pytorch_GAN_zoo"
# port to be mapped (host_port:container_port)
# SSH_PORT="52121:22"
# JPT_PORT="48888:8888"

# Stop and remove the existing container with the same name, if it exists
echo "Checking if container named '$CONTAINER_NAME' already exists..."
if [ "$(docker ps -aq -f name=^${CONTAINER_NAME}$)" ]; then
    echo "Stopping and removing existing container..."
    docker stop $CONTAINER_NAME
    docker rm $CONTAINER_NAME
fi

# Run the Docker container
echo "Starting Docker container..."
docker run -it --gpus all --network=host --name "$CONTAINER_NAME" -v "$VOLUME_DIR:/app/pytorch_GAN_zoo"  "$IMAGE_NAME"

# After creation, run a command inside the container
docker ps -a|grep $CONTAINER_NAME

# docker exec -it $CONTAINER_NAME bash