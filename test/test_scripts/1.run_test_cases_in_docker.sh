current_dir=$(pwd)

# makesure we are running in test/test_scripts/
case "$current_dir" in
  */test/test_scripts)
    # Current directory is correct, continue
    ;;
  *)
    echo "Error: This script must be run from the test/test_scripts/ directory."
    echo "Current directory: $current_dir"
    exit 1
    ;;
esac

# makesure docker is installed
echo "\n >>> Checking if docker is installed..."
sh ../../scripts/makesure_docker.sh

# makesure config.yml exists
echo "\n >>> Checking if config.yml exists..."
if [ ! -f "config.yml" ]; then
    echo "Error: config.yml does not exist."
    exit 1
fi

# read TEST_IMG_PATH from config.yml
TEST_IMG_PATH=$(grep -E '^TEST_IMG_PATH:' config.yml | cut -d ':' -f 2 | tr -d '[:space:]')

# docker pull
echo "\n >>> Pulling docker image $TEST_IMG_PATH"
docker pull $TEST_IMG_PATH

# Determine Project Root
# Assuming this script is in project_root/test/test_scripts/
PROJECT_ROOT=$(cd "$current_dir/../.." && pwd)
echo "\n >>> Mounting project root: $PROJECT_ROOT to /workspace in the container."

# start docker container
# Mount the project directory to /workspace in the container
echo "\n >>> Starting docker container teletron-test with image $TEST_IMG_PATH"
docker run --gpus all -itd \
           --shm-size 512G \
           --name teletron-test \
           -v "$PROJECT_ROOT":/workspace \
           $TEST_IMG_PATH /bin/bash

echo "\n >>> Setup container environment..."
docker exec -it -w /workspace teletron-test python3 scripts/setup_env.py


echo "\n >>> Running sanity check"
docker exec -it -w /workspace teletron-test python3 scripts/sanity_check.py

echo "\n >>> Running test cases"
docker exec -it -w /workspace teletron-test python3 test/test_cases/run_test_cases.py

# stop docker container
echo "\n >>> Stopping docker container..."
docker stop teletron-test

# remove docker container
echo "\n >>> Removing docker container..."
docker rm teletron-test

echo "\n >>> Test script finished."

