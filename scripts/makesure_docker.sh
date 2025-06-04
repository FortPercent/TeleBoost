#!/bin/sh

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Define SUDO_CMD based on user privileges
SUDO_CMD=""
if [ "$(id -u)" -eq 0 ]; then
  echo "Running as root. 'sudo' prefix will be omitted for privileged commands."
else
  if ! command_exists sudo; then
    echo "Error: 'sudo' command not found, but it's needed to perform installations as a non-root user."
    echo "Please install 'sudo' or run this script as root."
    exit 1
  fi
  SUDO_CMD="sudo"
  echo "Not running as root. Using 'sudo' for privileged commands."
fi

# Check if Docker is installed
if command_exists docker; then
    echo "Docker is already installed."
    docker --version
    exit 0
fi

echo "Docker is not installed. Attempting to install..."

# Check for apt (Debian/Ubuntu)
if command_exists apt; then
    echo "Using apt to install Docker..."
    $SUDO_CMD apt update
    $SUDO_CMD apt install -y apt-transport-https ca-certificates curl software-properties-common
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | $SUDO_CMD apt-key add -
    $SUDO_CMD add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
    $SUDO_CMD apt update
    $SUDO_CMD apt install -y docker-ce
    echo "Docker installation via apt attempted."
    # Add current user to docker group to run docker without sudo (optional, requires logout/login)
    # $SUDO_CMD usermod -aG docker ${USER}
    # echo "Please log out and log back in for group changes to take effect, or run 'newgrp docker' in your current shell."

# Check for yum (RHEL/CentOS)
elif command_exists yum; then
    echo "Using yum to install Docker..."
    $SUDO_CMD yum install -y yum-utils
    $SUDO_CMD yum-config-manager --add-repo https://download.docker.com/linux/centos/docker-ce.repo
    $SUDO_CMD yum install -y docker-ce docker-ce-cli containerd.io
    $SUDO_CMD systemctl start docker
    $SUDO_CMD systemctl enable docker
    echo "Docker installation via yum attempted."
    # Add current user to docker group to run docker without sudo (optional, requires logout/login)
    # $SUDO_CMD usermod -aG docker ${USER}
    # echo "Please log out and log back in for group changes to take effect, or run 'newgrp docker' in your current shell."

else
    echo "Neither apt nor yum found. Cannot automatically install Docker."
    echo "Please install Docker manually for your system."
    exit 1
fi

# Verify installation
if command_exists docker; then
    echo "Docker installed successfully."
    docker --version
    # You might want to start and enable the docker service if not done by the installer
    if command_exists systemctl && ! systemctl is-active --quiet docker; then
        echo "Attempting to start and enable Docker service..."
        $SUDO_CMD systemctl start docker
        $SUDO_CMD systemctl enable docker
        if systemctl is-active --quiet docker; then
            echo "Docker service started and enabled."
        else
            echo "Failed to start Docker service. Please check manually."
        fi
    fi
else
    echo "Docker installation failed. Please check the output above for errors."
    exit 1
fi

exit 0
