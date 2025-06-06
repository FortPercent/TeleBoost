import os

def os_system(cmd):
    print(f"Running command: {cmd}")
    ret=os.system(cmd)
    if ret != 0:
        print(f"Error: Command {cmd} failed with return code {ret}")
    else:
        print(f"Command {cmd} succeeded")

# chdir to file directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# install dependencies
piplist=[
    "setuptools==78.1.0",
    "yunchang==0.6.0",
    "omegaconf",
    "pyyaml"
]
for pkg in piplist:
    os_system(f"pip3 install {pkg}")

