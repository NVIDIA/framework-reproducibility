#!/bin/bash
if [ $# -ne 1 ];then
   echo "Usage: $0 tag-of-container"
   exit
fi
DIR=${PWD##*/}

# --network=host is a necessary work-around to the internet not being
# accessible from within the container when the host is connected to VPN.

docker build --network=host -t framework-determinism/${DIR} . --build-arg base=$1
