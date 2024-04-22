#!/bin/sh
apt-get install libaio-dev mpich vim git-lfs
CMAKE_ARGS="-DLLAMA_CUDA=on" FORCE_CMAKE=1 pip install llama-cpp-python
while getopts ':pc:h' opt; do
  case "$opt" in
    p)
      python setup.py --pip
      ;;
    c)
      python setup.py --conda
      ;;
    h)
      echo "Usage: $(basename $0) [-p] [-c]"
      exit 0
      ;;
    :)
      echo -e "option requires an argument.\nUsage: $(basename $0) [-p] [-c]"
      exit 1
      ;;
    ?)
      echo -e "Invalid command option.\nUsage: $(basename $0) [-p] [-c]"
      exit 1
      ;;
  esac
done
shift "$(($OPTIND -1))"