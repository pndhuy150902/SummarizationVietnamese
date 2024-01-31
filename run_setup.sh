#!/bin/sh
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
      echo -e "Invalid command option.\nUsage: $(basename $0) [-a] [-b] [-c arg]"
      exit 1
      ;;
  esac
done
shift "$(($OPTIND -1))"