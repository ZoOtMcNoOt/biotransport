#!/bin/bash
# dev.sh - Development helper script

set -e  # Exit immediately if a command exits with a non-zero status

usage() {
  echo "Usage: $0 {build|install|test|run example_name|clean}"
  exit 1
}

if [ $# -lt 1 ]; then
  usage
fi

case "$1" in
  build)
    # Optionally, customize parallel build jobs by setting JOBS variable, default is 2
    JOBS=${JOBS:-2}
    mkdir -p build && cd build
    cmake -DCMAKE_BUILD_TYPE=Release ..
    make -j${JOBS}
    ;;
  install)
    # Using python -m pip is more explicit
    python -m pip install -e .
    ;;
  test)
    pytest python/tests
    ;;
  run)
    if [ -z "$2" ]; then
      echo "Error: No example specified."
      usage
    fi
    python "examples/$2.py"
    ;;
  clean)
    echo "Cleaning build artifacts..."
    # Remove in-source generated files
    rm -f CMakeCache.txt cmake_install.cmake Makefile
    rm -rf CMakeFiles/
    rm -rf build/
    rm -rf python/biotransport.egg-info/
    ;;
  *)
    usage
    ;;
esac
