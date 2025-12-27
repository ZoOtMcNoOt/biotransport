#!/bin/bash
# dev.sh - Development helper script

set -e  # Exit immediately if a command exits with a non-zero status

usage() {
  echo "Usage: $0 {build|install|test|run example_path|clean}"
  echo "For examples: ./dev.sh run basic/1d_diffusion"
  echo "              ./dev.sh run intermediate/drug_diffusion_2d"
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
    cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5 ..
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

    # Check if the path exists directly
    if [ -f "examples/$2.py" ]; then
      python "examples/$2.py"
    # Check in subdirectories
    elif [ -f "examples/python/$2.py" ]; then
      python "examples/python/$2.py"
    elif [ -f "examples/basic/$2.py" ]; then
      python "examples/basic/$2.py"
    elif [ -f "examples/intermediate/$2.py" ]; then
      python "examples/intermediate/$2.py"
    elif [ -f "examples/advanced/$2.py" ]; then
      python "examples/advanced/$2.py"
    # Handle paths with subdirectory/filename format
    elif [[ "$2" == */* ]]; then
      if [ -f "examples/$2.py" ]; then
        python "examples/$2.py"
      elif [ -f "examples/python/$2.py" ]; then
        python "examples/python/$2.py"
      else
        echo "Error: Example file not found at examples/$2.py or examples/python/$2.py"
        exit 1
      fi
    else
      echo "Error: Example file not found."
      echo "Please specify the path relative to the examples directory:"
      echo "  ./dev.sh run basic/1d_diffusion"
      exit 1
    fi
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
