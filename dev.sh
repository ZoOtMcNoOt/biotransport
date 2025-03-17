#!/bin/bash
# dev.sh - Development helper script

case "$1" in
  build)
    mkdir -p build && cd build && cmake .. && make -j
    ;;
  install)
    pip install -e .
    ;;
  test)
    pytest python/tests
    ;;
  run)
    python examples/$2.py
    ;;
  *)
    echo "Usage: $0 {build|install|test|run example_name}"
    exit 1
esac