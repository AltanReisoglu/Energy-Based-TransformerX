#!/usr/bin/env bash
set -euo pipefail

# Simple entrypoint script to run common commands inside the container.
# Usage: docker run ... <command>
# Commands: help | train | run | jupyter | bash

CMD=${1:-help}
shift || true

case "$CMD" in
  help)
    echo "Usage: $0 {help|train|run|jupyter|bash}"
    echo "  help    - show this message"
    echo "  train   - run training script (nlp/train_ebt.py)"
    echo "  run     - run main.py"
    echo "  jupyter - start jupyter notebook"
    echo "  bash    - start an interactive shell"
    ;;
  train)
    echo "Starting training..."
    exec python nlp/train_ebt.py "$@"
    ;;
  run)
    echo "Running main.py..."
    exec python main.py "$@"
    ;;
  jupyter)
    echo "Starting Jupyter Notebook..."
    exec jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root "$@"
    ;;
  bash)
    exec /bin/bash
    ;;
  *)
    echo "Unknown command: $CMD"
    exit 2
    ;;
esac
