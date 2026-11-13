"""Support python -m federated_mcts via the same installed CLI."""
from federated_mcts.cli import run_main

if __name__ == "__main__":
    raise SystemExit(run_main())
