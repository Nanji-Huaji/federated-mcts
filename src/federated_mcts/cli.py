"""The installed command-line interface; importing it performs no initialization."""
from __future__ import annotations


def run_main(argv=None) -> int:
    from federated_mcts.evaluation.game24_runner_args import parse_args
    args = parse_args(argv)
    from federated_mcts.env_loader import load_dotenv
    from federated_mcts.experiments.runner import run
    load_dotenv()
    try:
        run(args, args.solve_method)
    except (ValueError, OSError) as error:
        import sys
        print(f"FATAL: {error}", file=sys.stderr)
        return 1
    return 0


def exp_main() -> None:
    from federated_mcts.experiments.sweep import main
    main()
