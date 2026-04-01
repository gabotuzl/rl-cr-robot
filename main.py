import argparse
from training.train import run_training
from training.test import run_testing

def parse_args():
    parser = argparse.ArgumentParser(description="Continuum Robot RL Controller")
    
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--scratch",      action="store_true", help="Train from scratch")
    mode.add_argument("--checkpoint",   action="store_true", help="Resume from latest checkpoint")
    mode.add_argument("--test",         action="store_true", help="Run evaluation")

    parser.add_argument("--checkpoint-path", type=str, default=None,
                        help="Specific checkpoint .zip to load (optional)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.scratch:
        run_training(from_scratch=True)
    elif args.checkpoint:
        run_training(from_scratch=False, checkpoint_path=args.checkpoint_path)
    elif args.test:
        run_testing(checkpoint_path=args.checkpoint_path)