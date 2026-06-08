import argparse
from enum import Enum

class LanguageModel(Enum):
    GEMMA = 1
    LLAMA = 2
    DEEPSEEK = 3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_steps', type=int, default=1000, help='Number of training steps')
    parser.add_argument('--checkpoint_freq', type=int, default=100, help='Checkpoint frequency (in steps)')
    parser.add_argument('--eval_freq', type=int, default=100, help='Evaluation frequency (in steps)')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden layer dimension')

    args = parser.parse_args()


if __name__ == '__main__':
    main()