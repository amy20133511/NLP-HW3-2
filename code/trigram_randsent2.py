#!/usr/bin/env python3
"""
Computes the total log probability of the sequences of tokens in each file,
according to a given smoothed trigram model.
"""
import argparse
import logging
import math
from pathlib import Path
import torch

from probs import Wordtype, LanguageModel, num_tokens, read_trigrams

##### CONSTANTS
BOS: Wordtype = "BOS"  # special word type for context at Beginning Of Sequence
EOS: Wordtype = "EOS"  # special word type for observed token at End Of Sequence
OOV: Wordtype = "OOV"  # special word type for all Out-Of-Vocabulary words
OOL: Wordtype = "OOL"  # special word type whose embedding is used for OOV and all other Out-Of-Lexicon words

log = logging.getLogger(Path(__file__).stem)  # For usage, see findsim.py in earlier assignment.


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model",
        type=Path,
        help="path to the sampling model",
    )
    parser.add_argument(
        "n_of_samples",
        type=int,
        help="number of sampled sentences",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        help="max length of sampled sentence"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=['cpu', 'cuda', 'mps'],
        help="device to use for PyTorch (cpu or cuda, or mps if you are on a mac)"
    )

    # for verbosity of logging
    parser.set_defaults(logging_level=logging.INFO)
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument(
        "-v", "--verbose", dest="logging_level", action="store_const", const=logging.DEBUG
    )
    verbosity.add_argument(
        "-q", "--quiet", dest="logging_level", action="store_const", const=logging.WARNING
    )

    return parser.parse_args()

import random
import math
def sample_a_word(x: Wordtype, y:Wordtype, lm: LanguageModel):
    weights = list()
    for word in lm.vocab:
        weight = lm.log_prob(x, y, word)
        weight = math.exp(weight)
        weights.append(weight)
    z = random.choices(list(lm.vocab), weights=weights, k=1)
    return z

def sample_a_sentence(lm: LanguageModel, max_length):
    sentence = list()
    x = BOS
    y = BOS
    z = sample_a_word(x, y, lm)
    while z!=EOS and len(sentence)<=max_length:
        sentence.append(z)
        x = y
        y = z
        z = sample_a_word(x, y, lm)
    print(" ".join(sentence))
    return 0


def main():
    args = parse_args()
    logging.basicConfig(level=args.logging_level)

    # Specify hardware device where all tensors should be computed and
    # stored.  This will give errors unless you have such a device
    # (e.g., 'gpu' will work in a Kaggle Notebook where you have
    # turned on GPU acceleration).
    if args.device == 'mps':
        if not torch.backends.mps.is_available():
            if not torch.backends.mps.is_built():
                logging.critical("MPS not available because the current PyTorch install was not "
                                 "built with MPS enabled.")
            else:
                logging.critical("MPS not available because the current MacOS version is not 12.3+ "
                                 "and/or you do not have an MPS-enabled device on this machine.")
            exit(1)
    torch.set_default_device(args.device)

    log.info("Testing...")
    lm = LanguageModel.load(args.model, device=args.device)
    print(lm.vocab)
    print(lm.log_prob(BOS, BOS, 'finish'))
    print(OOV in lm.vocab)


if __name__ == "__main__":
    main()

