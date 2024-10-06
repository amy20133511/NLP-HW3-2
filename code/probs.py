#!/usr/bin/env python3
# CS465 at Johns Hopkins University.
# Module to estimate n-gram probabilities.

# Updated by Jason Baldridge <jbaldrid@mail.utexas.edu> for use in NLP
# course at UT Austin. (9/9/2008)

# Modified by Mozhi Zhang <mzhang29@jhu.edu> to add the new log linear model
# with word embeddings.  (2/17/2016)

# Refactored by Arya McCarthy <xkcd@jhu.edu> because inheritance is cool
# and so is separating business logic from other stuff.  (9/19/2019)

# Patched by Arya McCarthy <arya@jhu.edu> to fix a counting issue that
# evidently was known pre-2016 but then stopped being handled?

# Further refactoring by Jason Eisner <jason@cs.jhu.edu> 
# and Brian Lu <zlu39@jhu.edu>.  (9/26/2021)

from __future__ import annotations

import logging
import math
import sys

from pathlib import Path

import torch
from torch import nn
from torch import optim
from jaxtyping import Float
from typeguard import typechecked
from typing import Counter
from collections import Counter
import pickle
from integerize import Integerizer   # look at integerize.py for more info
from tqdm import tqdm

log = logging.getLogger(Path(__file__).stem)  # For usage, see findsim.py in earlier assignment.

##### TYPE DEFINITIONS (USED FOR TYPE ANNOTATIONS)
from typing import Iterable, List, Optional, Set, Tuple, Union

Wordtype = str  # if you decide to integerize the word types, then change this to int
Vocab    = Set[Wordtype]
Zerogram = Tuple[()]
Unigram  = Tuple[Wordtype]
Bigram   = Tuple[Wordtype, Wordtype]
Trigram  = Tuple[Wordtype, Wordtype, Wordtype]
Ngram    = Union[Zerogram, Unigram, Bigram, Trigram]
Vector   = List[float]
TorchScalar = Float[torch.Tensor, ""] # a torch.Tensor with no dimensions, i.e., a scalar


##### CONSTANTS
BOS: Wordtype = "BOS"  # special word type for context at Beginning Of Sequence
EOS: Wordtype = "EOS"  # special word type for observed token at End Of Sequence
OOV: Wordtype = "OOV"  # special word type for all Out-Of-Vocabulary words
OOL: Wordtype = "OOL"  # special word type whose embedding is used for OOV and all other Out-Of-Lexicon words


##### UTILITY FUNCTIONS FOR CORPUS TOKENIZATION

def read_tokens(file: Path, vocab: Optional[Vocab] = None) -> Iterable[Wordtype]:
    """Iterator over the tokens in file.  Tokens are whitespace-delimited.
    If vocab is given, then tokens that are not in vocab are replaced with OOV."""

    # OPTIONAL SPEEDUP: You may want to modify this to integerize the
    # tokens, using integerizer.py as in previous homeworks.
    # In that case, redefine `Wordtype` from `str` to `int`.

    # PYTHON NOTE: This function uses `yield` to return the tokens one at
    # a time, rather than constructing the whole sequence and using
    # `return` to return it.
    #
    # A function that uses `yield` is called a "generator."  As with other
    # iterators, it computes new values only as needed.  The sequence is
    # never fully constructed as an single object in memory.
    #
    # You can iterate over the yielded sequence, for example, like this:
    #      for token in read_tokens(my_file, vocab):
    #          process(token)
    # Whenever the `for` loop needs another token, read_tokens magically picks up 
    # where it left off and continues running until the next `yield` statement.

    with open(file) as f:
        for line in f:
            for token in line.split():
                if vocab is None or token in vocab:
                    yield token
                else:
                    yield OOV  # replace this out-of-vocabulary word with OOV
            yield EOS  # Every line in the file implicitly ends with EOS.


def num_tokens(file: Path) -> int:
    """Give the number of tokens in file, including EOS."""
    return sum(1 for _ in read_tokens(file))


def read_trigrams(file: Path, vocab: Vocab) -> Iterable[Trigram]:
    """Iterator over the trigrams in file.  Each triple (x,y,z) is a token z
    (possibly EOS) with a left context (x,y)."""
    x, y = BOS, BOS
    for z in read_tokens(file, vocab):
        yield (x, y, z)
        if z == EOS:
            x, y = BOS, BOS  # reset for the next sequence in the file (if any)
        else:
            x, y = y, z  # shift over by one position.


def draw_trigrams_forever(file: Path, 
                          vocab: Vocab, 
                          randomize: bool = False) -> Iterable[Trigram]:
    """Infinite iterator over trigrams drawn from file.  We iterate over
    all the trigrams, then do it again ad infinitum.  This is useful for 
    SGD training.  
    
    If randomize is True, then randomize the order of the trigrams each time.  
    This is more in the spirit of SGD, but the randomness makes the code harder to debug, 
    and forces us to keep all the trigrams in memory at once.
    """
    trigrams = read_trigrams(file, vocab)
    if not randomize:
        import itertools
        return itertools.cycle(trigrams)  # repeat forever
    else:
        import random
        pool = tuple(trigrams)   
        while True:
            for trigram in random.sample(pool, len(pool)):
                yield trigram

##### READ IN A VOCABULARY (e.g., from a file created by build_vocab.py)

def read_vocab(vocab_file: Path) -> Vocab:
    vocab: Vocab = set()
    with open(vocab_file, "rt") as f:
        for line in f:
            word = line.strip()
            vocab.add(word)
    log.info(f"Read vocab of size {len(vocab)} from {vocab_file}")
    return vocab

##### LANGUAGE MODEL PARENT CLASS

class LanguageModel:

    def __init__(self, vocab: Vocab):
        super().__init__()

        self.vocab = vocab
        self.progress = 0   # To print progress.

        self.event_count:   Counter[Ngram] = Counter()  # numerator c(...) function.
        self.context_count: Counter[Ngram] = Counter()  # denominator c(...) function.
        # In this program, the argument to the counter should be an Ngram, 
        # which is always a tuple of Wordtypes, never a single Wordtype:
        # Zerogram: context_count[()]
        # Bigram:   context_count[(x,y)]   or equivalently context_count[x,y]
        # Unigram:  context_count[(y,)]    or equivalently context_count[y,]
        # but not:  context_count[(y)]     or equivalently context_count[y]  
        #             which incorrectly looks up a Wordtype instead of a 1-tuple

    @property
    def vocab_size(self) -> int:
        assert self.vocab is not None
        return len(self.vocab)

    # We need to collect two kinds of n-gram counts.
    # To compute p(z | xy) for a trigram xyz, we need c(xy) for the 
    # denominator and c(yz) for the backed-off numerator.  Both of these 
    # look like bigram counts ... but they are not quite the same thing!
    #
    # For a sentence of length N, we are iterating over trigrams xyz where
    # the position of z falls in 1 ... N+1 (so z can be EOS but not BOS),
    # and therefore
    # the position of y falls in 0 ... N   (so y can be BOS but not EOS).
    # 
    # When we write c(yz), we are counting *events z* with *context* y:
    #         c(yz) = |{i in [1, N]: w[i-1] w[i] = yz}|
    # We keep these "event counts" in `event_count` and use them in the numerator.
    # Notice that z=BOS is not possible (BOS is not a possible event).
    # 
    # When we write c(xy), we are counting *all events* with *context* xy:
    #         c(xy) = |{i in [1, N]: w[i-2] w[i-1] = xy}|
    # We keep these "context counts" in `context_count` and use them in the denominator.
    # Notice that y=EOS is not possible (EOS cannot appear in the context).
    #
    # In short, c(xy) and c(yz) count the training bigrams slightly differently.  
    # Likewise, c(y) and c(z) count the training unigrams slightly differently.
    #
    # Note: For bigrams and unigrams that don't include BOS or EOS -- which
    # is most of them! -- `event_count` and `context_count` will give the
    # same value.  So you could save about half the memory if you were
    # careful to store those cases only once.  (How?)  That would make the
    # code slightly more complicated, but would be worth it in a real system.

    def count_trigram_events(self, trigram: Trigram) -> None:
        """Record one token of the trigram and also of its suffixes (for backoff)."""
        (x, y, z) = trigram
        self.event_count[(x, y, z )] += 1
        self.event_count[   (y, z )] += 1
        self.event_count[      (z,)] += 1  # the comma is necessary to make this a tuple
        self.event_count[        ()] += 1

    def count_trigram_contexts(self, trigram: Trigram) -> None:
        """Record one token of the trigram's CONTEXT portion, 
        and also the suffixes of that context (for backoff)."""
        (x, y, _) = trigram    # we don't care about z
        self.context_count[(x, y )] += 1
        self.context_count[   (y,)] += 1
        self.context_count[     ()] += 1

    def log_prob(self, x: Wordtype, y: Wordtype, z: Wordtype) -> float:
        """Computes an estimate of the trigram log probability log p(z | x,y)
        according to the language model.  The log_prob is what we need to compute
        cross-entropy and to train the model.  It is also unlikely to underflow,
        in contrast to prob.  In many models, we can compute the log_prob directly, 
        rather than first computing the prob and then calling math.log."""
        class_name = type(self).__name__
        if class_name == LanguageModel.__name__:
            raise NotImplementedError("You shouldn't be calling log_prob on an instance of LanguageModel, but on an instance of one of its subclasses.")
        raise NotImplementedError(
            f"{class_name}.log_prob is not implemented yet (you should override LanguageModel.log_prob)"
        )

    def save(self, model_path: Path) -> None:
        log.info(f"Saving model to {model_path}")
        torch.save(self, model_path, pickle_protocol=pickle.HIGHEST_PROTOCOL)
            # torch.save is similar to pickle.dump but handles tensors too
        log.info(f"Saved model to {model_path}")

    @classmethod
    def load(cls, model_path: Path, device: str = 'cpu') -> "LanguageModel":
        log.info(f"Loading model from {model_path}")
        model = torch.load(model_path, map_location=device)
            # torch.load is similar to pickle.load but handles tensors too
            # map_location allows loading tensors on different device than saved
        if not isinstance(model, cls):
            raise ValueError(f"Type Error: expected object of type {cls} but got {type(result)} from file {model_path}")
        log.info(f"Loaded model from {model_path}")
        return model

    def train(self, file: Path) -> None:
        """Create vocabulary and store n-gram counts.  In subclasses, we might
        override this with a method that computes parameters instead of counts."""

        log.info(f"Training from corpus {file}")

        # Clear out any previous training.
        self.event_count   = Counter()
        self.context_count = Counter()

        for trigram in read_trigrams(file, self.vocab):
            self.count_trigram_events(trigram)
            self.count_trigram_contexts(trigram)
            self.show_progress()

        sys.stderr.write("\n")  # done printing progress dots "...."
        log.info(f"Finished counting {self.event_count[()]} tokens")

    def show_progress(self, freq: int = 5000) -> None:
        """Print a dot to stderr every 5000 calls (frequency can be changed)."""
        self.progress += 1
        if self.progress % freq == 1:
            sys.stderr.write(".")


##### SPECIFIC FAMILIES OF LANGUAGE MODELS

class CountBasedLanguageModel(LanguageModel):

    def log_prob(self, x: Wordtype, y: Wordtype, z: Wordtype) -> float:
        # For count-based language models, it is usually convenient
        # to compute the probability first (by dividing counts) and
        # then taking the log.
        prob = self.prob(x, y, z)
        if prob == 0.0:
            return -math.inf
        return math.log(prob)

    def prob(self, x: Wordtype, y: Wordtype, z: Wordtype) -> float:
        """Computes a smoothed estimate of the trigram probability p(z | x,y)
        according to the language model.
        """
        class_name = type(self).__name__
        if class_name == CountBasedLanguageModel.__name__:
            raise NotImplementedError("You shouldn't be calling prob on an instance of CountBasedLanguageModel, but on an instance of one of its subclasses.")
        raise NotImplementedError(
            f"{class_name}.prob is not implemented yet (you should override CountBasedLanguageModel.prob)"
        )


class UniformLanguageModel(CountBasedLanguageModel):
    def prob(self, x: Wordtype, y: Wordtype, z: Wordtype) -> float:
        return 1 / self.vocab_size


class AddLambdaLanguageModel(CountBasedLanguageModel):
    def __init__(self, vocab: Vocab, lambda_: float) -> None:
        super().__init__(vocab)
        if lambda_ < 0.0:
            raise ValueError(f"Negative lambda argument of {lambda_} could result in negative smoothed probs")
        self.lambda_ = lambda_

    def prob(self, x: Wordtype, y: Wordtype, z: Wordtype) -> float:
        assert isinstance(x, Wordtype)
        if not isinstance(y, Wordtype):
            print(y)
        assert isinstance(y, Wordtype)
        assert isinstance(z, Wordtype)
        if not self.event_count[x, y, z] <= self.context_count[x, y]:
            print(x,y,z)
        assert self.event_count[x, y, z] <= self.context_count[x, y]
        return ((self.event_count[x, y, z] + self.lambda_) /
                (self.context_count[x, y] + self.lambda_ * self.vocab_size))

        # Notice that summing the numerator over all values of typeZ
        # will give the denominator.  Therefore, summing up the quotient
        # over all values of typeZ will give 1, so sum_z p(z | ...) = 1
        # as is required for any probability function.


class BackoffAddLambdaLanguageModel(AddLambdaLanguageModel):
    def __init__(self, vocab: Vocab, lambda_: float) -> None:
        super().__init__(vocab, lambda_)

    def prob(self, x: Wordtype, y: Wordtype, z: Wordtype) -> float:
        # TODO: Reimplement me so that I do backoff
        ans = 1 / len(self.vocab)
        ans = (self.event_count[      (z,)] + self.lambda_ * len(self.vocab) * ans) / (self.context_count[     ()] + self.lambda_ * len(self.vocab))
        ans = (self.event_count[   (y, z )] + self.lambda_ * len(self.vocab) * ans) / (self.context_count[   (y,)] + self.lambda_ * len(self.vocab))
        ans = (self.event_count[(x, y, z )] + self.lambda_ * len(self.vocab) * ans) / (self.context_count[(x, y )] + self.lambda_ * len(self.vocab))
        return ans
        # return super().prob(x, y, z)
        # Don't forget the difference between the Wordtype z and the
        # 1-element tuple (z,). If you're looking up counts,
        # these will have very different counts!


class Lexicon:
    """
    Class that manages a lexicon.

    >>> my_lexicon = Lexicon.from_file(my_file)
    """

    def __init__(self, length = 0, embedding_dim = 0) -> None:
        """Load information into coupled word-index mapping and embedding matrix."""

        self.length = length
        self.embedding_dim = embedding_dim
        self.embeddings = torch.zeros((length, embedding_dim), dtype=torch.float32)
        self.vocab = Integerizer()


    @classmethod
    def from_file(cls, file: Path) -> Lexicon:
        
        with open(file) as f:
            first_line = next(f)  # Peel off the special first line.
            first_line = first_line.strip().split() 
            length = int(first_line[0])
            embedding_dim = int(first_line[1])

            # create lexicon
            lexicon = cls(length, embedding_dim)

            for line in f:
                line = line.strip().split()
                word = line[0]
                word_index = lexicon.vocab.index(word, add=True)
                # update embedding
                embedding = torch.tensor([float(val) for val in line[1:]], dtype=torch.float32)
                lexicon.embeddings[word_index] = embedding

        return lexicon
    
    def contains(self, word):
        if self.vocab.index(word) is None:
            return False
        return True
    
    def get_embedding(self, word): # assumption: word is in lexicon
        idx = self.vocab.index(word)
        return self.embeddings[idx]




class EmbeddingLogLinearLanguageModel(LanguageModel, nn.Module):
    # Note the use of multiple inheritance: we are both a LanguageModel and a torch.nn.Module.
    
    def __init__(self, vocab: Vocab, lexicon_file: Path, l2: float, epochs: int) -> None:
        super().__init__(vocab)
        if l2 < 0:
            raise ValueError("Negative regularization strength {l2}")
        self.l2: float = l2

        # TODO: ADD CODE TO READ THE LEXICON OF WORD VECTORS AND STORE IT IN A USEFUL FORMAT.
        
        # Load lexicon
        self.lexicon = Lexicon.from_file(lexicon_file)
        self.dim: int = self.lexicon.embedding_dim  # TODO: SET THIS TO THE DIMENSIONALITY OF THE VECTORS
        self.epochs = epochs
        ool_idx = self.lexicon.vocab.index('OOL')
        self.ool_embedding = self.lexicon.embeddings[ool_idx]

        # We wrap the following matrices in nn.Parameter objects.
        self.X = nn.Parameter(torch.zeros((self.dim, self.dim)), requires_grad=True)
        self.Y = nn.Parameter(torch.zeros((self.dim, self.dim)), requires_grad=True)


    def log_prob(self, x: Wordtype, y: Wordtype, z: Wordtype) -> float:
        """Return log p(z | xy) according to this language model."""
        res = self.log_prob_tensor(x, y, z)
        return res.item()


    @typechecked
    def log_prob_tensor(self, x: Wordtype, y: Wordtype, z: Wordtype) -> TorchScalar:
        """Return the same value as log_prob, but stored as a tensor."""
        
        # As noted below, it's important to use a tensor for training.
        # Most of your intermediate quantities, like logits below, will
        # also be stored as tensors.  (That is normal in PyTorch, so it
        # would be weird to append `_tensor` to their names.  We only
        # appended `_tensor` to the name of this method to distinguish
        # it from the class's general `log_prob` method.)

        # TODO: IMPLEMENT ME!
        # type annotations for Tensors.
        
        # Check if x and y are OOV; if so convert to OOL
        x = 'OOL' if (not x in self.vocab) or (x == 'OOV') else x
        y = 'OOL' if (not y in self.vocab) or (y == 'OOV') else y
        z = 'OOL' if (not z in self.vocab) or (z == 'OOV') else z

        # Retrieve embeddings for x and y, handle if x and y are OOL
        x_embed = self.lexicon.get_embedding(x) if self.lexicon.contains(x) else self.ool_embedding
        y_embed = self.lexicon.get_embedding(y) if self.lexicon.contains(y) else self.ool_embedding
        z_embed = self.lexicon.get_embedding(z) if self.lexicon.contains(z) else self.ool_embedding

        # numerator in log space
        numerator = x_embed @ self.X @ z_embed + y_embed @ self.Y @ z_embed

        # denominator in log space
        denominator = torch.logsumexp(self.logits(x, y), dim = 0)
        
        # log space division
        return numerator - denominator


    def logits(self, x: Wordtype, y: Wordtype) -> Float[torch.Tensor,"vocab"]:
        """Return a vector of the logs of the unnormalized probabilities f(xyz) * θ 
        for the various types z in the vocabulary.
        These are commonly known as "logits" or "log-odds": the values that you 
        exponentiate and renormalize in order to get a probability distribution."""
        # TODO: IMPLEMENT ME!
        
        # Check if x and y are OOV; if so convert to OOL
        x = 'OOL' if (not x in self.vocab) or (x == 'OOV') else x
        y = 'OOL' if (not y in self.vocab) or (y == 'OOV') else y

        # Retrieve embeddings for x and y, handle if x and y are OOL
        x_embed = self.lexicon.get_embedding(x) if self.lexicon.contains(x) else self.ool_embedding
        y_embed = self.lexicon.get_embedding(y) if self.lexicon.contains(y) else self.ool_embedding
        
        # Create tensor. Cols - embeddings, # rows = vocab_size.
        E = torch.empty((self.dim, self.vocab_size), dtype=torch.float32)
        for i, word in enumerate(self.vocab):
            if word == "OOV" or not self.lexicon.contains(word): # OOV or OOL -> OOL embedding
                E[:, i] = self.ool_embedding
            else:
                E[:, i] = self.lexicon.get_embedding(word)

        xX_E = (x_embed @ self.X) @ E
        yY_E = (y_embed @ self.Y) @ E
        logit = xX_E + yY_E

        return logit


    def train(self, file: Path):    # type: ignore
        
        # Optimization hyperparameters.
        gamma0 = 0.00001  # for spam detection
        # gamma0 = 0.01 # for language ID

        optimizer = optim.SGD(self.parameters(), lr=gamma0)

        # Initialize the parameter matrices to be full of zeros.
        nn.init.zeros_(self.X)   # type: ignore
        nn.init.zeros_(self.Y)   # type: ignore

        N = num_tokens(file)
        log.info(f"Start optimizing on {N} training tokens...")

        #####################
        # TODO
        
        print(f"Training from corpus {file}")
        # self.epochs = 0
        for i in range(self.epochs):
            total_F = 0.0

            for (x, y, z) in tqdm(read_trigrams(file, self.vocab), total=N, desc=f"Epoch {i+1}/{self.epochs}"):
                optimizer.zero_grad()
                log_prob = self.log_prob_tensor(x, y, z)
                
                # print((x, y, z))
                # print(f"log prob tensor: {log_prob}")
                
                # L2 regularization = sum of squared parameters
                all_params = torch.cat([param.view(-1) for param in self.parameters() if param.requires_grad])
                l2_sum = torch.sum(all_params ** 2)
                regularizer = (self.l2 / N) * l2_sum

                F_i = log_prob - regularizer
                # F_i = log_prob
                # print((x, y, z))
                # print(F_i)
                # print(F_i.shape)
                F_i = F_i.unsqueeze(0)
                # print(F_i.shape)
                # print(log_prob)
                # print(regularizer)

                (-F_i).backward()
                # print("check 3")
                
                total_F += F_i.item()
                optimizer.step()
                
            avg_F = total_F / N if N > 0 else float('-inf')
            log.info(f"epoch {i+1}: F = {avg_F}")

        # print(self.X, self.Y)

        log.info("done optimizing.")


class ImprovedLogLinearLanguageModel(EmbeddingLogLinearLanguageModel):
    # TODO: IMPLEMENT ME!
    
    # This is where you get to come up with some features of your own, as
    # described in the reading handout.  This class inherits from
    # EmbeddingLogLinearLanguageModel and you can override anything, such as
    # `log_prob`.

    # OTHER OPTIONAL IMPROVEMENTS: You could override the `train` method.
    # Instead of using 10 epochs, try "improving the SGD training loop" as
    # described in the reading handout.  Some possibilities:
    #
    # * You can use the `draw_trigrams_forever` function that we
    #   provided to shuffle the trigrams on each epoch.
    #
    # * You can choose to compute F_i using a mini-batch of trigrams
    #   instead of a single trigram, and try to vectorize the computation
    #   over the mini-batch.
    #
    # * Instead of running for exactly 10*N trigrams, you can implement
    #   early stopping by giving the `train` method access to dev data.
    #   This will run for as long as continued training is helpful,
    #   so it might run for more or fewer than 10*N trigrams.
    #
    # * You could use a different optimization algorithm instead of SGD, such
    #   as `torch.optim.Adam` (https://pytorch.org/docs/stable/optim.html).
    #
    pass
