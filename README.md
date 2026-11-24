# NNGen

Neural-network surrogates for fitting biophysical neuron models.

Fitting the channel conductances of a detailed NEURON model with a genetic algorithm is slow, because every candidate parameter set needs a full simulation. NNGen trains a feed-forward neural network on the populations the genetic algorithm has already simulated. The network learns to map a parameter set straight to its electrophysiological scores. It can then stand in for the simulator and screen candidate conductances before any of them are simulated.

This is an early project (2015, Idan Segev lab, Hebrew University).

## How it works

1. **Data.** Each generation of the genetic algorithm writes a block to `whole_population_*.txt`. A block holds one row per individual: 28 feature scores (the error on each electrophysiological target), followed by 31 parameter values (channel conductances and related model parameters).
2. **Parsing.** `dataParser.py` reads the population file into per-generation inputs and outputs, and splits them into training and test sets by generation.
3. **Normalization.** `Normalizers.py` applies z-score and min-max normalization, with inverses so predictions can be mapped back to the original units.
4. **Surrogate model.** `Lasagne2.py` trains a regression MLP with two hidden layers of 400 sigmoid units (31 inputs → 28 outputs, Nesterov momentum), using Lasagne/nolearn. It reports the per-feature and total absolute error on held-out generations. `Tests/KerasTest.py` is an equivalent Keras version.
5. **Analysis.** `Analyzer.py` compares how the true and predicted scores rank individuals by total error. Getting the ranking right is what matters for selection in a genetic algorithm.

## Files

| File | Purpose |
|---|---|
| `dataParser.py` | Parse genetic-algorithm population files and build train/test sets |
| `Normalizers.py` | Normalization helpers and their inverses, plus outlier rejection |
| `Lasagne.py`, `Lasagne2.py` | Train the MLP surrogate (Lasagne / nolearn) |
| `PN.py` | Earlier data-loading and normalization script for a 4-output version |
| `Analyzer.py` | Compare true and predicted error rankings |
| `dataAnalysis.py` | Plot the distribution of each output feature |
| `genetic.py` | Outline of the genetic-algorithm loop that would use the surrogate (unfinished) |
| `Tests/` | Keras version and test scripts |
| `whole_population_0.txt` | Example population file from the genetic algorithm |
| `model.pickle`, `BestNet.pickle` | Saved trained networks |

## Status

This is legacy research code. It was written for Python 2 with Theano-era Lasagne, nolearn and Keras 0.x, and it won't run on a current stack without porting. It's kept here for reference.

## Author

[Toviah Moldwin](https://toviahmoldwin.com)
