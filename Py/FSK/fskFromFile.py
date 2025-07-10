#!/usr/bin/python3

"""Demodulate FSK from a waveform stored in a text file."""

import sys
import numpy as np

# Allow importing the fskReceiver class
from fskReceiver import fskReceiver


def load_waveform(filename, repeat=1):
    """Load complex samples from a whitespace separated text file.

    Parameters
    ----------
    filename : str
        Path to the text file containing real and imaginary parts.
    repeat : int, optional
        Number of times to replicate the waveform.

    Returns
    -------
    np.ndarray
        Complex numpy array of samples.
    """
    data = np.loadtxt(filename)
    if data.ndim == 1:
        complex_samples = data.astype(np.complex128)
    else:
        complex_samples = data[:, 0] + 1j * data[:, 1]
    if repeat > 1:
        complex_samples = np.tile(complex_samples, repeat)
    return complex_samples


def main():
    if len(sys.argv) < 2:
        print("Usage: python fskFromFile.py <waveform.txt> [repeat]")
        sys.exit(1)

    filename = sys.argv[1]
    repeat = int(sys.argv[2]) if len(sys.argv) > 2 else 1

    samples = load_waveform(filename, repeat)

    f0 = 250
    f1 = 625
    fs = 2e3
    sym_len = 40e-3
    rx = fskReceiver(f0, f1, fs, len(samples), sym_len)
    decision_var = rx.fskSymbolEstimate(samples)

    print(decision_var)


if __name__ == "__main__":
    main()
