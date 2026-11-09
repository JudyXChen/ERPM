"""Column-oriented CSV helpers shared by the analysis scripts."""
import csv

import numpy as np


def read_columns(path):
    data = np.genfromtxt(path, delimiter=",", names=True, ndmin=1)
    return {name: data[name] for name in data.dtype.names}


def write_csv(path, columns):
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(zip(*columns.values()))
