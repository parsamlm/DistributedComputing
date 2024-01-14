import csv
from datetime import datetime
import gzip
import os
import os.path
import functools
import math
import random
from tempfile import NamedTemporaryFile
from urllib.request import urlopen

MUSTANG_URL = 'https://ftp.pdl.cmu.edu/pub/datasets/ATLAS/mustang/mustang_release_v1.0beta.csv.gz'

# NOTE: if you want to shuffle a trace, have a look at the `random.shuffle` function.

def weibull_generator(shape, mean):
    """Returns a callable that outputs random variables with a Weibull distribution having the given shape and mean."""

    return functools.partial(random.weibullvariate, mean / math.gamma(1 + 1 / shape), shape)


def isoformat2ts(date_string):
    return datetime.fromisoformat(date_string).timestamp()

def parse_mustang(path=None):
    """Parses the Mustang trace and returns a list of (delay, size) pairs."""

    if path is None:
        path = MUSTANG_URL.split('/')[-1]
    if not os.path.exists(path):
        with urlopen(MUSTANG_URL) as url, NamedTemporaryFile(delete=False) as tmp:
            print(f"Downloading Mustang dataset (temporary file: {tmp.name})...", end=' ', flush=True)
            tmp.write(url.read())
            os.rename(tmp.name, path)
        print("done.")
    with gzip.open(path, 'rt', newline='') as f:
        result = []
        last_submit = None
        for row in csv.DictReader(f):
            if row['job_status'] != 'COMPLETED':
                continue
            time_columns = ['submit_time', 'start_time', 'end_time']
            try:
                submit, start, end = (isoformat2ts(row[column]) for column in time_columns)
            except ValueError:  # some values have a missing `start_time` column. We ignore them.
                continue
            delay = submit - last_submit if last_submit is not None else 0
            assert delay >= 0
            result.append((delay, (end - start) * int(row['node_count'])))
    print(f"{len(result):,} jobs parsed")
    return result


def normalize_trace(trace, lambd, mu=1):
    """Renormalize a trace such that the average delays and size are respectively `1/lambd` and `1/mu`."""

    n = len(trace)
    delay_sum = size_sum = 0
    for delay, size in trace:
        delay_sum += delay
        size_sum += size
    delay_factor = n * delay_sum / lambd
    size_factor = n * size_sum / mu
    return [(delay * delay_factor, size * size_factor) for delay, size in trace]


if __name__ == '__main__':  # sanity check

    normalize_trace(parse_mustang(), 0.7)

    n_items = 1_000_000

    for shape in 0.5, 1, 2:
        for mean in 0.5, 1, 2:
            gen = weibull_generator(shape, mean)
            m = sum(gen() for _ in range(n_items)) / n_items
            print(f"shape={shape:3}, mean={mean:3}; theoretical mean: {mean:.3f}; experimental mean: {m:.3f}")
