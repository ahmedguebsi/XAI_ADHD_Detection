import argparse
import re
import os
import sys
from datetime import datetime
from itertools import chain, combinations
from os import getcwd
from pathlib import Path
from typing import Dict, TypeVar

from IPython.display import display
from pandas import DataFrame

T = TypeVar("T")


def serialize_functions(*rest):
    current = rest[len(rest) - 1]
    rest = rest[:-1]
    return lambda x: current(serialize_functions(*rest)(x) if rest else x)


def is_arg_default(arg_name: str, parser: argparse.ArgumentParser, args: argparse.Namespace):
    return parser.get_default("arg_name") == vars(args)[arg_name]


def get_timestamp():
    return datetime.today().strftime("%Y-%m-%d-%H-%M-%S")


def dict_apply_procedture(old_dict: Dict[str, T], procedure) -> Dict[str, T]:
    return {k: procedure(v) for k, v in old_dict.items()}


def isnull_any(df):
    return df.isnull().any()


def isnull_values_sum(df):
    return df.isnull().values.sum() > 0


def isnull_sum(df):
    return df.isnull().sum() > 0


def isnull_values_any(df):
    return df.isnull().values.any()


def rows_with_null(df):
    return df[df.isnull().any(axis=1)]


def get_tmin_tmax(start, duration, end_cutoff):
    return (start - end_cutoff, start + duration - end_cutoff)


def to_numpy_reshape(x):
    return DataFrame.to_numpy(x).reshape(-1, 1)



def get_mat_file_name(file_path):
    file_name = os.path.basename(file_path)
    file_name_without_extension = os.path.splitext(file_name)[0]
    return file_name_without_extension

def get_mat_filename(i_child: int, state: str):
    return "{i_child}_{state}.mat".format(i_child=i_child, state=state)


def glimpse_df(df: DataFrame):

    print("\nShowing first 3 data points\n")
    display(df.head(n=3))

    print("\nShowing last 3 data points\n")
    display(df.tail(n=3))

    print("\nShowing 3 radnom data points\n")
    display(df.sample(n=3))

    display(df.describe())


def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(0, len(s) + 1))


def get_dictionary_leaves(dictionary: dict):
    def get_leaves(pair):
        key, value = pair
        if type(value) is dict:
            return get_dictionary_leaves(value)
        return [[key, value]]

    result = []
    for pair in dictionary.items():
        result.extend(get_leaves(pair))
    return result


def dict_to_byte_metadata(dictionary: dict):
    pairs = get_dictionary_leaves(dictionary)
    return ",".join(map(lambda key_value: " ".join([str(key_value[0]), str(key_value[1])]), pairs)).encode()


def dict_to_string(dictionary: dict):
    pairs = get_dictionary_leaves(dictionary)
    if pairs is None:
        return ""
    return "___".join(map(lambda key_value: "_".join([str(key_value[0]), str(key_value[1])]), pairs))


class SocketConcatenator(object):
    """
    Class adds an abillity to write to mulitple sockets
    For example, write both to stdout and to a file
    """

    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()



def stdout_to_file(file: Path):
    print()
    print("Standard output piped to file:")
    print(file)
    print()
    f = open(Path(getcwd(), file), "w")
    sys.stdout = SocketConcatenator(sys.stdout, f)


def print_report(file: Path, regex_filter):
    regex = re.compile(regex_filter)
    readlines = open(file, "r").readlines()
    filtered_lines = [s for s in readlines if regex.match(s)]
    for line in filtered_lines:
        print(line.strip())


if __name__ == "__main__":
    pass