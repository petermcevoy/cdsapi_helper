import hashlib
import os
import sys
import re
from pathlib import Path
from typing import Dict, List, Union

import pandas as pd

REQUEST_DATABASE_FILE = Path("./cds_requests.csv")
RE_FILENAMESPEC = re.compile(r"\{(\w+)\}")


def format_bytes(size):
    power = 2**10
    n = 0
    power_labels = {0: "B", 1: "KB", 2: "MB", 3: "GB", 4: "TB"}
    while size > power:
        size /= power
        n += 1
    return f"{size:.2f} {power_labels[n]}"


def print_files_and_size(filepaths: List[Path]):
    num_bytes = 0
    for filepath in filepaths:
        print(f"{filepath}", file=sys.stdout)
        num_bytes += filepath.stat().st_size
    print(f"Files amount to {format_bytes(num_bytes)}.", file=sys.stderr)


def request_to_df(request: dict, reply: dict, req_hash: str) -> pd.DataFrame:
    df = pd.DataFrame([request])
    df["request_hash"] = req_hash
    df["request_id"] = reply["request_id"]
    df["state"] = reply["state"]
    return df


def build_filename(dataset: str, request: dict, filename_spec: str) -> str:
    flattened_request = dict(request)
    flattened_request["dataset"] = dataset

    def replace_filespec(match):
        tag = match.group(1)
        return flattened_request[tag]

    filetype = ".nc" if request["format"] in ["netcdf", "netcdf_legacy"] else ".grib"

    filename = RE_FILENAMESPEC.sub(replace_filespec, filename_spec)
    filename += filetype
    filename = os.path.join(os.path.curdir, filename)
    return filename


# https://github.com/schollii/sandals/blob/master/json_sem_hash.py
JsonType = Union[str, int, float, List["JsonType"], "JsonTree"]
JsonTree = Dict[str, JsonType]
StrTreeType = Union[str, List["StrTreeType"], "StrTree"]
StrTree = Dict[str, StrTreeType]


def sorted_dict_str(data: JsonType) -> StrTreeType:
    if type(data) == dict:
        return {k: sorted_dict_str(data[k]) for k in sorted(data.keys())}
    elif type(data) == list:
        return [sorted_dict_str(val) for val in data]
    else:
        return str(data)


def get_json_sem_hash(data: JsonTree, hasher=hashlib.sha256) -> str:
    return hasher(bytes(repr(sorted_dict_str(data)), "UTF-8")).hexdigest()


def str_to_list(string: str) -> list:
    return string.strip("[]").replace("'", "").replace(" ", "").split(",")


def resolve_and_get_local_cache(cache_dir: Path):
    cache_dir.mkdir(exist_ok=True)

    local_cache_entries = {f: cache_dir / f for f in os.listdir(cache_dir)}

    # TODO: Consider having a list of ready-only cache_dirs that we can symlink or copy from.

    return local_cache_entries
