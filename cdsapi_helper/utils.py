import hashlib
import os
import sys
import re
from pathlib import Path
from typing import Dict, List, Union
from datetime import datetime
from collections import abc

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

class Area:
    """Helper class to convert areas."""

    def __init__(self, extent: list[float|int]) -> None:
        """Construct lat-lon area from extent[min-lat, min-lon, max-lat, max-lon].

        Args:
            extent: Boundary of the region ([min-lat, min-lon, max-lat, max-lon]).

        """
        assert len(extent) == 4, (
            f"Area must be specified as [min-latitude, min-longitude, "
            f"max-latitude, max-longitude] but found '{extent}'."
        )
        self.lat: list[float] = [extent[0], extent[2]]
        self.lon: list[float] = [extent[1], extent[3]]
        # Validate latitudes
        assert abs(self.lat[0] - self.lat[1]) <= 180, (
            f"Latitudes cannot be further than 180° apart (got {self.lat})."
        )
        assert self.lat[0] < self.lat[1], (
            "CDS-API does not support selection across coordinate boundaries "
            f"(got latitudes {self.lat}). Area must be specified as [min-latitude, "
            "min-longitude, max-latitude, max-longitude]."
        )
        assert (-90 <= self.lat[0] <= 90) and (-90 <= self.lat[1] <= 90), (
            f"Latitudes must be in range [-90, 90] but found {self.lat}. Area must "
            "be specified as [min-latitude, min-longitude, max-latitude"
            ", max-longitude]."
        )
        # Validate longitudes
        assert abs(self.lon[0] - self.lon[1]) <= 360, (
            f"Longitudes cannot be further than 360° apart (got {self.lat})."
        )
        assert self.lon[0] < self.lon[1], (
            "CDS-API does not support selection across coordinate boundaries "
            f"(got longitudes {self.lon}). Area must be specified as [min-latitude, "
            "min-longitude, max-latitude, max-longitude]."
        )
        assert (-180 <= self.lat[0] <= 360) and (-180 <= self.lat[1] <= 360), (
            f"Longitudes must be in range [-180, 360] but found {self.lon}. Area must "
            "be specified as [min-latitude, min-longitude, max-latitude"
            ", max-longitude]."
        )

    @property
    def extent(self) -> list[float|int]:
        """Extend of the region (lon_min, lon_max, lat_min, lat_max)."""
        return self.lon + self.lat

    @staticmethod
    def lat2str(lat: float) -> str:
        """Latitude string respresentation.

        Args:
            lat: The latitude.

        Returns:
            str: String respresentation of latitude bounds.

        """
        return f"{abs(lat)}{'SN'[lat >= 0]}"

    @staticmethod
    def lon2str(lon: float) -> str:
        """Lonitude string representation.

        Args:
            lon: The longitude

        Returns:
            str: String representation of a lonitude.

        """
        return f"{abs(lon)}{'WE'[lon >= 0]}"

    def __str__(self) -> str:
        """String representation like '20S-20N_20W-20E'."""
        return (
            f"{self.lat2str(self.lat[0])}-"
            f"{self.lat2str(self.lat[1])}_"
            f"{self.lon2str(self.lon[0])}-"
            f"{self.lon2str(self.lon[1])}"
        )

def _format_for_request(
    name: str,
    format_str: str,
    width_str: int|None,
    val: str|int|list[str|int]|None,
    default: list[str]|None,
    min_val: int,
    max_val: int,
) -> list[str]:
    if val is None:
        assert default is not None, f"{name}-variable is missing from request but required"
        return default
    formatted_values: list[str] = []
    if not isinstance(val, abc.Sized) or isinstance(val, str):
        val = [val]
    for v in val:
        if isinstance(v, str):
            assert v.isdigit(), f"{name} '{v}' is not a valid {name}, use integer or '1' or '01' etc."
            v = int(v)
        assert min_val <= v <= max_val, f"invalid {name} '{v}' must be in range [{min_val}, {max_val}]"
        if width_str is None:
            formatted_values.append(format_str.format(value=v))
        else:
            formatted_values.append(format_str.format(value=v, width=width_str))
    return sorted(formatted_values)


def format_and_validate_request(
    request: dict[str,list[str|int|float|list[float|int]]|str|int],
) -> dict[str,list[str]|list[list[float|int]]|str]:
    """Format the fields of a request, adding defaults where necessary."""
    assert 'area'  in request, "Field 'area' in request is required but missing."
    assert 'year'  in request, "Field 'year' in request is required but missing."
    assert 'month' in request, "Field 'month' in request is required but missing."
    unknown_keys = request.keys() - {
        'variable',
        'area',
        'year',
        'month',
        'day',
        'time',
        'pressure_level',
        'format',
        'data_format',
        'download_format',
        'product_type',
    }
    assert len(unknown_keys) == 0, (
        f"The request fields {unknown_keys} are unknown add them here or remove them."
    )

    if 'pressure_level' in request:
        request['pressure_level'] = _format_for_request(
            name='pressure-level',
            format_str="{value}",
            width_str=None,
            val=request['pressure_level'],
            default=None,
            min_val=1,
            max_val=1000,
        )

    current_date = datetime.now()
    request['year'] = _format_for_request(
        name='year',
        format_str="{value:0{width}}",
        width_str=4,
        val=request['year'],
        default=None,
        min_val=1940,
        max_val=current_date.year,
    )

    request['month'] = _format_for_request(
        name='month',
        format_str="{value:0{width}}",
        width_str=2,
        val=request['month'],
        default=None,
        min_val=1,
        max_val=12,
    )

    request['day'] = _format_for_request(
        name='day',
        format_str="{value:0{width}}",
        width_str=2,
        val=request.get('day'),
        default=[f"{d:02}" for d in range(31)],
        min_val=1,
        max_val=31,
    )

    time = request.get('time')
    if time is None:
        request['time'] = [f"{hour:02}:00" for hour in range(24)]
    else:
        if isinstance(time, str):
            time = [time]
        formatted_time: list[str] = []
        for t in time:
            assert ':' in t, f"Time must be specified in format '00:00' but found '{t}'."
            hour, minute = t.split(':')
            assert hour.isdigit() and minute.isdigit(), (
                f"Time must be specified in format '00:00' but found '{t}'."
            )
            assert 0 <= int(hour) <= 23, f"Hour of time value must be in [0,23] but found '{t}'."
            assert 0 <= int(minute) <= 59, f"Minute of time value must be in [0,59] but found '{t}'."
            formatted_time.append(f"{int(hour):02}:{int(minute):02}")
        request['time'] = formatted_time

    most_recent_date = datetime(
        year=int(request['year'][-1]),
        month=int(request['month'][-1]),
        day=int(request['day'][-1]),
        hour=23,
        minute=59,
        second=59,
    )
    assert current_date > most_recent_date, (
        f"Time of all requests must be in the past but found requests up to {most_recent_date.date}"
    )

    area = request['area']
    assert isinstance(area, list) and len(area) >= 1, (
        "Area must be specified as [min-latitude, min-longitude, max-latitude, max-longitude]."
    )
    if not isinstance(area[0], abc.Sized):
        area = [area]
    request['area'] = [Area(extent).extent for extent in area]
    return request

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
