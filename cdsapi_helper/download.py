from functools import partial
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Union

import cdsapi
import pandas as pd
from requests.exceptions import HTTPError

from .utils import get_json_sem_hash, request_to_df


class RequestEntry:
    def __init__(self, dataset, request, filename_spec):
        self.dataset = dataset
        self.request = request
        self.filename_spec = filename_spec

    def get_sha256(self):
        return get_json_sem_hash({"dataset": self.dataset, "request": self.request})


# Check to ensure hash stability:
# fmt: off
expected_hash = "23cf15695d9f9396a8d39ee97f86e894bae0fa09e9c6ca86db619384428acda9"
assert (
    RequestEntry(dataset='reanalysis-era5-pressure-levels', request={'product_type': 'reanalysis', 'format': 'netcdf', 'variable': 'temperature', 'year': '2015', 'month': '01', 'day': '01', 'pressure_level': ['1', '2', '3', '5', '7', '10', '20', '30', '50', '70', '100', '125', '150', '175', '200', '225', '250', '300', '350', '400', '450', '500', '550', '600', '650', '700', '750', '775', '800', '825', '850', '875', '900', '925', '950', '975', '1000'], 'time': ['00:00', '01:00', '02:00', '03:00', '04:00', '05:00', '06:00', '07:00', '08:00', '09:00', '10:00', '11:00', '12:00', '13:00', '14:00', '15:00', '16:00', '17:00', '18:00', '19:00', '20:00', '21:00', '22:00', '23:00']}, filename_spec='not_relevant').get_sha256()
    == expected_hash
), "RequestEntry.get_sha256() did not produce the expected hash!"
assert (
    RequestEntry(dataset='reanalysis-era5-pressure-levels', request={'format': 'netcdf', 'product_type': 'reanalysis', 'variable': 'temperature', 'year': '2015', 'month': '01', 'day': '01', 'pressure_level': ['1', '2', '3', '5', '7', '10', '20', '30', '50', '70', '100', '125', '150', '175', '200', '225', '250', '300', '350', '400', '450', '500', '550', '600', '650', '700', '750', '775', '800', '825', '850', '875', '900', '925', '950', '975', '1000'], 'time': ['00:00', '01:00', '02:00', '03:00', '04:00', '05:00', '06:00', '07:00', '08:00', '09:00', '10:00', '11:00', '12:00', '13:00', '14:00', '15:00', '16:00', '17:00', '18:00', '19:00', '20:00', '21:00', '22:00', '23:00']}, filename_spec='not_relevant').get_sha256()
    == expected_hash
), "RequestEntry.get_sha256() did not produce the expected hash!"
# fmt: on


def send_request(request_entries: list[RequestEntry], dry_run: bool) -> None:
    client = cdsapi.Client(wait_until_complete=False, delete=False)

    try:
        df = pd.read_csv("./cds_requests.csv", index_col=0, dtype=str)
    except FileNotFoundError:
        df = pd.DataFrame()

    for req_entry in request_entries:
        req_hash = req_entry.get_sha256()
        try:
            duplicate = df["request_hash"].isin([req_hash]).any()
        except KeyError:
            duplicate = False
        if not duplicate:
            if not dry_run:
                result = client.retrieve(req_entry.dataset, req_entry.request)
                reply = result.reply
            else:
                print(
                    f"Would have sent request for {req_entry.dataset}, {req_entry.request}"
                )
                # TODO: This causes issues when doing dry-run...
                reply = {"state": "test_state", "request_id": "test_id"}
            r_df = request_to_df(req_entry.request, reply, req_hash)
            df = pd.concat([df, r_df])
        else:
            print("Request already sent.")

    # Save it.
    df = df.reset_index(drop=True)
    df.to_csv("./cds_requests.csv")


def update_request(dry_run: bool) -> None:
    client = cdsapi.Client(timeout=600, wait_until_complete=False, delete=False)
    try:
        df = pd.read_csv("./cds_requests.csv", index_col=0, dtype=str)
    except FileNotFoundError:
        print("Nothing to update.")
        return

    print("Updating requests...")
    for request in df.itertuples():
        if (
            request.state != "completed"
            and request.state != "downloaded"
            and request.state != "deleted"
        ):
            try:
                if not dry_run:
                    result = client.client.get_remote(request.request_id)
                    result.update()
                    df.at[request.Index, "state"] = result.reply["state"]
            except HTTPError as err:
                print(f"Request {request.Index} not found")
                print(err)
                df.at[request.Index, "state"] = "deleted"

    df.to_csv("./cds_requests.csv")


def download_request(
    output_folder: Path, n_jobs: int = 5, dry_run: bool = False
) -> None:
    try:
        df = pd.read_csv("./cds_requests.csv", index_col=0, dtype=str)
    except FileNotFoundError:
        return
    client = cdsapi.Client(timeout=600, wait_until_complete=False, delete=False)
    print("Downloading completed requests...")
    # Some parallel downloads.
    download_helper_p = partial(
        download_helper, output_folder=output_folder, client=client, dry_run=dry_run
    )
    with ThreadPool(processes=n_jobs) as p:
        results = p.map(download_helper_p, df.itertuples())

    # Write new states.
    df.state = results
    # Save them.
    df.to_csv("./cds_requests.csv")


def download_helper(
    request: pd.core.frame.pandas,
    output_folder: Path,
    client: cdsapi.Client,
    dry_run: bool = False,
) -> str:
    if request.state == "completed":
        try:
            result = client.client.get_remote(request.request_id)
            result.update()
            filename = output_folder / request.request_hash
            if not dry_run:
                result.download(filename)
                return "downloaded"
            else:
                return request.state
        except HTTPError as e:
            print("Request not found")
            print(e)
            return request.state
    else:
        # No change to state.
        return request.state
