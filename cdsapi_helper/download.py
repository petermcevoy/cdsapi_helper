from functools import partial
from multiprocessing.pool import ThreadPool
from pathlib import Path
from queue import Queue
from time import sleep

import cdsapi
import pandas as pd
from requests.exceptions import HTTPError

from .utils import get_json_sem_hash, request_to_df, REQUEST_DATABASE_FILE

MAX_ACTIVE_REQUESTS = 128

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

def wait_and_download_requests(df: pd.DataFrame, cache_dir: Path, wait_for_all: bool) -> pd.DataFrame:
    if df.size == 0:
        return df
    wait_time_min = 1
    while True:
        df = update_request_state(df)
        # Anything in the queue ready for download?
        if (df.state == "completed").any():
            df = download_completed_requests(df, cache_dir)
            wait_time_min = 1
            if not wait_for_all:
                break
        elif df.state.isin({"queued", "running", "accepted"}).any():
            # Wait a maximum of 30 min before checking the status again.
            print(f"Requests are running, waiting {wait_time_min:0.2f} min.")
            sleep(60 * wait_time_min)
            wait_time_min = min(30, 2 * wait_time_min)
        else:
            break
    return df

def process_requests(request_entries: list[RequestEntry], cache_dir: Path, dry_run: bool) -> None:
    df = pd.DataFrame()
    request_hashes = {}
    if REQUEST_DATABASE_FILE.exists():
        df = pd.read_csv(REQUEST_DATABASE_FILE, index_col=0, dtype=str)
        request_hashes = set(df.get("request_hash", []))

    num_active_requests = 0
    for entry in request_entries:
        hash = entry.get_sha256()
        if hash in request_hashes:
            print("Request already sent.")
            continue
        if dry_run:
            print(f"Would sent request for {entry.dataset}, {entry.request}")
            continue
        if num_active_requests == MAX_ACTIVE_REQUESTS:
            df = wait_and_download_requests(df, cache_dir, wait_for_all=False)
        # send new request via CDS-API.
        client = _get_cds_client_cached()
        result = client.retrieve(entry.dataset, entry.request)
        req_df = request_to_df(entry.request, result.reply, hash)
        df = req_df if df.size == 0 else pd.concat([df, req_df], ignore_index=True)
        num_active_requests += 1
    # wait for remain requsts and save
    df = wait_and_download_requests(df, cache_dir, wait_for_all=True)
    df = df.reset_index(drop=True)
    df.to_csv(REQUEST_DATABASE_FILE)

def update_request_state(df: pd.DataFrame) -> pd.DataFrame:
    print("Updating requests...")
    rows_to_drop = []
    for request in df.itertuples():
        state = request.state
        idx = request.Index
        if state not in ("completed", "downloaded", "deleted"):
            try:
                client = _get_cds_client_cached()
                result = client.client.get_remote(request.request_id)
                result.update()
                state = result.reply["state"]
            except HTTPError as err:
                print(f"Request {idx} not found")
                print(err)
                state = "deleted"
            df.at[idx, "state"] = state
        if state == "deleted":
            rows_to_drop.append(idx)
    df.drop(rows_to_drop, inplace=True)
    return df

def download_completed_requests(df: pd.DataFrame, output_folder: Path) -> pd.DataFrame:
    new_states = []
    for row in df.itertuples():
        new_state = download_single_request(
            state=row.state,
            request_id=row.request_id,
            request_hash=row.request_hash,
            output_folder=output_folder,
        )
        new_states.append(new_state)
    df['state'] = new_states
    return df


CDS_CLIENT = None

def _get_cds_client_cached() -> cdsapi.Client:
    """Get the CDS client, will create a new one if it doesn't exist"""
    global CDS_CLIENT

    if CDS_CLIENT is not None:
        return CDS_CLIENT

    CDS_CLIENT = cdsapi.Client(timeout=600, wait_until_complete=False, delete=False)
    return CDS_CLIENT

def download_single_request(
    state: str,
    request_id,
    request_hash: str,
    output_folder: Path,
) -> str:
    if state != "completed":
        return state

    try:
        client = _get_cds_client_cached()
        result = client.client.get_remote(request_id)
        result.update()

        # Delete previous, possibly corrupt, file
        filename = output_folder / request_hash
        if filename.exists():
            filename.unlink()
        result.download(filename)
        return "downloaded"
    except HTTPError as e:
        print("Request not found")
        print(e)
        print(f'request.state: {state}')
        # The request should be deleted
        return "deleted"
