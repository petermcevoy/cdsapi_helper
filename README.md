# cdsapi_helper
**Work in progress**

This is a small utility package for working with the [cdsapi](https://github.com/ecmwf/cdsapi) from ECMWF.

The main feature is the `download_cds` CLI.
This reads a `.toml` specification of what you want to download, sends the requests and stores, among other things, the request id in `cds_requests.csv` in the current folder.
The `.csv` is then read on subsequent runs and used to query the status of the request.
It also makes sure that requests are not re-submitted on subsequent runs.

If one or more requests are “completed” the program will download the files.
By default, it will download 5 files in parallel, but this can be specified by the user by the `--n-jobs` option.

Filenames are based on the parameters of the request, see `filename_spec` in the specification below.

**Example request specification:**
```toml
dataset = "reanalysis-era5-pressure-levels"
looping_variables = ["variable", "year"]
filename_spec = "{dataset}/{variable}-{year}"

[request]
product_type = "reanalysis"
format = "netcdf"
variable = ["specific_humidity", "u_component_of_wind", "v_component_of_wind"]
year = ["2022", "2021", "2020"]
month = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]
day = [
  "01",
  "02",
  "03",
  "04",
  "05",
  "06",
  "07",
  "08",
  "09",
  "10",
  "11",
  "12",
  "13",
  "14",
  "15",
]
pressure_level = [
  '300',
  '350',
  '400',
  '450',
  '500',
]
time = ["00:00", "06:00", "12:00", "18:00"]
```
The request should be a standard cdspi request, and it will be expanded according to `looping_variables` when run.
This means that when run with the above in `example_spec.toml` e.g. `download_cds ./example_spec.toml`, one request will be sent for each combination of the entries is `variable` and `year` (9 requests/files in total).

## Installation
The easiest way to install is to clone this repository, `cd cdsapi_helper` and
```
pip install .
```
This should install and make `download_cds` available.

**Dependencies**
- cdsapi
- click
- pandas

