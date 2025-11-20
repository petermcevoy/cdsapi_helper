#!/usr/bin/env python
import os
import sys
from copy import deepcopy
from itertools import product
from pathlib import Path
import datetime
import string

import click
import tomli

from .download import (
    process_requests,
    RequestEntry,
)
from .utils import (
    resolve_and_get_local_cache,
    print_files_and_size,
    format_bytes,
    format_and_validate_request,
    build_filename,
    REQUEST_DATABASE_FILE,
)


def generate_request_entries_from_specs(spec_paths):
    request_entries = []
    for spec_path in spec_paths:
        with open(spec_path, mode="rb") as fp:
            spec = tomli.load(fp)

        dataset = spec["dataset"]
        request = format_and_validate_request(spec["request"])

        assert isinstance(spec["filename_spec"], str), "Field 'filename_spec' in TOML must be a string."
        looping_variables: set[str] = {
            field_name for _, field_name, _, _ in string.Formatter().parse(spec["filename_spec"])
        }
        assert all(var in request for var in looping_variables), (
            f"The variables {looping_variables - request.keys()} are "
            "missing in the request but required from 'filename_spec'."
        )
        request_list = []
        for permutation_values in product(
            *[request[var_name] for var_name in looping_variables]
        ):
            sub_request = deepcopy(request)
            # change the values according to the current permutation
            for var_name, var_value in zip(looping_variables, permutation_values, strict=True):
                sub_request[var_name] = var_value

            request_entries.append(RequestEntry(
                dataset=dataset,
                request=sub_request,
                filename_spec=spec["filename_spec"],
            ))

    # Remove requests with invalid dates
    def has_valid_date(req: RequestEntry):
        year = int(req.request["year"])
        month = int(req.request["month"])
        day = int(req.request["day"])

        try:
            _ = datetime.datetime(year=year, month=month, day=day)
        except ValueError:
            return False

        return True

    request_entries = list(filter(has_valid_date, request_entries))

    return request_entries


@click.group()
@click.option(
    "--cache-dir",
    "cache_dir",
    show_default=True,
    default=Path("./cache"),
    type=Path,
    help="Directory for local cache where downloads are stored and output files are linked to",
)
@click.pass_context
def download_cds(
    ctx,
    cache_dir: Path,
) -> None:
    ctx.ensure_object(dict)
    ctx.obj["cache_dir"] = cache_dir


@download_cds.command(
    help="List files in stdout for files that are not accounted for by spec files and exit",
)
@click.argument("spec_paths", type=click.Path(exists=True), nargs=-1)
@click.pass_context
def list_dangling(
    ctx,
    spec_paths: list[str],
) -> None:
    cache_dir = ctx.obj["cache_dir"]
    request_entries = generate_request_entries_from_specs(spec_paths)
    local_cache = resolve_and_get_local_cache(cache_dir)
    dangling_cache_files = set(local_cache) - {r.get_sha256() for r in request_entries}
    print_files_and_size([cache_dir / file for file in dangling_cache_files])
    sys.exit(0)


@download_cds.command(
    help="List cache files in stdout expected by the specifications and exit. Exit success if all files exist.",
)
@click.argument("spec_paths", type=click.Path(exists=True), nargs=-1)
@click.pass_context
def list_files(
    ctx,
    spec_paths: list[str],
) -> None:
    cache_dir = ctx.obj["cache_dir"]

    request_entries = generate_request_entries_from_specs(spec_paths)
    expected_files = {cache_dir / file.get_sha256() for file in request_entries}

    for file in expected_files:
        click.echo(file)

    # Summary about size and potentially missing files
    local_cache = {cache_dir / file for file in resolve_and_get_local_cache(cache_dir)}
    expected_existing = local_cache.intersection(expected_files)

    num_bytes_existing = sum((f.stat().st_size for f in expected_existing))
    click.echo(f"Existing files amount to {format_bytes(num_bytes_existing)}", err=True)

    expected_missing = expected_files - expected_existing
    click.echo(
        f"There are {len(expected_existing)} expected files that exist.",
        err=True,
    )
    click.echo(
        click.style(
            f"There are {len(expected_missing)} expected files that are missing.",
            fg="green" if len(expected_missing) == 0 else "red",
        ),
        err=True,
    )
    if len(expected_missing) > 0:
        sys.exit(1)

    sys.exit(0)


@download_cds.command(
    help="Download files and create output directories according to spec files."
)
@click.pass_context
@click.argument("spec_paths", type=click.Path(exists=True), nargs=-1)
@click.option(
    "--dry-run",
    "dry_run",
    is_flag=True,
    show_default=True,
    default=False,
    help="Dry run: no download and no symlinks.",
)
@click.option(
    "--n-jobs",
    "n_jobs",
    show_default=True,
    default=5,
    type=click.INT,
)
@click.option(
    "--output-dir",
    "output_dir",
    show_default=True,
    default=Path("./output"),
    type=Path,
    help="Directory from which to create files according to filename_spec in spec files.",
)
def download(
    ctx,
    spec_paths: list[str],
    dry_run: bool,
    n_jobs: int,
    output_dir: Path,
) -> None:
    cache_dir = ctx.obj["cache_dir"]

    # check that cache_dir and output_dir and in the same sub path
    if output_dir.parent != cache_dir.parent:
        raise ValueError('cache_dir and output_dir need to have the same parent directory')

    request_entries = generate_request_entries_from_specs(spec_paths)
    click.echo(f"{len(request_entries)} request(s) generated in total.", err=True)

    local_cache = resolve_and_get_local_cache(cache_dir)
    remaining_requests = list(
        filter(
            lambda r: r.get_sha256() not in local_cache,
            request_entries,
        )
    )

    click.echo(
        f"{len(request_entries)-len(remaining_requests)} local cache hits", err=True
    )
    click.echo(f"{len(remaining_requests)} local cache misses", err=True)

    process_requests(remaining_requests, cache_dir, num_jobs=n_jobs, dry_run=dry_run)

    # Check that all requests are downloaded.
    for req_entry in request_entries:
        cache_file = cache_dir / req_entry.get_sha256()
        if not cache_file.exists():
            click.echo("All requests are not downloaded. Try again when data is ready.", err=True)
            click.echo(
                click.style(
                    f"Missing expected cache file {cache_file} for request {req_entry.request}",
                    fg="red",
                ),
                err=True,
            )
            sys.exit(1)

    # Create links to cached files according to filename_spec
    num_links = 0
    count_missing = 0
    for req_entry in request_entries:
        output_file = output_dir / build_filename(
            req_entry.dataset, req_entry.request, req_entry.filename_spec
        )
        cache_file = cache_dir / req_entry.get_sha256()

        if not cache_file.exists():
            click.echo(
                f"Warning: Missing entry {cache_file} for {req_entry.request}", err=True
            )
            count_missing = count_missing + 1

        output_file.parent.mkdir(parents=True, exist_ok=True)
        if output_file.exists():
            os.remove(output_file)

        relative_path_to_cache_entry = os.path.relpath(cache_file, output_file.parent)
        os.symlink(relative_path_to_cache_entry, output_file)
        num_links += 1

    click.echo(f"Created {num_links} symlinks.", err=True)

    assert count_missing == 0, "There were missing files!"

    # List summary of files not declared by input specs
    local_cache = resolve_and_get_local_cache(cache_dir)
    dangling_cache_files = set(local_cache) - {r.get_sha256() for r in request_entries}
    if len(dangling_cache_files) > 0:
        dangling_bytes = 0
        for file in dangling_cache_files:
            dangling_bytes += (cache_dir / file).stat().st_size
        click.echo(
            f"There are {len(dangling_cache_files)} ({format_bytes(dangling_bytes)}) dangling cache files not accounted for by input spec files.",
            err=True,
        )
        click.echo("Use `list-dangling` subcommand to display these files.", err=True)
