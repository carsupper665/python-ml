from __future__ import annotations

import contextlib
import io
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from rich.progress import BarColumn, DownloadColumn, Progress, TaskProgressColumn, TextColumn, TimeRemainingColumn, TransferSpeedColumn

from mltrainer.core.model_registry import get_dataset_spec


class DatasetDownloadError(RuntimeError):
    pass


def ensure_dataset_available(dataset: str, progress_callback=None) -> Path:
    spec = get_dataset_spec(dataset)
    if spec.file_path.exists():
        return spec.file_path

    if progress_callback is not None:
        progress_callback(f"找不到資料集 `{dataset}`，開始下載...")

    return download_dataset(dataset)


def download_dataset(dataset: str) -> Path:
    spec = get_dataset_spec(dataset)
    if spec.file_path.exists():
        return spec.file_path

    spec.file_path.parent.mkdir(parents=True, exist_ok=True)
    if spec.download_source == "http":
        _download_http(spec.download_url, spec.file_path, spec.name)
    elif spec.download_source == "kaggle":
        _download_kaggle(spec.kaggle_dataset, spec.file_path, spec.name)
    else:
        raise DatasetDownloadError(f"Unsupported download source for dataset '{dataset}'.")

    if not spec.file_path.exists():
        raise DatasetDownloadError(f"Download finished but '{spec.file_path.name}' was not created.")
    return spec.file_path


def download_all_datasets() -> list[Path]:
    return [download_dataset(dataset) for dataset in ["heart", "winequality-red"]]


def _progress() -> Progress:
    return Progress(
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
    )


def _download_http(url: str | None, destination: Path, dataset_name: str) -> None:
    if not url:
        raise DatasetDownloadError(f"Missing download URL for dataset '{dataset_name}'.")

    with _progress() as progress:
        task_id = progress.add_task(f"下載 {dataset_name}", total=None)

        def reporthook(block_num: int, block_size: int, total_size: int) -> None:
            if total_size > 0 and progress.tasks[task_id].total != total_size:
                progress.update(task_id, total=total_size)
            downloaded = block_num * block_size
            progress.update(task_id, completed=min(downloaded, total_size) if total_size > 0 else downloaded)

        try:
            urllib.request.urlretrieve(url, destination, reporthook=reporthook)
        except Exception as exc:
            raise DatasetDownloadError(f"Failed to download '{dataset_name}' from {url}: {exc}") from exc


def _download_kaggle(dataset_slug: str | None, destination: Path, dataset_name: str) -> None:
    if not dataset_slug:
        raise DatasetDownloadError(f"Missing Kaggle dataset slug for dataset '{dataset_name}'.")

    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            from kaggle.api.kaggle_api_extended import KaggleApi

            api = KaggleApi()
            api.authenticate()
    except BaseException as exc:
        raise DatasetDownloadError(
            "Kaggle download requires API credentials. Please configure 'kaggle.json' first, then rerun the command."
        ) from exc

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        archive_path = temp_path / f"{dataset_name}.zip"

        with _progress() as progress:
            task_id = progress.add_task(f"下載 {dataset_name}", total=None)
            try:
                api.dataset_download_files(dataset_slug, path=temp_dir, force=True, quiet=False)
            except Exception as exc:
                raise DatasetDownloadError(f"Failed to download '{dataset_name}' from Kaggle: {exc}") from exc
            finally:
                progress.update(task_id, completed=1, total=1)

        if not archive_path.exists():
            zip_candidates = sorted(temp_path.glob("*.zip"))
            if not zip_candidates:
                raise DatasetDownloadError(f"Kaggle download for '{dataset_name}' did not produce a zip file.")
            archive_path = zip_candidates[0]

        extracted_csv = _extract_first_csv(archive_path, temp_path)
        shutil.move(str(extracted_csv), destination)


def _extract_first_csv(archive_path: Path, extract_dir: Path) -> Path:
    with zipfile.ZipFile(archive_path) as zip_file:
        csv_members = [member for member in zip_file.namelist() if member.lower().endswith(".csv")]
        if not csv_members:
            raise DatasetDownloadError(f"Archive '{archive_path.name}' does not contain any CSV file.")
        member = csv_members[0]
        zip_file.extract(member, extract_dir)
    return extract_dir / member
