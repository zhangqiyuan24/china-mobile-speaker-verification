#!/usr/bin/env python3
"""Prepare VoxCeleb2, VoxCeleb1 test, MUSAN, and RIRS for applications/speaker_rec.

The output format follows the WeSpeaker VoxCeleb recipe, while the VoxCeleb2
conversion/download behavior is adapted from the project's existing data
utilities. Every stage is deterministic and safe to rerun.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import pickle
import shutil
import stat
import subprocess
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence


VOX2_URLS = (
    "https://huggingface.co/datasets/ProgramComputer/voxceleb/resolve/main/"
    "vox2/vox2_aac_1.zip",
    "https://huggingface.co/datasets/ProgramComputer/voxceleb/resolve/main/"
    "vox2/vox2_aac_2.zip",
)
VOX1_TEST_URL = (
    "https://huggingface.co/datasets/ProgramComputer/voxceleb/resolve/main/"
    "vox1/vox1_test_wav.zip"
)
VOX1_PROTOCOL_URL = "https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/veri_test2.txt"
VOX1_TEST_MD5 = "185fdc63c3c739954633d50379a3d102"
VOX1_PROTOCOL_MD5 = "b73110731c9223c1461fe49cb48dddfc"
MUSAN_URL = "https://www.openslr.org/resources/17/musan.tar.gz"
RIRS_URL = "https://www.openslr.org/resources/28/rirs_noises.zip"
MUSAN_MD5 = "0c472d4fc0c5141eca47ad1ffeb2a7df"
RIRS_MD5 = "e6f48e257286e05de56413b4779d8ffb"

DEFAULT_VOX2_FILES = 1_092_009
DEFAULT_VOX1_TEST_FILES = 4_874
DEFAULT_VOX1_TRIALS = 37_611
DEFAULT_MUSAN_FILES = 2_016
DEFAULT_RIRS_FILES = 60_000  # simulated_rirs only, matching WeSpeaker
MIN_WAV_BYTES = 44 + 1_000 * 2
STAGE_ORDER = ("download", "extract", "convert", "manifests", "lmdb")

_CONVERT_SOURCE_ROOT = ""
_CONVERT_OUTPUT_ROOT = ""
_CONVERT_DELETE_SOURCE = False
_CONVERT_TIMEOUT = 120


@dataclass(frozen=True)
class Asset:
    name: str
    url: str
    md5: str | None = None


def log(message: str) -> None:
    print(message, flush=True)


def md5sum(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(
    asset: Asset,
    destination: Path,
    token: str | None,
    retries: int,
) -> Path:
    """Download an asset with Range resume and optional MD5 validation."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        if asset.md5 and md5sum(destination) != asset.md5:
            raise RuntimeError(
                f"MD5 mismatch for existing {destination}; remove it and rerun"
            )
        log(f"[download] cached: {destination}")
        return destination

    partial = destination.with_name(destination.name + ".part")
    for attempt in range(retries + 1):
        existing = partial.stat().st_size if partial.exists() else 0
        headers = {"User-Agent": "clean-lite-vox2-preparer/1.0"}
        if token and "huggingface.co" in asset.url:
            headers["Authorization"] = f"Bearer {token}"
        if existing:
            headers["Range"] = f"bytes={existing}-"

        request = urllib.request.Request(asset.url, headers=headers)
        try:
            log(
                f"[download] {asset.url} -> {destination}"
                + (f" (resume at {existing:,} bytes)" if existing else "")
            )
            with urllib.request.urlopen(request, timeout=60) as response:
                status = getattr(response, "status", response.getcode())
                if existing and status != 206:
                    log("[download] server ignored Range; restarting file")
                    partial.unlink(missing_ok=True)
                    continue

                remaining = int(response.headers.get("Content-Length", "0"))
                total = existing + remaining if remaining else 0
                mode = "ab" if existing else "wb"
                downloaded = existing
                last_report = time.monotonic()
                with partial.open(mode) as output:
                    while True:
                        chunk = response.read(4 * 1024 * 1024)
                        if not chunk:
                            break
                        output.write(chunk)
                        downloaded += len(chunk)
                        now = time.monotonic()
                        if now - last_report >= 10:
                            suffix = f"/{total:,}" if total else ""
                            log(
                                f"[download] {asset.name}: {downloaded:,}{suffix} bytes"
                            )
                            last_report = now

            partial.replace(destination)
            if asset.md5 and md5sum(destination) != asset.md5:
                raise RuntimeError(f"Wrong MD5 for downloaded {destination}")
            return destination
        except (OSError, urllib.error.URLError) as error:
            if attempt == retries:
                raise RuntimeError(
                    f"Download failed after {retries + 1} attempts: {asset.url}"
                ) from error
            delay = min(30, 2**attempt)
            log(f"[download] transient error: {error}; retrying in {delay}s")
            time.sleep(delay)

    raise AssertionError("unreachable")


def _safe_destination(root: Path, member_name: str) -> Path:
    if Path(member_name).is_absolute():
        raise RuntimeError(f"Archive contains an absolute path: {member_name}")
    target = (root / member_name).resolve()
    try:
        target.relative_to(root.resolve())
    except ValueError as error:
        raise RuntimeError(
            f"Archive path escapes output directory: {member_name}"
        ) from error
    return target


def extract_archive(archive: Path, destination: Path) -> None:
    """Extract ZIP/TAR after rejecting traversal paths and links."""
    destination.mkdir(parents=True, exist_ok=True)
    marker = destination / f".{archive.name}.extracted"
    if marker.is_file():
        log(f"[extract] cached: {archive}")
        return

    log(f"[extract] {archive} -> {destination}")
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as bundle:
            for member in bundle.infolist():
                _safe_destination(destination, member.filename)
                mode = (member.external_attr >> 16) & 0o170000
                if mode == stat.S_IFLNK:
                    raise RuntimeError(
                        f"Archive symlink is not allowed: {member.filename}"
                    )
            bundle.extractall(destination)
    elif tarfile.is_tarfile(archive):
        with tarfile.open(archive, "r:*") as bundle:
            members = bundle.getmembers()
            for member in members:
                _safe_destination(destination, member.name)
                if member.issym() or member.islnk():
                    raise RuntimeError(f"Archive link is not allowed: {member.name}")
            bundle.extractall(destination, members=members)
    else:
        raise RuntimeError(f"Unsupported or corrupt archive: {archive}")
    marker.touch()


def resolved_vox2_urls(args: argparse.Namespace) -> tuple[str, ...]:
    return tuple(args.vox2_url or VOX2_URLS)


def archive_name(url: str) -> str:
    name = Path(urllib.parse.urlparse(url).path).name
    if not name:
        raise ValueError(f"Cannot determine archive name from URL: {url}")
    return name


def require_archives(paths: Sequence[Path], label: str) -> list[Path]:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing {label} archive(s): {', '.join(missing)}. "
            "Run the download stage or pass explicit archive paths."
        )
    return list(paths)


def get_vox2_archives(args: argparse.Namespace, downloads: Path) -> list[Path]:
    paths = (
        [Path(path).expanduser().resolve() for path in args.vox2_archive]
        if args.vox2_archive
        else [downloads / archive_name(url) for url in resolved_vox2_urls(args)]
    )
    return require_archives(paths, "VoxCeleb2")


def get_single_archive(
    explicit: str | None,
    default_path: Path,
    label: str,
) -> Path:
    path = Path(explicit).expanduser().resolve() if explicit else default_path
    return require_archives([path], label)[0]


def run_download(args: argparse.Namespace, downloads: Path) -> None:
    if not args.vox2_source and not args.vox2_archive:
        for url in resolved_vox2_urls(args):
            asset = Asset(archive_name(url), url)
            download_file(
                asset, downloads / asset.name, args.hf_token, args.download_retries
            )
    elif args.vox2_source:
        log("[download] --vox2-source supplied; skipping VoxCeleb2 download")
    else:
        log("[download] --vox2-archive supplied; skipping VoxCeleb2 download")

    if not args.vox1_test_source and not args.vox1_test_archive:
        asset = Asset(
            archive_name(args.vox1_test_url),
            args.vox1_test_url,
            VOX1_TEST_MD5 if args.vox1_test_url == VOX1_TEST_URL else None,
        )
        download_file(
            asset, downloads / asset.name, args.hf_token, args.download_retries
        )
    elif args.vox1_test_source:
        log("[download] --vox1-test-source supplied; skipping VoxCeleb1 download")
    else:
        log("[download] --vox1-test-archive supplied; skipping VoxCeleb1 download")

    source_protocol = None
    if args.vox1_test_source:
        source_root = Path(args.vox1_test_source).expanduser().resolve()
        for candidate in (
            source_root / "veri_test2.txt",
            source_root.parent / "veri_test2.txt",
        ):
            if candidate.is_file():
                source_protocol = candidate
                break
    if not args.vox1_protocol and source_protocol is None:
        asset = Asset(
            "veri_test2.txt",
            args.vox1_protocol_url,
            VOX1_PROTOCOL_MD5 if args.vox1_protocol_url == VOX1_PROTOCOL_URL else None,
        )
        download_file(asset, downloads / asset.name, None, args.download_retries)

    if not args.musan_source and not args.musan_archive:
        asset = Asset("musan.tar.gz", MUSAN_URL, MUSAN_MD5)
        download_file(asset, downloads / asset.name, None, args.download_retries)
    if not args.rirs_source and not args.rirs_archive:
        asset = Asset("rirs_noises.zip", RIRS_URL, RIRS_MD5)
        download_file(asset, downloads / asset.name, None, args.download_retries)


def maybe_delete_archive(archive: Path, downloads: Path, enabled: bool) -> None:
    if not enabled:
        return
    try:
        archive.resolve().relative_to(downloads.resolve())
    except ValueError:
        log(f"[extract] keeping external archive: {archive}")
        return
    archive.unlink(missing_ok=True)


def run_extract(args: argparse.Namespace, data_root: Path, downloads: Path) -> None:
    raw_root = data_root / "raw"
    if not args.vox2_source:
        vox2_destination = raw_root / "vox2_aac"
        for archive in get_vox2_archives(args, downloads):
            extract_archive(archive, vox2_destination)
            maybe_delete_archive(archive, downloads, args.delete_archives)

    if not args.musan_source:
        archive = get_single_archive(
            args.musan_archive, downloads / "musan.tar.gz", "MUSAN"
        )
        extract_archive(archive, raw_root)
        maybe_delete_archive(archive, downloads, args.delete_archives)

    if not args.rirs_source:
        archive = get_single_archive(
            args.rirs_archive, downloads / "rirs_noises.zip", "RIRS_NOISES"
        )
        extract_archive(archive, raw_root)
        maybe_delete_archive(archive, downloads, args.delete_archives)

    prepare_vox1_test(args, data_root, downloads)


def _has_direct_audio(root: Path, extensions: Sequence[str]) -> bool:
    if not root.is_dir():
        return False
    for extension in extensions:
        try:
            next(root.glob(f"*/*/*{extension}"))
            return True
        except StopIteration:
            pass
    return False


def locate_audio_root(root: Path, extensions: Sequence[str]) -> Path:
    """Locate the directory immediately above speaker/video/audio."""
    candidates = (
        root / "dev" / "aac",
        root / "aac",
        root / "dev" / "wav",
        root / "wav",
        root,
    )
    for candidate in candidates:
        if _has_direct_audio(candidate, extensions):
            return candidate

    for extension in extensions:
        try:
            sample = next(root.rglob(f"*{extension}"))
        except StopIteration:
            continue
        if len(sample.parents) >= 3:
            candidate = sample.parent.parent.parent
            if _has_direct_audio(candidate, extensions):
                return candidate
    raise FileNotFoundError(
        f"No speaker/video/audio tree with {extensions} found under {root}"
    )


def iter_direct_audio(root: Path, extensions: Sequence[str]) -> Iterator[Path]:
    extensions_lower = {extension.lower() for extension in extensions}
    for speaker in sorted(path for path in root.iterdir() if path.is_dir()):
        for video in sorted(path for path in speaker.iterdir() if path.is_dir()):
            for audio in sorted(path for path in video.iterdir() if path.is_file()):
                if audio.suffix.lower() in extensions_lower:
                    yield audio


def count_direct_audio(root: Path, extensions: Sequence[str]) -> int:
    return sum(1 for _ in iter_direct_audio(root, extensions))


def validate_count(
    label: str,
    actual: int,
    expected: int,
    allow_incomplete: bool,
) -> None:
    if actual == 0:
        raise RuntimeError(f"No {label} files found")
    if expected and actual != expected:
        message = f"{label} count is {actual:,}; expected {expected:,}"
        if not allow_incomplete:
            raise RuntimeError(message + " (use --allow-incomplete only for testing)")
        log(f"WARNING: {message}")


def normalize_vox1_wav_layout(source_root: Path, target_root: Path) -> Path:
    """Expose a VoxCeleb1 test tree as ``target_root/wav/<spk>/<video>``."""
    source_wav = locate_audio_root(source_root, (".wav",)).resolve()
    target_root.mkdir(parents=True, exist_ok=True)
    target_wav = target_root / "wav"

    if target_wav.exists() or target_wav.is_symlink():
        if target_wav.resolve() != source_wav:
            raise RuntimeError(
                f"Existing {target_wav} points to {target_wav.resolve()}, "
                f"not requested source {source_wav}"
            )
        return target_wav

    if source_wav == target_root.resolve():
        # Handle archives that contain speaker directories at their root
        # instead of the standard top-level wav/ directory.
        speaker_dirs = [
            path
            for path in target_root.iterdir()
            if path.is_dir() and any(path.glob("*/*.wav"))
        ]
        if not speaker_dirs:
            raise RuntimeError(f"No VoxCeleb1 speaker directories under {target_root}")
        target_wav.mkdir()
        for speaker_dir in speaker_dirs:
            shutil.move(str(speaker_dir), target_wav / speaker_dir.name)
        return target_wav

    target_wav.symlink_to(source_wav, target_is_directory=True)
    return target_wav


def find_vox1_protocol_source(
    args: argparse.Namespace,
    target_root: Path,
    downloads: Path,
) -> Path:
    candidates: list[Path] = []
    if args.vox1_protocol:
        candidates.append(Path(args.vox1_protocol).expanduser().resolve())
    if args.vox1_test_source:
        source_root = Path(args.vox1_test_source).expanduser().resolve()
        candidates.extend(
            (source_root / "veri_test2.txt", source_root.parent / "veri_test2.txt")
        )
    candidates.extend((target_root / "veri_test2.txt", downloads / "veri_test2.txt"))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "veri_test2.txt not found; run download or pass --vox1-protocol"
    )


def install_vox1_protocol(source: Path, target: Path) -> None:
    if source.resolve() == target.resolve():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    shutil.copyfile(source, temporary)
    temporary.replace(target)


def validate_vox1_test(
    root: Path,
    expected_files: int,
    expected_trials: int,
    allow_incomplete: bool,
) -> None:
    wav_root = root / "wav"
    protocol = root / "veri_test2.txt"
    if not wav_root.is_dir():
        raise FileNotFoundError(f"VoxCeleb1 test WAV root not found: {wav_root}")
    if not protocol.is_file():
        raise FileNotFoundError(f"VoxCeleb1 protocol not found: {protocol}")

    wav_count = count_direct_audio(wav_root, (".wav",))
    validate_count("VoxCeleb1 test WAV", wav_count, expected_files, allow_incomplete)

    references: set[str] = set()
    trial_count = 0
    with protocol.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            fields = line.split()
            if len(fields) != 3 or fields[0] not in {"0", "1"}:
                raise ValueError(f"Malformed protocol line {protocol}:{line_number}")
            for reference in fields[1:]:
                relative = Path(reference)
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError(
                        f"Unsafe protocol path {reference!r} at "
                        f"{protocol}:{line_number}"
                    )
                references.add(reference)
            trial_count += 1
    validate_count(
        "VoxCeleb1 verification trial",
        trial_count,
        expected_trials,
        allow_incomplete,
    )

    missing = [
        reference
        for reference in sorted(references)
        if not (wav_root / reference).is_file()
    ]
    if missing:
        examples = "\n".join(f"  {reference}" for reference in missing[:20])
        raise RuntimeError(
            f"VoxCeleb1 protocol references {len(missing):,} missing WAVs; first entries:\n"
            f"{examples}"
        )
    log(
        f"[vox1-test] {wav_count:,} WAVs, {trial_count:,} trials, "
        f"{len(references):,} referenced utterances"
    )


def prepare_vox1_test(
    args: argparse.Namespace,
    data_root: Path,
    downloads: Path,
) -> None:
    target_root = data_root / "vox1-test"
    if args.vox1_test_source:
        source_root = Path(args.vox1_test_source).expanduser().resolve()
    else:
        archive = get_single_archive(
            args.vox1_test_archive,
            downloads / archive_name(args.vox1_test_url),
            "VoxCeleb1 test",
        )
        extract_archive(archive, target_root)
        maybe_delete_archive(archive, downloads, args.delete_archives)
        source_root = target_root

    normalize_vox1_wav_layout(source_root, target_root)
    protocol_source = find_vox1_protocol_source(args, target_root, downloads)
    install_vox1_protocol(protocol_source, target_root / "veri_test2.txt")
    validate_vox1_test(
        target_root,
        args.expected_vox1_test_files,
        args.expected_vox1_trials,
        args.allow_incomplete,
    )


def _init_convert_worker(
    source_root: str,
    output_root: str,
    delete_source: bool,
    timeout: int,
) -> None:
    global _CONVERT_SOURCE_ROOT
    global _CONVERT_OUTPUT_ROOT
    global _CONVERT_DELETE_SOURCE
    global _CONVERT_TIMEOUT
    _CONVERT_SOURCE_ROOT = source_root
    _CONVERT_OUTPUT_ROOT = output_root
    _CONVERT_DELETE_SOURCE = delete_source
    _CONVERT_TIMEOUT = timeout


def _convert_one(source_name: str) -> tuple[str, str]:
    source = Path(source_name)
    relative = source.relative_to(_CONVERT_SOURCE_ROOT)
    output = Path(_CONVERT_OUTPUT_ROOT) / relative.with_suffix(".wav")
    try:
        if output.is_file() and output.stat().st_size >= MIN_WAV_BYTES:
            if _CONVERT_DELETE_SOURCE:
                source.unlink(missing_ok=True)
            return str(output), "skipped"
        output.parent.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(
            (
                "ffmpeg",
                "-nostdin",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(source),
                "-vn",
                "-ar",
                "16000",
                "-ac",
                "1",
                "-acodec",
                "pcm_s16le",
                str(output),
            ),
            capture_output=True,
            timeout=_CONVERT_TIMEOUT,
        )
        if result.returncode != 0:
            output.unlink(missing_ok=True)
            reason = result.stderr.decode("utf-8", errors="replace").strip()
            return str(
                output
            ), f"failed: ffmpeg rc={result.returncode}: {reason[-300:]}"
        if output.stat().st_size < MIN_WAV_BYTES:
            output.unlink(missing_ok=True)
            return str(output), "failed: output is too short"
        if _CONVERT_DELETE_SOURCE:
            source.unlink(missing_ok=True)
        return str(output), "converted"
    except subprocess.TimeoutExpired:
        output.unlink(missing_ok=True)
        return str(output), "failed: ffmpeg timeout"
    except Exception as error:  # worker must report failures to the parent
        output.unlink(missing_ok=True)
        return str(output), f"failed: {error}"


def _conversion_marker(vox2_dir: Path) -> Path:
    return vox2_dir / ".conversion_complete.json"


def run_convert(args: argparse.Namespace, data_root: Path) -> Path:
    vox2_dir = data_root / "vox2_dev"
    output_root = vox2_dir / "wav"
    marker = _conversion_marker(vox2_dir)
    if marker.is_file():
        state = json.loads(marker.read_text())
        marked_root = Path(state["wav_root"])
        if marked_root.is_dir() and state.get("count") == args.expected_vox2_files:
            log(f"[convert] cached: {marked_root}")
            return marked_root

    source_base = (
        Path(args.vox2_source).expanduser().resolve()
        if args.vox2_source
        else data_root / "raw" / "vox2_aac"
    )

    try:
        wav_source = locate_audio_root(source_base, (".wav",))
    except FileNotFoundError:
        wav_source = None
    if wav_source is not None:
        count = count_direct_audio(wav_source, (".wav",))
        validate_count(
            "VoxCeleb2 WAV", count, args.expected_vox2_files, args.allow_incomplete
        )
        log(f"[convert] source already contains WAV; using {wav_source}")
        return wav_source

    output_root.mkdir(parents=True, exist_ok=True)
    existing_count = count_direct_audio(output_root, (".wav",))
    if args.expected_vox2_files and existing_count == args.expected_vox2_files:
        marker.write_text(
            json.dumps(
                {"wav_root": str(output_root.resolve()), "count": existing_count},
                indent=2,
            )
            + "\n"
        )
        log(f"[convert] cached complete WAV tree: {output_root}")
        return output_root

    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required to convert VoxCeleb2 AAC/M4A")
    source_root = locate_audio_root(source_base, (".m4a", ".aac"))

    if args.delete_converted_source:
        try:
            source_root.resolve().relative_to((data_root / "raw").resolve())
        except ValueError as error:
            raise RuntimeError(
                "Refusing to delete files from external --vox2-source; "
                "omit --delete-converted-source"
            ) from error

    sources = [str(path) for path in iter_direct_audio(source_root, (".m4a", ".aac"))]
    if not sources:
        raise RuntimeError(f"No AAC/M4A files found under {source_root}")
    log(
        f"[convert] {len(sources):,} files from {source_root} -> {output_root} "
        f"with {args.workers} workers"
    )

    initializer_args = (
        str(source_root),
        str(output_root),
        args.delete_converted_source,
        args.ffmpeg_timeout,
    )
    converted = skipped = failed = 0
    failures: list[tuple[str, str]] = []
    if args.workers == 1:
        _init_convert_worker(*initializer_args)
        results: Iterable[tuple[str, str]] = map(_convert_one, sources)
        pool = None
    else:
        pool = mp.Pool(
            args.workers,
            initializer=_init_convert_worker,
            initargs=initializer_args,
        )
        results = pool.imap_unordered(_convert_one, sources, chunksize=64)

    try:
        for index, (output, status) in enumerate(results, start=1):
            if status == "converted":
                converted += 1
            elif status == "skipped":
                skipped += 1
            else:
                failed += 1
                if len(failures) < 50:
                    failures.append((output, status))
            if index % 10_000 == 0 or index == len(sources):
                log(
                    f"[convert] {index:,}/{len(sources):,}: "
                    f"converted={converted:,}, skipped={skipped:,}, failed={failed:,}"
                )
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    if failures:
        log("[convert] first failures:")
        for output, reason in failures:
            log(f"  {output}: {reason}")
    final_count = count_direct_audio(output_root, (".wav",))
    validate_count(
        "VoxCeleb2 WAV",
        final_count,
        args.expected_vox2_files,
        args.allow_incomplete,
    )
    if failed and not args.allow_incomplete:
        raise RuntimeError(f"VoxCeleb2 conversion failed for {failed:,} files")
    if args.expected_vox2_files and final_count == args.expected_vox2_files:
        marker.write_text(
            json.dumps(
                {"wav_root": str(output_root.resolve()), "count": final_count},
                indent=2,
            )
            + "\n"
        )
    return output_root


def resolve_vox2_wav_root(args: argparse.Namespace, data_root: Path) -> Path:
    if args.vox2_source:
        source = Path(args.vox2_source).expanduser().resolve()
        try:
            return locate_audio_root(source, (".wav",))
        except FileNotFoundError:
            pass
    output = data_root / "vox2_dev" / "wav"
    if output.is_dir():
        return output
    marker = _conversion_marker(data_root / "vox2_dev")
    if marker.is_file():
        return Path(json.loads(marker.read_text())["wav_root"])
    raise FileNotFoundError("VoxCeleb2 WAV tree not found; run the convert stage")


def atomic_replace(temp_paths: Sequence[tuple[Path, Path]]) -> None:
    for temporary, final in temp_paths:
        temporary.replace(final)


def write_vox2_manifests(
    wav_root: Path,
    output_dir: Path,
    expected_count: int,
    allow_incomplete: bool,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    finals = {
        "data": output_dir / "data.list",
        "utt": output_dir / "utt2spk",
        "spk": output_dir / "spk2utt",
        "wav": output_dir / "wav.scp",
    }
    temps = {name: path.with_name(path.name + ".tmp") for name, path in finals.items()}
    count = speakers = 0
    try:
        with (
            temps["data"].open("w", encoding="utf-8") as data_stream,
            temps["utt"].open("w", encoding="utf-8") as utt_stream,
            temps["spk"].open("w", encoding="utf-8") as spk_stream,
            temps["wav"].open("w", encoding="utf-8") as wav_stream,
        ):
            for speaker_dir in sorted(
                path for path in wav_root.iterdir() if path.is_dir()
            ):
                speaker = speaker_dir.name
                speaker_keys: list[str] = []
                for video_dir in sorted(
                    path for path in speaker_dir.iterdir() if path.is_dir()
                ):
                    for wav in sorted(video_dir.glob("*.wav")):
                        key = f"{speaker}/{video_dir.name}/{wav.name}"
                        absolute = str(wav.resolve())
                        data_stream.write(
                            json.dumps(
                                {"key": key, "spk": speaker, "wav": absolute},
                                separators=(",", ":"),
                            )
                            + "\n"
                        )
                        utt_stream.write(f"{key} {speaker}\n")
                        wav_stream.write(f"{key} {absolute}\n")
                        speaker_keys.append(key)
                        count += 1
                if speaker_keys:
                    spk_stream.write(f"{speaker} {' '.join(speaker_keys)}\n")
                    speakers += 1
        validate_count("VoxCeleb2 manifest", count, expected_count, allow_incomplete)
        atomic_replace([(temps[name], finals[name]) for name in finals])
    except Exception:
        for path in temps.values():
            path.unlink(missing_ok=True)
        raise
    log(f"[manifests] VoxCeleb2: {count:,} utterances, {speakers:,} speakers")
    return count


def resolve_augmentation_root(
    explicit: str | None,
    default_root: Path,
    nested_name: str,
) -> Path:
    root = Path(explicit).expanduser().resolve() if explicit else default_root
    nested = root / nested_name
    if nested.is_dir():
        root = nested
    if not root.is_dir():
        raise FileNotFoundError(f"Augmentation source directory not found: {root}")
    return root


def write_audio_scp(
    source_root: Path,
    output: Path,
    label: str,
    expected_count: int,
    allow_incomplete: bool,
) -> int:
    files = sorted(source_root.rglob("*.wav"))
    validate_count(label, len(files), expected_count, allow_incomplete)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for wav in files:
            relative = wav.relative_to(source_root)
            parts = relative.parts[-3:]
            key = Path(*parts).as_posix()
            absolute = str(wav.resolve())
            if any(character.isspace() for character in absolute):
                raise RuntimeError(
                    f"wav.scp paths cannot contain whitespace: {absolute}"
                )
            stream.write(f"{key} {absolute}\n")
    temporary.replace(output)
    log(f"[manifests] {label}: {len(files):,} files -> {output}")
    return len(files)


def run_manifests(
    args: argparse.Namespace,
    data_root: Path,
    wav_root: Path | None,
) -> None:
    validate_vox1_test(
        data_root / "vox1-test",
        args.expected_vox1_test_files,
        args.expected_vox1_trials,
        args.allow_incomplete,
    )
    wav_root = wav_root or resolve_vox2_wav_root(args, data_root)
    write_vox2_manifests(
        wav_root,
        data_root / "vox2_dev",
        args.expected_vox2_files,
        args.allow_incomplete,
    )

    raw_root = data_root / "raw"
    musan_root = resolve_augmentation_root(
        args.musan_source, raw_root / "musan", "musan"
    )
    rirs_root = resolve_augmentation_root(
        args.rirs_source,
        raw_root / "RIRS_NOISES",
        "simulated_rirs",
    )
    write_audio_scp(
        musan_root,
        data_root / "musan" / "wav.scp",
        "MUSAN",
        args.expected_musan_files,
        args.allow_incomplete,
    )
    write_audio_scp(
        rirs_root,
        data_root / "rirs" / "wav.scp",
        "RIRS simulated_rirs",
        args.expected_rirs_files,
        args.allow_incomplete,
    )


def read_scp(path: Path) -> list[tuple[str, Path]]:
    entries: list[tuple[str, Path]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            fields = line.rstrip("\n").split(maxsplit=1)
            if len(fields) != 2:
                raise ValueError(f"Malformed {path}:{line_number}")
            key, filename = fields
            audio = Path(filename)
            if not audio.is_file():
                raise FileNotFoundError(
                    f"Missing audio in {path}:{line_number}: {audio}"
                )
            entries.append((key, audio))
    return entries


def lmdb_complete(path: Path, expected_count: int) -> bool:
    if not (path / "data.mdb").is_file():
        return False
    try:
        import lmdb

        database = lmdb.open(str(path), readonly=True, lock=False, readahead=False)
        with database.begin(write=False) as transaction:
            encoded = transaction.get(b"__keys__")
        database.close()
        return encoded is not None and len(pickle.loads(encoded)) == expected_count
    except Exception:
        return False


def make_lmdb(
    scp_path: Path,
    output: Path,
    commit_interval: int,
    rebuild: bool,
) -> None:
    try:
        import lmdb
    except ImportError as error:
        raise RuntimeError(
            "The 'lmdb' package is required for the lmdb stage: pip install lmdb"
        ) from error

    entries = read_scp(scp_path)
    if lmdb_complete(output, len(entries)):
        log(f"[lmdb] cached: {output}")
        return
    if output.exists():
        if not rebuild:
            raise RuntimeError(
                f"Incomplete LMDB exists at {output}; rerun with --rebuild-lmdb"
            )
        shutil.rmtree(output)

    temporary = output.with_name(output.name + ".tmp")
    if temporary.exists():
        if not rebuild:
            raise RuntimeError(
                f"Interrupted LMDB build exists at {temporary}; "
                "rerun with --rebuild-lmdb"
            )
        shutil.rmtree(temporary)

    total_bytes = sum(audio.stat().st_size for _, audio in entries)
    map_size = max(1 << 30, int(total_bytes * 1.25) + (256 << 20))
    temporary.parent.mkdir(parents=True, exist_ok=True)
    database = lmdb.open(str(temporary), map_size=map_size)
    keys: list[str] = []
    transaction = database.begin(write=True)
    try:
        for index, (key, audio) in enumerate(entries, start=1):
            if not transaction.put(key.encode(), audio.read_bytes(), overwrite=False):
                raise RuntimeError(f"Duplicate LMDB key {key!r} from {scp_path}")
            keys.append(key)
            if index % commit_interval == 0:
                transaction.commit()
                transaction = database.begin(write=True)
            if index % 5_000 == 0 or index == len(entries):
                log(f"[lmdb] {output.name}: {index:,}/{len(entries):,}")
        transaction.commit()
        transaction = None
        with database.begin(write=True) as keys_transaction:
            keys_transaction.put(b"__keys__", pickle.dumps(keys))
        database.sync()
    except Exception:
        if transaction is not None:
            transaction.abort()
        database.close()
        raise
    database.close()
    temporary.replace(output)
    log(f"[lmdb] wrote {len(keys):,} entries to {output}")


def run_lmdb(args: argparse.Namespace, data_root: Path) -> None:
    make_lmdb(
        data_root / "musan" / "wav.scp",
        data_root / "musan" / "lmdb",
        args.lmdb_commit_interval,
        args.rebuild_lmdb,
    )
    make_lmdb(
        data_root / "rirs" / "wav.scp",
        data_root / "rirs" / "lmdb",
        args.lmdb_commit_interval,
        args.rebuild_lmdb,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare training, augmentation, and validation data for VoxCeleb2."
    )
    parser.add_argument("--data-root", required=True, help="Output dataset root")
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=("all",) + STAGE_ORDER,
        default=("all",),
        help="Ordered stages to execute (default: all)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(32, os.cpu_count() or 1),
        help="Parallel ffmpeg workers",
    )
    parser.add_argument("--ffmpeg-timeout", type=int, default=120)
    parser.add_argument("--download-retries", type=int, default=5)
    parser.add_argument(
        "--hf-token",
        default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN"),
        help="Hugging Face token (defaults to HF_TOKEN/HUGGINGFACE_TOKEN)",
    )
    parser.add_argument(
        "--vox2-url",
        action="append",
        help="VoxCeleb2 archive URL; repeat for multipart releases",
    )
    parser.add_argument(
        "--vox2-archive",
        action="append",
        default=[],
        help="Existing VoxCeleb2 archive; repeat for multipart releases",
    )
    parser.add_argument(
        "--vox2-source",
        help="Existing extracted AAC/M4A or WAV tree; bypass download/extraction",
    )
    parser.add_argument(
        "--vox1-test-url",
        default=VOX1_TEST_URL,
        help="VoxCeleb1 test WAV archive URL",
    )
    parser.add_argument(
        "--vox1-protocol-url",
        default=VOX1_PROTOCOL_URL,
        help="VoxCeleb1 veri_test2.txt URL",
    )
    parser.add_argument(
        "--vox1-test-archive",
        help="Existing vox1_test_wav.zip",
    )
    parser.add_argument(
        "--vox1-test-source",
        help="Existing extracted VoxCeleb1 test root or WAV directory",
    )
    parser.add_argument(
        "--vox1-protocol",
        help="Existing veri_test2.txt",
    )
    parser.add_argument("--musan-archive", help="Existing musan.tar.gz")
    parser.add_argument("--musan-source", help="Existing extracted MUSAN directory")
    parser.add_argument("--rirs-archive", help="Existing rirs_noises.zip")
    parser.add_argument("--rirs-source", help="Existing extracted RIRS directory")
    parser.add_argument("--delete-archives", action="store_true")
    parser.add_argument(
        "--delete-converted-source",
        action="store_true",
        help="Delete extracted AAC/M4A files after successful conversion",
    )
    parser.add_argument("--expected-vox2-files", type=int, default=DEFAULT_VOX2_FILES)
    parser.add_argument(
        "--expected-vox1-test-files",
        type=int,
        default=DEFAULT_VOX1_TEST_FILES,
    )
    parser.add_argument(
        "--expected-vox1-trials",
        type=int,
        default=DEFAULT_VOX1_TRIALS,
    )
    parser.add_argument("--expected-musan-files", type=int, default=DEFAULT_MUSAN_FILES)
    parser.add_argument("--expected-rirs-files", type=int, default=DEFAULT_RIRS_FILES)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Permit nonstandard file counts (intended for tests/subsets only)",
    )
    parser.add_argument("--lmdb-commit-interval", type=int, default=100)
    parser.add_argument(
        "--rebuild-lmdb",
        action="store_true",
        help="Replace an incomplete/interrupted generated LMDB",
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.lmdb_commit_interval < 1:
        parser.error("--lmdb-commit-interval must be at least 1")
    if "all" in args.stages and len(args.stages) > 1:
        parser.error("Use --stages all by itself")
    return args


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root).expanduser().resolve()
    downloads = data_root / "downloads"
    data_root.mkdir(parents=True, exist_ok=True)
    stages = STAGE_ORDER if "all" in args.stages else tuple(args.stages)

    wav_root: Path | None = None
    for stage in STAGE_ORDER:
        if stage not in stages:
            continue
        log(f"\n=== {stage} ===")
        if stage == "download":
            run_download(args, downloads)
        elif stage == "extract":
            run_extract(args, data_root, downloads)
        elif stage == "convert":
            wav_root = run_convert(args, data_root)
        elif stage == "manifests":
            run_manifests(args, data_root, wav_root)
        elif stage == "lmdb":
            run_lmdb(args, data_root)

    log("\nSelected stages complete. Expected training paths:")
    log(f"  train_data: {data_root / 'vox2_dev' / 'data.list'}")
    log(f"  train_label: {data_root / 'vox2_dev' / 'utt2spk'}")
    log(f"  noise_lmdb_file: {data_root / 'musan' / 'lmdb'}")
    log(f"  reverb_lmdb_file: {data_root / 'rirs' / 'lmdb'}")
    log(f"  validation.vox1: {data_root / 'vox1-test'}")


if __name__ == "__main__":
    main()
