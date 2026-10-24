"""Public data for examples and tutorials, downloaded on first use and cached on disk.

- kmtnet(event): KMTNet pySIS photometry from the KMTNet event archive (https://kmtnet.kasi.re.kr/~ulens/).
  Please acknowledge KMTNet when you use these data.
- arxiv_source(arxiv_id): the LaTeX source of an arXiv paper (figures included), e.g. to read data points from
  vector graphics with lenscape.eps.

Cache directory: $LENSCAPE_DATA, else ~/.cache/lenscape/data.
"""
import os, re, tarfile, urllib.request
from . import data as D

KMTNET = "https://kmtnet.kasi.re.kr/~ulens/event/{year}/data/KB{yy}{num:04d}/pysis/pysis.tar.gz"
ARXIV = "https://arxiv.org/e-print/{id}"


def cache_dir():
    d = os.environ.get("LENSCAPE_DATA") or os.path.join(os.path.expanduser("~"), ".cache", "lenscape", "data")
    os.makedirs(d, exist_ok=True)
    return d


def download(url, path):
    """Download url to path (skipped if the file exists); returns path."""
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        req = urllib.request.Request(url, headers={"User-Agent": "lenscape"})
        with urllib.request.urlopen(req, timeout=120) as r, open(path + ".part", "wb") as f:
            f.write(r.read())
        os.replace(path + ".part", path)
    return path


def _extract(archive, dest):
    """Extract a tar archive into dest, refusing absolute paths, '..' and links (archives are untrusted)."""
    os.makedirs(dest, exist_ok=True)
    root = os.path.realpath(dest)
    with tarfile.open(archive) as tf:
        members = []
        for m in tf.getmembers():
            target = os.path.realpath(os.path.join(dest, m.name))
            if not (m.isfile() or m.isdir()) or not target.startswith(root + os.sep):
                continue
            members.append(m)
        if hasattr(tarfile, "data_filter"):                    # Python >= 3.12 (also 3.10.12+, 3.11.4+)
            tf.extractall(dest, members=members, filter="data")
        else:
            tf.extractall(dest, members=members)
    return dest


def kmtnet(event, band="I", window=None, t_offset=2450000.0):
    """KMTNet pySIS photometry of an event, e.g. kmtnet("KMT-2022-BLG-0114") -> Data with the sets KMTA22, KMTC22,
    KMTS22 (band "I" or "V"). Time axis HJD - t_offset; window = (t1, t2) on that axis."""
    m = re.fullmatch(r"KMT-(\d{4})-BLG-(\d{4})", event)
    if not m:
        raise ValueError(f"event name {event!r}: expected KMT-YYYY-BLG-NNNN")
    year, num = int(m.group(1)), int(m.group(2))
    dest = os.path.join(cache_dir(), event)
    tgz = download(KMTNET.format(year=year, yy=year % 100, num=num), os.path.join(dest, "pysis.tar.gz"))
    _extract(tgz, os.path.join(dest, "pysis"))
    files = sorted(os.path.join(r, f) for r, _, fs in os.walk(os.path.join(dest, "pysis")) for f in fs
                   if f.endswith(f"_{band}.pysis"))
    return D.read_pysis(files, t_offset=t_offset, window=window)


def arxiv_source(arxiv_id):
    """Download and extract the source of an arXiv paper; returns the directory with its files."""
    dest = os.path.join(cache_dir(), "arxiv", arxiv_id)
    arch = download(ARXIV.format(id=arxiv_id), os.path.join(dest, "source.tar.gz"))
    return _extract(arch, os.path.join(dest, "src"))
