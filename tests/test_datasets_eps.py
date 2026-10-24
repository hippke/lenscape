"""Offline parts of lenscape.datasets (safe archive extraction) and lenscape.eps (vector data from an EPS)."""
import io, os, tarfile
import numpy as np
from lenscape import datasets, eps


def test_extract_skips_unsafe_members(tmp_path):
    arch = tmp_path / "a.tar.gz"
    with tarfile.open(arch, "w:gz") as tf:
        for name, data in (("ok/KMTC22_I.pysis", b"1 2 3"), ("../evil.txt", b"x"), ("/abs.txt", b"y")):
            info = tarfile.TarInfo(name); info.size = len(data); tf.addfile(info, io.BytesIO(data))
        link = tarfile.TarInfo("ok/link"); link.type = tarfile.SYMTYPE; link.linkname = "/etc/passwd"; tf.addfile(link)
    dest = tmp_path / "out"
    datasets._extract(str(arch), str(dest))
    assert (dest / "ok" / "KMTC22_I.pysis").read_bytes() == b"1 2 3"
    assert not (tmp_path / "evil.txt").exists() and not (dest / "ok" / "link").exists()


EPS = """%!PS-Adobe-2.0 EPSF-2.0
gsave
0 0 0 C 100 100 M 900 100 L
255 99 71 C 400 500 M 408 500 L 410 504 L 408 508 L 400 508 L 398 504 L 399 501 L 400 500 L
255 99 71 C 404 480 M 404 528 L
"""


def test_eps_marker_with_error_bar(tmp_path):
    p = tmp_path / "f.eps"
    p.write_text(EPS)
    subs = eps.subpaths(str(p))
    assert len(subs) == 3
    pts = eps.points_with_errors(subs, eps.calibrate(100, 0.0, 900, 8.0), eps.calibrate(100, 20.0, 900, 12.0),
                                 (0, 1000, 0, 1000), {(255, 99, 71): "KMTC"}, size=(5, 30))
    name, x, y, e = pts[0]
    assert name == "KMTC" and np.isclose(x, 3.04) and np.isclose(y, 15.96) and np.isclose(e, 0.24)
