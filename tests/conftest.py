import os, sys
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))           # run the tests without installing the package
REF = os.path.join(ROOT, "tests", "data")               # stored VBBinaryLensing reference magnifications
