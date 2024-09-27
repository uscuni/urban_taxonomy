import contextlib
from importlib.metadata import PackageNotFoundError, version

from . import utils

with contextlib.suppress(PackageNotFoundError):
    __version__ = version("core")
