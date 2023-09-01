import os
from enum import IntEnum

# Probably a better name and method for doing this, but I thought
# we could use this to put other intermediate artefacts and graphs
dump_dir = os.getenv("DUMPDIR")

class DownloadError(Exception):
    def __init__(self, status_code: int, reason: str, url: str):
        self.status_code = status_code
        self.reason = reason
        self.url = url

    @property
    def msg(self) -> str:
        return f"Download failed, status {self.status_code}: {self.reason}"

class LandUseClass(IntEnum):
    UNDISTURBED = 1
    DEGRADED = 2
    DEFORESTED = 3
    REGROWTH = 4
    WATER = 5
    OTHER = 6
