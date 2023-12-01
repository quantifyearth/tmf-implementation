import os
from enum import IntEnum


# A directory for storing intermediate or partial results
partials_dir = os.getenv("TMF_PARTIALS")

if partials_dir is not None:
    os.makedirs(partials_dir, exist_ok=True)

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
