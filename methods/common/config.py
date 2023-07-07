from typing import Optional
from dataclasses import dataclass
import json

@dataclass
class Config:
    vcs_id : str
    country_code : str
    project_start: int
    # agb : Optional[[float]]

def from_file(filename : str) -> Config:
    with open(filename) as json_file:
        data = json.load(json_file)
        config = Config(
            vcs_id=data["vcs_id"],
            country_code=data["country_code"],
            project_start=data["project_start"]
        )
        return config