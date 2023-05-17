# File path resolution across space and time.
# Something like DNS might work well here, maybe. The idea being there
# should be no hardcoded paths to files or data anywhere making it portable
# to a distributed pipeline or similar. If you want to save something you ask
# for a path and give a local, programming-level name to this object. For now,
# this is crudely done via JSON mappings that can be merged a abstracted upon
# later.
import os
import json
from typing import Dict

# This is not efficient, or concurrent-safe... but it'll do.
class Arkpath:
    base_dir = os.getenv("ARKDIR", default=os.getcwd())
    paths: Dict[str, str] = {}

    def _refresh(self):
        path = os.path.join(self.base_dir, "arkdir.json")
        if os.path.exists(path):
            with open(path, "r") as fp:
                self.paths = json.load(fp)

    def save(self, name: str) -> str:
        self._refresh()
        path = os.path.join(self.base_dir, name)
        self.paths[name] = path
        with open(os.path.join(self.base_dir, "arkdir.json"), "w") as fp:
            json.dump(self.paths,fp)
        return path

    def load(self, name: str) -> str:
        maybe = self.paths.get(name)
        if maybe is None:
            self._refresh()
            maybe = self.paths.get(name)
            if maybe is None:
                raise NameError(name)
            else:
                return maybe
        else:
            return maybe