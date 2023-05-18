import os
import json
from typing import Dict, Any

# This is not efficient, or concurrent-safe... but it'll do. The schema is relatively simple.
# A arkdir.json file contains mappings for objects (blobs of data). An entry is simply:
#
#   { base_dir : string; map : (key * relative_path) list }
#
# The file path for an object is obtained by concating base_dir and relative_path.
# The arkdir.json file is just a list of such objects. Most applications will only need
# one entry most likely, but in the case of merging two independent outputs into one
# input, we need the ability to have more than one. Note that the keys should be
# _globally_ unique for files which means we could in theory merge everything into one arkdir
# but that doesn't work with mounting volumes, so for now that is up to the programmer to do
# that correctly.
#
# A running program should only ever save files into its unique entry (which may need creating).
# The weak invariant is that (unless you are _incredibly_ unlucky), the program's ARKDIR shouldn't conflict with
# any others that might already exist. We use the unique hash of the build ID in the pipeline
# to ensure this.

class Arkpath:
    base_dir = os.getenv("ARKDIR", default=os.getcwd())
    paths: Dict[str, Dict[str, str]] = {}

    # TODO: this is inefficient
    def _refresh(self):
        # A list of entries
        path = os.path.join(self.base_dir, "arkdir.json")
        try:
            with open(path, "r", encoding="utf-8") as file_pointer:
                entries = json.load(file_pointer)
                new_paths = {}
                for entry in entries:
                    base_dir = entry["base_dir"]
                    kvs = entry["map"]
                    new_paths[base_dir] = kvs
                self.paths = new_paths
        except OSError:
            pass

    # TODO: distinguish between names and relative paths, for now they are same.
    def save(self, name: str) -> str:
        self._refresh()
        if self.paths.get(self.base_dir) is None:
            self.paths[self.base_dir] = {}
        self.paths[self.base_dir][name] = name
        with open(os.path.join(self.base_dir, "arkdir.json"), "w", encoding="utf-8") as file_pointer:
            # Reconstruct the schema
            dump = []
            for base, entry in self.paths.items():
                bdir: Any = {}
                bdir["base_dir"] = base
                bdir["map"] = entry
                dump.append(bdir)
            json.dump(dump, file_pointer)
        return os.path.join(self.base_dir, name)

    def lookup(self, name: str):
        for base, entry in self.paths.items():
            try:
                value = entry[name]
                return os.path.join(base, value)
            except KeyError:
                pass
        return None

    def load(self, name: str) -> str:
        maybe = self.lookup(name)
        if maybe is None:
            self._refresh()
            maybe = self.lookup(name)
            if maybe is None:
                raise NameError(name)
        return maybe