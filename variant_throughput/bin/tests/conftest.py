"""Put the parent `bin/` dir on sys.path so tests import sibling modules by bare
name (`_streaming`, `_pairs`) exactly as Nextflow does on PATH."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
