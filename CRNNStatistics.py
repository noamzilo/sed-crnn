import os
import csv
import json
from typing import List, Dict, Any, Optional

class CRNNStatistics:
    """
    Collects and dumps statistics for SED-CRNN video inference runs.
    """
    def __init__(self):
        self.stats: List[Dict[str, Any]] = []

    def add(self, stat: Dict[str, Any]) -> None:
        """Add statistics for a single video."""
        self.stats.append(stat)

    def dump_csv(self, path: str, fieldnames: Optional[List[str]] = None) -> None:
        """Dump all statistics to a CSV file at the given path."""
        if not self.stats:
            raise ValueError("No statistics to dump.")
        if fieldnames is None:
            # Use keys from the first stat as fieldnames
            fieldnames = list(self.stats[0].keys())
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for stat in self.stats:
                writer.writerow(stat)

    def dump_json(self, path: str) -> None:
        """Dump all statistics to a JSON file at the given path."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, indent=2) 