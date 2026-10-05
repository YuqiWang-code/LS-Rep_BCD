"""Teacher Capability Registry (Stage 3).

Strict separation of test vs agent-trainable information:
- ``research_report.json`` may contain formal test deltas (paper evidence only).
- ``agent_train_registry.json`` contains only train/val info (Agent input; no test leakage).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict


class TeacherRegistry:
    def __init__(self, registry_dir: str | Path):
        self.root = Path(registry_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.research_path = self.root / "research_report.json"
        self.agent_path = self.root / "agent_train_registry.json"

    def load(self, which: str) -> Dict:
        path = self.research_path if which == "research" else self.agent_path
        if not path.is_file():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    def save(self, data: Dict, which: str) -> None:
        path = self.research_path if which == "research" else self.agent_path
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def record_research(self, teacher_id: str, dataset: str, entry: Dict) -> None:
        data = self.load("research")
        data.setdefault(teacher_id, {})[dataset] = entry
        self.save(data, "research")

    def record_agent_train(self, teacher_id: str, dataset: str, entry: Dict) -> None:
        data = self.load("agent")
        data.setdefault(teacher_id, {})[dataset] = entry
        self.save(data, "agent")


__all__ = ["TeacherRegistry"]
