"""Teacher Capability Registry (Stage 3).

Strict separation of test vs agent-trainable information:
- ``research_report.json`` may contain formal test deltas (paper evidence only).
- ``agent_train_registry.json`` contains only train/val info (Agent input; no leakage).

P0 hardening (review 2026-10-08): the accessor is an explicit enum, the Agent-side
loader **refuses** any ``test_*`` field (fail-fast), and the research report can only
be opened through a dedicated research-only API that is not used by Agent tooling.
"""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Dict


class RegistryKind(str, Enum):
    """Only these two registries exist; anything else is an error."""

    AGENT_TRAIN = "agent_train"
    RESEARCH = "research"


AGENT_FORBIDDEN_PREFIX = "test_"


def _assert_no_test_leak(node, path: str = "root") -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            if isinstance(key, str) and key.startswith(AGENT_FORBIDDEN_PREFIX):
                raise PermissionError(
                    f"agent_train_registry contains forbidden test field: {path}.{key}"
                )
            _assert_no_test_leak(value, f"{path}.{key}")


class TeacherRegistry:
    def __init__(self, registry_dir: str | Path):
        self.root = Path(registry_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.research_path = self.root / "research_report.json"
        self.agent_path = self.root / "agent_train_registry.json"

    # ---- paths -------------------------------------------------------------
    def path_of(self, kind: RegistryKind) -> Path:
        return self.research_path if kind is RegistryKind.RESEARCH else self.agent_path

    # ---- generic access (enum-only) ---------------------------------------
    def load(self, kind: RegistryKind) -> Dict:
        if not isinstance(kind, RegistryKind):
            raise TypeError("kind must be a RegistryKind member")
        path = self.path_of(kind)
        if not path.is_file():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    def save(self, data: Dict, kind: RegistryKind) -> None:
        if not isinstance(kind, RegistryKind):
            raise TypeError("kind must be a RegistryKind member")
        if kind is RegistryKind.AGENT_TRAIN:
            _assert_no_test_leak(data)
        self.path_of(kind).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---- Agent-side API (hard leak guard) ---------------------------------
    def load_agent_train(self) -> Dict:
        data = self.load(RegistryKind.AGENT_TRAIN)
        _assert_no_test_leak(data)
        return data

    # ---- research-only API (never called by Agent tooling) ----------------
    def load_research(self) -> Dict:
        return self.load(RegistryKind.RESEARCH)

    def record_research(self, teacher_id: str, dataset: str, entry: Dict) -> None:
        data = self.load_research()
        data.setdefault(teacher_id, {})[dataset] = entry
        self.save(data, RegistryKind.RESEARCH)

    def record_agent_train(self, teacher_id: str, dataset: str, entry: Dict) -> None:
        data = self.load_agent_train()
        data.setdefault(teacher_id, {})[dataset] = entry
        self.save(data, RegistryKind.AGENT_TRAIN)


__all__ = ["TeacherRegistry", "RegistryKind", "AGENT_FORBIDDEN_PREFIX"]
