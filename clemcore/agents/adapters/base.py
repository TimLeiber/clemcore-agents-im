from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class AgentRunResult:
    success: bool
    artifacts: dict[str, Path | str | int | float | bool | None] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class ExternalAgentHarness(ABC):
    @abstractmethod
    def run_episode(self,
                    instruction: str,
                    output_dir: Path | str | None = None) -> AgentRunResult:
        pass