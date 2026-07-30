from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class AgentRunResult:
    """Store the standardized result returned by an agent harness.

    Attributes:
        success: whether the harness completed the episode successfully
        artifacts: files and scalar values produced during the run
        metadata: structured details describing the run
    """

    success: bool
    artifacts: dict[str, Path | str | int | float | bool | None] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class ExternalAgentHarness(ABC):
    """Define the interface implemented by every external-agent harness."""

    @abstractmethod
    def run_episode(self,
                    instruction: str,
                    output_dir: Path | str | None = None) -> AgentRunResult:
        """Run one agent episode.

        Args:
            instruction: task instruction passed to the external agent
            output_dir: optional directory for adapter artifacts

        Returns:
            the standardized result of the agent run
        """

        raise NotImplementedError
