import os

from clemcore.agents.runner import DEFAULT_INSTRUCTION, run_external_agent_episode


result = run_external_agent_episode(
    agent_name=os.environ["CLEM_AGENT_NAME"],
    registry_path="/app/agent_registry.json",
    output_root=None,
    instruction=DEFAULT_INSTRUCTION,
)

print("success:", result.success)

for name, path in result.artifacts.items():
    print(f"{name}: {path}")
