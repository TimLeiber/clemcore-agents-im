from clemcore.agents.runner import DEFAULT_INSTRUCTION, run_external_agent_episode


REGISTRY_PATH = "~/im_workspace/clembench/agent_registry.json"
OUTPUT_ROOT = "~/im_workspace/clembench/openenv-records/external-agents"


def main() -> None:
    result = run_external_agent_episode(
        agent_name="claude-code-sonnet",
        registry_path=REGISTRY_PATH,
        output_root=OUTPUT_ROOT,
        instruction=DEFAULT_INSTRUCTION,
    )

    print("success:", result.success)

    for name, path in result.artifacts.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()