import json
import os
from pathlib import Path
from typing import Any

from clemcore.backends import ModelRegistry


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _model_spec_to_dict(model_spec: Any) -> dict[str, Any]:
    if isinstance(model_spec, dict):
        return dict(model_spec)

    return {
        "model_name": getattr(model_spec, "model_name", None),
        "model_id": getattr(model_spec, "model_id", None),
        "backend": getattr(model_spec, "backend", None),
        "model_config": getattr(model_spec, "model_config", None),
    }


def _find_model_spec(clem_model: str) -> dict[str, Any]:
    registry = ModelRegistry.from_packaged_and_cwd_files()

    for model_spec in registry:
        model_spec_dict = _model_spec_to_dict(model_spec)

        if model_spec_dict.get("model_name") == clem_model:
            return model_spec_dict

    raise KeyError(
        f"Could not find clem_model={clem_model!r}. "
        "Run this from the clembench directory so model_registry.json is discoverable."
    )


def _load_key_json() -> dict[str, Any]:
    candidates = []

    if os.environ.get("CLEM_KEY_FILE"):
        candidates.append(Path(os.environ["CLEM_KEY_FILE"]).expanduser())

    candidates.extend([
        Path.cwd() / "key.json",
        Path.home() / ".clemcore" / "key.json",
    ])

    merged = {}

    for path in reversed(candidates):
        if path.exists():
            data = _read_json(path)

            if isinstance(data, dict):
                merged.update(data)

    return merged


def _openrouter_key_config() -> dict[str, Any]:
    key_json = _load_key_json()
    config = key_json.get("openrouter", {})

    if not isinstance(config, dict):
        return {}

    return config


def _openrouter_api_key(config: dict[str, Any]) -> str:
    api_key = config.get("api_key") or os.environ.get("OPENROUTER_API_KEY")

    if not api_key:
        raise RuntimeError(
            "No OpenRouter API key found. Expected clembench/key.json entry "
            "'openrouter.api_key' or env var OPENROUTER_API_KEY."
        )

    return str(api_key)


def _openai_key_config() -> dict[str, Any]:
    key_json = _load_key_json()
    config = key_json.get("openai", {})

    if not isinstance(config, dict):
        return {}

    return config


def _openai_api_key(config: dict[str, Any]) -> str:
    api_key = config.get("api_key") or os.environ.get("OPENAI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "No OpenAI API key found. Expected clembench/key.json entry "
            "'openai.api_key' or env var OPENAI_API_KEY."
        )

    return str(api_key)


def _openai_compatible_key_config() -> dict[str, Any]:
    key_json = _load_key_json()
    config = key_json.get("openai_compatible", {})

    if not isinstance(config, dict):
        return {}

    return config


def _openai_compatible_api_key(config: dict[str, Any]) -> str:
    api_key = (
        config.get("api_key")
        or os.environ.get("OPENAI_COMPATIBLE_API_KEY")
        or os.environ.get("CLEM_OPENAI_COMPATIBLE_API_KEY")
    )

    if not api_key:
        raise RuntimeError(
            "No OpenAI-compatible API key found. Expected clembench/key.json entry "
            "'openai_compatible.api_key' or env var OPENAI_COMPATIBLE_API_KEY."
        )

    return str(api_key)


def _openai_compatible_base_url(config: dict[str, Any]) -> str:
    base_url = config.get("base_url") or os.environ.get("OPENAI_COMPATIBLE_BASE_URL")

    if not base_url:
        raise RuntimeError(
            "No OpenAI-compatible base URL found. Expected clembench/key.json entry "
            "'openai_compatible.base_url' or env var OPENAI_COMPATIBLE_BASE_URL."
        )

    return str(base_url).rstrip("/")


def _model_context_window(model_spec: dict[str, Any]) -> int | None:
    context_size = model_spec.get("context_size")

    if context_size is None:
        return None

    if isinstance(context_size, int):
        return context_size

    text = str(context_size).strip().lower().replace(",", "")

    if text.endswith("k") and text[:-1].replace(".", "", 1).isdigit():
        return int(float(text[:-1]) * 1000)

    if text.isdigit():
        return int(text)

    return None


def _model_reasoning_enabled(model_spec: dict[str, Any]) -> bool:
    model_config = model_spec.get("model_config") or {}

    if not isinstance(model_config, dict):
        return False

    extra_body = model_config.get("extra_body") or {}

    if not isinstance(extra_body, dict):
        return False

    chat_template_kwargs = extra_body.get("chat_template_kwargs") or {}

    if isinstance(chat_template_kwargs, dict):
        enable_thinking = chat_template_kwargs.get("enable_thinking")

        if isinstance(enable_thinking, bool):
            return enable_thinking

    reasoning = extra_body.get("reasoning") or {}

    if isinstance(reasoning, dict):
        effort = reasoning.get("effort")

        if effort == "none":
            return False

        if effort is not None:
            return True

    return False


def _openrouter_anthropic_base_url(config: dict[str, Any]) -> str:
    base_url = str(config.get("base_url") or "https://openrouter.ai/api").rstrip("/")

    # clemcore/OpenRouter configs often use the OpenAI-compatible /api/v1
    # endpoint. Claude Code needs OpenRouter's Anthropic-compatible endpoint.
    if base_url.endswith("/api/v1"):
        return base_url[:-3]

    return base_url


def _openrouter_openai_base_url(config: dict[str, Any]) -> str:
    base_url = str(config.get("base_url") or "https://openrouter.ai/api/v1").rstrip("/")

    # OpenAI-compatible clients need OpenRouter's OpenAI-compatible endpoint.
    if base_url.endswith("/api") and not base_url.endswith("/api/v1"):
        return f"{base_url}/v1"

    return base_url


def resolve_clem_model_for_claude_code(clem_model: str) -> dict[str, Any]:
    model_spec = _find_model_spec(clem_model)
    backend = model_spec.get("backend")

    if backend != "openrouter":
        raise NotImplementedError(
            "MVP limitation: Claude Code registry-model support currently only "
            f"supports clembench models with backend='openrouter'. "
            f"Model {clem_model!r} has backend={backend!r}."
        )

    model_id = model_spec.get("model_id") or model_spec.get("model_name")
    key_config = _openrouter_key_config()

    return {
        "harness": "claude_code",
        "clem_model": model_spec["model_name"],
        "backend": "openrouter",
        "model": model_id,
        "display_model": model_spec["model_name"],
        "env": {
            "ANTHROPIC_BASE_URL": _openrouter_anthropic_base_url(key_config),
            "ANTHROPIC_AUTH_TOKEN": _openrouter_api_key(key_config),
            "ANTHROPIC_API_KEY": "",
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        },
    }


def resolve_clem_model_for_codex(clem_model: str) -> dict[str, Any]:
    model_spec = _find_model_spec(clem_model)
    backend = model_spec.get("backend")

    model_id = model_spec.get("model_id") or model_spec.get("model_name")

    if backend == "openrouter":
        key_config = _openrouter_key_config()

        return {
            "harness": "codex",
            "clem_model": model_spec["model_name"],
            "backend": "openrouter",
            "model": model_id,
            "display_model": model_spec["model_name"],
            "base_url": _openrouter_openai_base_url(key_config),
            "env": {
                "OPENROUTER_API_KEY": _openrouter_api_key(key_config),
                "OPENAI_API_KEY": _openrouter_api_key(key_config),
            },
        }

    if backend == "openai":
        key_config = _openai_key_config()

        return {
            "harness": "codex",
            "clem_model": model_spec["model_name"],
            "backend": "openai",
            "model": model_id,
            "display_model": model_spec["model_name"],
            "base_url": "https://api.openai.com/v1",
            "env": {
                "OPENAI_API_KEY": _openai_api_key(key_config),
            },
        }

    raise NotImplementedError(
        "MVP limitation: Codex registry-model support currently only "
        f"supports clembench models with backend='openrouter' or backend='openai'. "
        f"Model {clem_model!r} has backend={backend!r}."
    )



def resolve_clem_model_for_hermes(clem_model: str) -> dict[str, Any]:
    model_spec = _find_model_spec(clem_model)
    backend = model_spec.get("backend")
    model_id = model_spec.get("model_id") or model_spec.get("model_name")

    if backend == "openrouter":
        key_config = _openrouter_key_config()

        return {
            "harness": "hermes",
            "clem_model": model_spec["model_name"],
            "backend": "openrouter",
            "model": model_id,
            "display_model": model_spec["model_name"],
            "base_url": _openrouter_openai_base_url(key_config),
            "env": {
                "OPENROUTER_API_KEY": _openrouter_api_key(key_config),
            },
        }

    raise NotImplementedError(
        "MVP limitation: Hermes registry-model support currently only "
        f"supports clembench models with backend='openrouter'. "
        f"Model {clem_model!r} has backend={backend!r}."
    )



def resolve_clem_model_for_openclaw(clem_model: str) -> dict[str, Any]:
    model_spec = _find_model_spec(clem_model)
    backend = model_spec.get("backend")
    model_id = model_spec.get("model_id") or model_spec.get("model_name")

    if backend == "openrouter":
        key_config = _openrouter_key_config()

        return {
            "harness": "openclaw",
            "clem_model": model_spec["model_name"],
            "backend": "openrouter",
            "model": f"openrouter/{model_id}",
            "display_model": model_spec["model_name"],
            "env": {
                "OPENROUTER_API_KEY": _openrouter_api_key(key_config),
            },
        }




    if backend == "openai_compatible":
        key_config = _openai_compatible_key_config()
        provider_name = "clem_openai_compatible"
        api_key_env = "CLEM_OPENAI_COMPATIBLE_API_KEY"
        runtime_model = f"{provider_name}/{model_id}"

        model_definition: dict[str, Any] = {
            "id": model_id,
            "name": model_spec["model_name"],
            "reasoning": _model_reasoning_enabled(model_spec),
            "input": ["text"],
        }

        model_config = model_spec.get("model_config") or {}
        extra_body = {}

        if isinstance(model_config, dict):
            extra_body = model_config.get("extra_body") or {}

        chat_template_kwargs = {}

        if isinstance(extra_body, dict):
            raw_chat_template_kwargs = extra_body.get("chat_template_kwargs") or {}

            if isinstance(raw_chat_template_kwargs, dict):
                chat_template_kwargs = raw_chat_template_kwargs

        if "enable_thinking" in chat_template_kwargs:
            # The clembench registry remains the source of truth:
            # chat_template_kwargs.enable_thinking controls how Qwen is run.
            # OpenClaw only emits the Qwen chat-template control field when
            # model.reasoning is true, so this activates that compatibility path.
            model_definition["reasoning"] = True
            model_definition["compat"] = {
                "thinkingFormat": "qwen-chat-template",
            }

        openclaw_model_config: dict[str, Any] = {
            "alias": model_spec["model_name"],
        }

        if chat_template_kwargs:
            openclaw_model_config["params"] = {
                "chat_template_kwargs": chat_template_kwargs,
            }

        context_window = _model_context_window(model_spec)

        if context_window is not None:
            model_definition["contextWindow"] = context_window

        model_definition["maxTokens"] = 8192

        return {
            "harness": "openclaw",
            "clem_model": model_spec["model_name"],
            "backend": "openai_compatible",
            "model": runtime_model,
            "display_model": model_spec["model_name"],
            "env": {
                api_key_env: _openai_compatible_api_key(key_config),
            },
            "openclaw_config_patch": {
                "models": {
                    "mode": "merge",
                    "providers": {
                        provider_name: {
                            "baseUrl": _openai_compatible_base_url(key_config),
                            "apiKey": f"${{{api_key_env}}}",
                            "api": "openai-completions",
                            "models": [
                                model_definition,
                            ],
                        },
                    },
                },
                "agents": {
                    "defaults": {
                        "models": {
                            runtime_model: openclaw_model_config,
                        },
                    },
                },
            },
        }

    raise NotImplementedError(
        "MVP limitation: OpenClaw registry-model support currently only "
        f"supports clembench models with backend='openrouter' or backend='openai_compatible'. "
        f"Model {clem_model!r} has backend={backend!r}."
    )


def resolve_agent_model_connection(agent_name: str,
                                   registry_path: str | Path) -> dict[str, Any] | None:
    registry_path = Path(registry_path)
    registry = _read_json(registry_path)

    if isinstance(registry, dict):
        registry = [registry]

    for entry in registry:
        if entry.get("agent_name") != agent_name:
            continue

        backend = entry.get("backend")
        agent_config = entry.get("agent_config", {})
        clem_model = agent_config.get("clem_model")

        if not clem_model:
            return None

        if backend == "claude_code":
            return resolve_clem_model_for_claude_code(clem_model)

        if backend == "codex":
            return resolve_clem_model_for_codex(clem_model)

        if backend == "hermes":
            return resolve_clem_model_for_hermes(clem_model)

        if backend == "openclaw":
            return resolve_clem_model_for_openclaw(clem_model)

        raise NotImplementedError(
            f"No model resolver implemented yet for backend={backend!r}."
        )

    raise KeyError(f"Could not find agent {agent_name!r} in {registry_path}.")
