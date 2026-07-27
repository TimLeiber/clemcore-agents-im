from pathlib import Path
from typing import Optional, Dict, Any

from openenv.core import create_app

from clemcore.clemgame import episode_results_folder_callbacks
from clemcore.clemgame.callbacks.base import GameBenchmarkCallbackList
from clemcore.clemgame.envs.openenv.models import ClemGameAction, ClemGameObservation
from clemcore.agents.mcp.environment import ClemGameMCPEnvironment, SelectableClemGameEnvironment


def create_clem_mcp_app(game_name: str,
                        *,
                        learner_agent: str = "player_0",
                        env_agents: Optional[Dict[str, str]] = None,
                        game_instance_split: str | None = None,
                        instances_filename: str | None = None,
                        single_pass: bool = False,
                        gen_args: Optional[Dict[str, Any]] = None,
                        callbacks: GameBenchmarkCallbackList | None = None,
                        results_dir: str | Path | None = None,
                        run_dir: str | None = None,
                        player_model_infos: Any = None):

    if callbacks is None and results_dir is not None and run_dir is not None:
        callbacks = episode_results_folder_callbacks(
            run_dir=run_dir,
            result_dir_path=results_dir,
            player_model_infos=player_model_infos,
        )

    def make_env():
        base_env = SelectableClemGameEnvironment(
            game_name,
            instances_filename=instances_filename,
            game_instance_split=game_instance_split,
            single_pass=single_pass,
            learner_agent=learner_agent,
            env_agents=env_agents,
            gen_args=gen_args,
            callbacks=callbacks,
        )

        wrapped_env = ClemGameMCPEnvironment(base_env)

        return wrapped_env

    return create_app(
        make_env,
        ClemGameAction,
        ClemGameObservation,
        env_name="clem_mcp_env",
    )
