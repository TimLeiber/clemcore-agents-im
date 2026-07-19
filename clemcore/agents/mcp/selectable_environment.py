import logging
from typing import Any, Callable, Dict

from datasets import load_dataset
from gymnasium import Env
from pettingzoo.utils.wrappers import OrderEnforcingWrapper

from clemcore.backends import load_models
from clemcore.clemgame.callbacks.base import GameBenchmarkCallbackList
from clemcore.clemgame.envs.openenv.models import (
    ClemGameAction,
    ClemGameObservation,
    ClemGameState,
)
from clemcore.clemgame.envs.pettingzoo import check_agent_mapping_for_training
from clemcore.clemgame.envs.pettingzoo.master import GameMasterEnv
from clemcore.clemgame.envs.pettingzoo.wrappers import (
    AECToGymWrapper,
    GameBenchmarkWrapper,
    GameInstanceIteratorWrapper,
    SinglePlayerWrapper,
)
from clemcore.clemgame.instances import GameInstances, to_instance_filter
from clemcore.clemgame.master import GameState
from clemcore.clemgame.registry import GameRegistry


module_logger = logging.getLogger(__name__)


class SelectableClemGameEnvironment:
    """
    Agent-specific OpenEnv environment for selecting exact clembench rows.

    This mirrors regular clembench's outer loop behavior without changing
    clemcore.clemgame internals. The pipeline can select an instance by passing
    experiment_name and game_id into reset/start_game.
    """

    def __init__(self,
                 game_name: str,
                 *,
                 instances_filename: str | None = None,
                 game_instance_split: str | None = None,
                 single_pass: bool = False,
                 learner_agent: str = "player_0",
                 env_agents: Dict[str, str] | None = None,
                 gen_args: Dict[str, Any] | None = None,
                 callbacks: GameBenchmarkCallbackList | None = None,
                 reward_func: Callable[[dict, str, GameState, dict], float] | None = None,
                 feedback_func: Callable[[dict, str, GameState, dict], str | None] | None = None):
        module_logger.info(
            "Initialize SelectableClemGameEnvironment: "
            "game_name=%s, instances_filename=%s, game_instance_split=%s, learner_agent=%s, env_agents=%s",
            game_name,
            instances_filename,
            game_instance_split,
            learner_agent,
            env_agents,
        )

        self._game_name = game_name
        self._instances_filename = instances_filename
        self._game_instance_split = game_instance_split
        self._single_pass = single_pass
        self._learner_agent = learner_agent
        self._callbacks = callbacks
        self._reward_func = reward_func
        self._feedback_func = feedback_func
        self._game_env: Env | None = None
        self._state = ClemGameState(
            game_name=game_name,
            episode_id="episode_0",
            step_count=0,
            episode_count=0,
        )

        env_agents = env_agents or {}

        game_registry = GameRegistry.from_directories_and_cwd_files()
        self._game_spec = game_registry.get_game_spec(game_name)

        if instances_filename:
            self._game_spec.instances = instances_filename

        check_agent_mapping_for_training(
            self._game_spec,
            {learner_agent: "learner", **env_agents},
        )

        if self._game_spec.is_multi_player():
            agent_models = load_models(list(env_agents.values()), gen_args)
            self._env_agents = {
                agent_id: agent_model
                for agent_id, agent_model in zip(env_agents.keys(), agent_models)
            }
        else:
            self._env_agents = env_agents

    def _base_instances_filter(self) -> Callable[[dict], bool] | None:
        if not self._game_instance_split:
            return None

        dataset = load_dataset(
            "colab-potsdam/playpen-data",
            "instances",
            split=self._game_instance_split,
        )
        return to_instance_filter(dataset)

    def _selected_instances_filter(self,
                                   experiment_name: str | None,
                                   game_id: int | str | None) -> Callable[[dict], bool] | None:
        base_filter = self._base_instances_filter()

        if experiment_name is None and game_id is None:
            return base_filter

        if experiment_name is None or game_id is None:
            raise ValueError(
                "Both experiment_name and game_id must be provided for exact instance selection."
            )

        selected_game_id = int(game_id)

        def filter_fn(row: dict) -> bool:
            if base_filter is not None and not base_filter(row):
                return False

            return (
                row["experiment"]["name"] == experiment_name
                and int(row["game_instance"]["game_id"]) == selected_game_id
            )

        return filter_fn

    def _create_game_env(self,
                         instances_filter: Callable[[dict], bool] | None) -> Env:
        game_env = GameBenchmarkWrapper(
            GameMasterEnv,
            game_spec=self._game_spec,
            callbacks=self._callbacks,
            reward_func=self._reward_func,
            feedback_func=self._feedback_func,
        )
        game_env = OrderEnforcingWrapper(game_env)

        game_instances = GameInstances.from_game_spec(self._game_spec)
        game_instances = game_instances.filter(instances_filter)

        if len(game_instances) == 0:
            raise ValueError("No game instances matched the requested selection.")

        game_env = GameInstanceIteratorWrapper(
            game_env,
            game_instances,
            single_pass=self._single_pass,
        )
        game_env = SinglePlayerWrapper(
            game_env,
            self._learner_agent,
            env_agents=self._env_agents,
        )
        game_env = AECToGymWrapper(game_env)

        return game_env

    def reset(self,
              seed=None,
              episode_id=None,
              experiment_name: str | None = None,
              game_id: int | str | None = None,
              **kwargs) -> ClemGameObservation:
        if episode_id is not None:
            kwargs["episode_id"] = episode_id

        if self._game_env is not None:
            self._game_env.close()

        instances_filter = self._selected_instances_filter(
            experiment_name=experiment_name,
            game_id=game_id,
        )
        self._game_env = self._create_game_env(instances_filter)

        module_logger.info(
            "Reset SelectableClemGameEnvironment '%s' for experiment=%s, game_id=%s",
            self._state.game_name,
            experiment_name,
            game_id,
        )

        options = kwargs if kwargs else None
        observation, info = self._game_env.reset(seed=seed, options=options)

        self._state.step_count = 0
        self._state.episode_count += 1
        self._state.episode_id = f"episode_{self._state.episode_count}"

        return ClemGameObservation(context=observation)

    def step(self,
             action: ClemGameAction,
             timeout_s=None,
             **kwargs) -> ClemGameObservation:
        if self._game_env is None:
            raise RuntimeError("Environment has not been reset. Call start_game first.")

        observation, reward, done, truncated, info = self._game_env.step(action.response)
        self._state.step_count += 1

        return ClemGameObservation(
            context=observation,
            reward=float(reward),
            done=done,
            metadata={**info, **dict(truncated=truncated)},
        )

    @property
    def state(self) -> ClemGameState:
        return self._state

    def close(self) -> None:
        module_logger.info("Close SelectableClemGameEnvironment %s", self._state.game_name)

        if self._game_env is not None:
            self._game_env.close()
            self._game_env = None
