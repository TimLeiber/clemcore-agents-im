from fastmcp import FastMCP

from openenv.core.env_server.mcp_environment import MCPEnvironment
from openenv.core.env_server.mcp_types import Action, Observation

from clemcore.clemgame.envs.openenv.models import ClemGameAction, ClemGameObservation, ClemGameState
from clemcore.clemgame.envs.openenv.server.environment import ClemGameEnvironment


class ClemGameMCPEnvironment(MCPEnvironment):
    """
    Experimental MCP-facing wrapper around ClemGameEnvironment.

    This exposes a clembench game as MCP tools while preserving the OpenEnv
    environment lifecycle.
    """

    def __init__(self, clem_env: ClemGameEnvironment):
        self.clem_env = clem_env

        mcp = FastMCP("clem-game")

        super().__init__(mcp)

        @self.tool()
        def submit_response(response: str) -> dict:
            """
            submit a response/move to the current clembench game.
            """
            obs = self.clem_env.step(ClemGameAction(response=response))
            return {"context": obs.context,
                "reward": obs.reward,
                "done": obs.done,
                "metadata": obs.metadata,
            }

        @self.tool()
        def get_state() -> dict:
            """ get the current clembench game state."""
            state = self.clem_env.state
            return state.model_dump()

        @self.tool()
        def start_game(game_id: int | None = None,
                       experiment_name: str | None = None) -> dict:
            """start/reset the current clembench game and return the initial observation."""

            reset_kwargs = {}

            if game_id is not None:
                reset_kwargs["game_id"] = game_id

            if experiment_name is not None:
                reset_kwargs["experiment_name"] = experiment_name

            obs = self.clem_env.reset(**reset_kwargs)
            return {"context": obs.context,
                "reward": obs.reward,
                "done": obs.done,
                "metadata": obs.metadata}

    def reset(self, seed=None, episode_id=None, **kwargs) -> ClemGameObservation:
        return self.clem_env.reset(seed=seed, episode_id=episode_id, **kwargs)

    def _step_impl(self,
                   action: Action,
                   timeout_s: float | None = None,
                   **kwargs) -> Observation:
        if isinstance(action, ClemGameAction):
            return self.clem_env.step(action, timeout_s=timeout_s, **kwargs)

        raise TypeError(f"Unsupported action type: {type(action)}")

    @property
    def state(self) -> ClemGameState:
        return self.clem_env.state

    def close(self) -> None:
        self.clem_env.close()
        super().close()