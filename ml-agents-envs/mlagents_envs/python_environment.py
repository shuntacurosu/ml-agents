import atexit

import numpy as np
from typing import Dict, List, Optional, Tuple, Mapping as MappingType

from py_exec_env import ACTION_TYPE, CONTINUOUS_SIZE, DISCRETE_BRANCHES, ENV_NAME, OBS_SIZE, ActionType, AgentInfo, PyExecEnv

from mlagents_envs.logging_util import get_logger
from mlagents_envs.side_channel.side_channel import SideChannel

from mlagents_envs.base_env import (
    ActionSpec,
    BaseEnv,
    DecisionSteps,
    DimensionProperty,
    ObservationSpec,
    ObservationType,
    TerminalSteps,
    BehaviorSpec,
    ActionTuple,
    BehaviorName,
    AgentId,
    BehaviorMapping,
)
from mlagents_envs.timers import timed
from mlagents_envs.exception import (
    UnityEnvironmentException,
    UnityActionException,
    UnityTimeOutException,
    UnityCommunicatorStoppedException,
)

logger = get_logger(__name__)

BRAIN_NAME = f"{ENV_NAME}?team=0"

class PythonEnvironment(BaseEnv):

    def __init__(
        self,
        file_name: Optional[str] = None,
        worker_id: int = 0,
        base_port: Optional[int] = None,
        seed: int = 0,
        no_graphics: bool = False,
        no_graphics_monitor: bool = False,
        timeout_wait: int = 60,
        additional_args: Optional[List[str]] = None,
        side_channels: Optional[List[SideChannel]] = None,
        log_folder: Optional[str] = None,
        num_areas: int = 1,
    ):
        """
        file_name: Unity環境バイナリの名前。
        base_port: Unity環境に接続するためのベースラインポート番号。worker_idはこれを増やして使用します。環境が指定されていない場合（つまり、file_nameがNoneの場合）、DEFAULT_EDITOR_PORTが使用されます。
        worker_id: base_portからのオフセット。複数の環境を同時にトレーニングするために使用されます。
        no_graphics: Unityシミュレータをグラフィックスなしモードで実行するかどうか。
        no_graphics_monitor: メインワーカーをグラフィックスモードで実行し、残りのワーカーをグラフィックスなしモードで実行するかどうか。
        timeout_wait: 環境からの接続を待つ時間（秒単位）。
        args: 追加のUnityコマンドライン引数。
        side_channels: Unityとのno-rl通信用の追加のサイドチャネル。
        log_folder: Unity Playerログファイルを書き込むためのオプションのフォルダ。絶対パスが必要です。
        """
        atexit.register(self._close)

        self._timeout_wait: int = timeout_wait
        self._worker_id = worker_id

        self._env = PyExecEnv()

        self._loaded = True
        self._env_state: Dict[str, Tuple[DecisionSteps, TerminalSteps]] = {}    # reset(), step()で返すstate(obs, reward)
        self._env_specs: Dict[str, BehaviorSpec] = {}                           # 環境の情報
        self._env_actions: Dict[str, ActionTuple] = {}                          # 内部で使用するaction
        self._is_first_message = True                                           # 用途は不明だが実装は単純なので実装する
        self._update_behavior_specs()                                           # _env_specsの更新

    def reset(self) -> None:
        if self._loaded:
            self._update_behavior_specs()
            rl_output = self._env.reset()
            self._update_state(rl_output)
            self._is_first_message = False
            self._env_actions.clear()
        else:

            raise UnityEnvironmentException("No Unity environment is loaded.")

    @timed
    def step(self) -> None:
        if self._is_first_message:
            return self.reset()
        if not self._loaded:
            raise UnityEnvironmentException("No Unity environment is loaded.")
        # fill the blanks for missing actions
        for group_name in self._env_specs:
            if group_name not in self._env_actions:
                n_agents = 0
                if group_name in self._env_state:
                    n_agents = len(self._env_state[group_name][0])
                self._env_actions[group_name] = self._env_specs[
                    group_name
                ].action_spec.empty_action(n_agents)
        
        if ACTION_TYPE == ActionType.CONTINUE:
            rl_output = self._env.step(self._env_actions[BRAIN_NAME].continuous)
        elif ACTION_TYPE == ActionType.DISCRETE:
            rl_output = self._env.step(self._env_actions[BRAIN_NAME].discrete)
        if rl_output is None:
            raise UnityCommunicatorStoppedException("Communicator has exited.")
        self._update_behavior_specs()
        self._update_state(rl_output)
        self._env_actions.clear()


    def _update_behavior_specs(self) -> None:
        """
        _env_specsを更新する。
        __init__, reset, stepから呼ばれる
        """
        if self._is_first_message == True:
            new_spec = self._behavior_spec_from_proto()
            self._env_specs[BRAIN_NAME] = new_spec
            logger.info(f"Connected new brain: {BRAIN_NAME}")


    def _behavior_spec_from_proto(self):
        observation_specs = []
        observation_specs.append(
            ObservationSpec(
                name=f"VectorSensor_size{OBS_SIZE}",
                shape=(OBS_SIZE,),
                observation_type=ObservationType(0),
                dimension_property=tuple([DimensionProperty(1)])
            )
        )

        # proto from communicator < v1.3 does not set action spec, use deprecated fields instead
        action_spec = ActionSpec(
            continuous_size = CONTINUOUS_SIZE,
            discrete_branches = DISCRETE_BRANCHES
        )
        
        return BehaviorSpec(observation_specs, action_spec)

    def _update_state(self, agent_info_list: List[AgentInfo]) -> None:
        """
        Collects experience information from all external brains in environment at current step.
        """
        self._env_state[BRAIN_NAME] = self._steps_from_proto(agent_info_list)

    @timed
    def _steps_from_proto(self, agent_info_list: List[AgentInfo]) -> Tuple[DecisionSteps, TerminalSteps]:
        decision_agent_info_list = [
            agent_info for agent_info in agent_info_list if not agent_info.done
        ]
        terminal_agent_info_list = [
            agent_info for agent_info in agent_info_list if agent_info.done
        ]

        # obs
        decision_obs_list = [np.array(
            [agent_info.obs for agent_info in decision_agent_info_list], dtype=np.float32
        )]
        terminal_obs_list = [np.array(
            [agent_info.obs for agent_info in terminal_agent_info_list], dtype=np.float32
        )]

        # reward
        decision_rewards = np.array(
            [agent_info.reward for agent_info in decision_agent_info_list], dtype=np.float32
        )
        terminal_rewards = np.array(
            [agent_info.reward for agent_info in terminal_agent_info_list], dtype=np.float32
        )

        # group_reward
        decision_group_rewards = np.array(
            [agent_info.group_reward for agent_info in decision_agent_info_list],
            dtype=np.float32,
        )
        terminal_group_rewards = np.array(
            [agent_info.group_reward for agent_info in terminal_agent_info_list],
            dtype=np.float32,
        )

        # group_id
        decision_group_id = [agent_info.group_id for agent_info in decision_agent_info_list]
        terminal_group_id = [agent_info.group_id for agent_info in terminal_agent_info_list]

        # max_step_reached
        max_step = np.array(
            [agent_info.max_step_reached for agent_info in terminal_agent_info_list],
            dtype=bool,
        )
        decision_agent_id = np.array(
            [agent_info.id for agent_info in decision_agent_info_list], dtype=np.int32
        )
        terminal_agent_id = np.array(
            [agent_info.id for agent_info in terminal_agent_info_list], dtype=np.int32
        )

        action_mask = None

        return (
            DecisionSteps(
                decision_obs_list,
                decision_rewards,
                decision_agent_id,
                action_mask,
                decision_group_id,
                decision_group_rewards,
            ),
            TerminalSteps(
                terminal_obs_list,
                terminal_rewards,
                max_step,
                terminal_agent_id,
                terminal_group_id,
                terminal_group_rewards,
            ),
        )

    @property
    def behavior_specs(self) -> MappingType[str, BehaviorSpec]:
        return BehaviorMapping(self._env_specs)

    def _assert_behavior_exists(self, behavior_name: str) -> None:
        if behavior_name not in self._env_specs:
            raise UnityActionException(
                f"The group {behavior_name} does not correspond to an existing "
                f"agent group in the environment"
            )

    def set_actions(self, behavior_name: BehaviorName, action: ActionTuple) -> None:
        self._assert_behavior_exists(behavior_name)
        if behavior_name not in self._env_state:
            return
        action_spec = self._env_specs[behavior_name].action_spec
        num_agents = len(self._env_state[behavior_name][0])
        action = action_spec._validate_action(action, num_agents, behavior_name)
        self._env_actions[behavior_name] = action

    def set_action_for_agent(
        self, behavior_name: BehaviorName, agent_id: AgentId, action: ActionTuple
    ) -> None:
        self._assert_behavior_exists(behavior_name)
        if behavior_name not in self._env_state:
            return
        action_spec = self._env_specs[behavior_name].action_spec
        action = action_spec._validate_action(action, 1, behavior_name)
        if behavior_name not in self._env_actions:
            num_agents = len(self._env_state[behavior_name][0])
            self._env_actions[behavior_name] = action_spec.empty_action(num_agents)
        try:
            index = np.where(self._env_state[behavior_name][0].agent_id == agent_id)[0][
                0
            ]
        except IndexError as ie:
            raise IndexError(
                "agent_id {} is did not request a decision at the previous step".format(
                    agent_id
                )
            ) from ie
        if action_spec.continuous_size > 0:
            self._env_actions[behavior_name].continuous[index] = action.continuous[0, :]
        if action_spec.discrete_size > 0:
            self._env_actions[behavior_name].discrete[index] = action.discrete[0, :]

    def get_steps(
        self, behavior_name: BehaviorName
    ) -> Tuple[DecisionSteps, TerminalSteps]:
        self._assert_behavior_exists(behavior_name)
        return self._env_state[behavior_name]

    def close(self):
        """
        Sends a shutdown signal to the unity environment, and closes the socket connection.
        """
        if self._loaded:
            self._close()
        else:
            raise UnityEnvironmentException("No Unity environment is loaded.")

    def _close(self, timeout: Optional[int] = None) -> None:
        """
        Close the communicator and environment subprocess (if necessary).

        :int timeout: [Optional] Number of seconds to wait for the environment to shut down before
            force-killing it.  Defaults to `self.timeout_wait`.
        """
        self._loaded = False
        if self._env is not None:
            self._env.close()
            self._env = None
