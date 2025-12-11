# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Bring a box to a target and orientation."""

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx
from mujoco.mjx._src import math
from mujoco_playground._src import mjx_env
from mujoco_playground._src.manipulation.ur3 import ur3
from mujoco_playground._src.mjx_env import State  # pylint: disable=g-importing-member
import numpy as np


def default_config() -> config_dict.ConfigDict:
  """Returns the default config for bring_to_target tasks."""
  config = config_dict.create(
      ctrl_dt=0.02,
      sim_dt=0.005,
      episode_length=300,
      action_repeat=1,
      action_scale=0.04,

    # ---- CONFIGS PARA REWARD ESPARSO
    #   reward_config=config_dict.create(
    #       scales=config_dict.create(
    #           # --- ZERAR TUDO QUE É SHAPING ---
    #           gripper_box=0.0,
    #           box_target=0.0,
    #           robot_target_qpos=0.0,
    #           is_lifted=0.0,
    #           hoist_reward=0.0,
    #           is_grasped=0.0,
    #           approach_vel=0.0,
    #           joint_vel=0.0,
    #           ctrl_norm=0.0,
    #           success_bonus=0.0, 
              
    #           # --- ATIVAR RECOMPENSA ESPARSA ---
    #           # Damos um valor alto para que, quando ele acertar por sorte,
    #           # o gradiente seja forte o suficiente para ele "lembrar".
    #           sparse_success=10.0,
              
    #           # Opcional: penalidade leve de colisão
    #           no_floor_collision=0.1, 
    #       )
    #   ),

    # ----- CONFIGS PARA REWARD DENSO
      reward_config=config_dict.create(
          scales=config_dict.create(
              # Aproximação
              # Soma de reach+precision chega a ~2.0, então 3.0 * 2.0 = 6.0 total
              gripper_box=3.0, 
              
              # Objetivo Principal
              # prêmio por levar a caixa ao alvo
              box_target=25.0,
              
              # Altura e Levantamento
              # Incentiva manter o cubo alto e na altura do alvo
              hoist_reward=5.0, 
              # Peso menor para o binário, apenas indicativo
              is_lifted=2.0,    
              
              # Regularização e Segurança
              # Baixo para permitir movimento do braço (não travar)
              robot_target_qpos=0.1, 
              # Evita colisão bruta com chão
              no_floor_collision=0.25,
              
              # Penalidades Globais (Suaves)
              # Evita movimentos erráticos e força excessiva
              joint_vel=-0.02, 
              ctrl_norm=-0.01,
              
              # Bônus Final
              # Recompensa extra se terminar no alvo e estável
              success_bonus=10.0, 
          )
      ),
      impl='jax',
      nconmax=12 * 8192, #24 * 8192,
      njmax=128,
  )
  return config


class Ur3PickCube(ur3.Ur3Base):
  """Bring a box to a target."""

  def __init__(
      self,
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
      sample_orientation: bool = False,
  ):
    xml_path = (
        mjx_env.ROOT_PATH
        / "manipulation"
        / "ur3"
        / "xmls"
        / "ur3_cube_task.xml"
    )
    super().__init__(
        xml_path,
        config,
        config_overrides,
    )
    self._post_init(obj_name="object_box", keyframe="home")
    self._sample_orientation = sample_orientation

    # Contact sensor IDs.
    # self._floor_hand_found_sensor = [
    #     self._mj_model.sensor(f"{geom}_floor_found").id
    #     for geom in ["left_finger_pad", "right_finger_pad", "hand_capsule"]
    # ]

    self._floor_hand_found_sensor = []

  def reset(self, rng: jax.Array) -> State:
    rng, rng_box, rng_target = jax.random.split(rng, 3)

    # ========================================================
    # 1. POSIÇÃO DA CAIXA (Zona Frontal)
    # ========================================================
    # X: Centralizado na largura do robô.
    # Y: Na frente (negativo), mas não tão longe que precise esticar demais.
    
    # X entre -0.15 e 0.15 (Faixa estreita de 30cm)
    box_x = jax.random.uniform(rng_box, (1,), minval=-0.15, maxval=0.15)
    
    rng_box, rng_box_y = jax.random.split(rng_box)
    # Y entre -0.45 e -0.25 (Distância confortável de alcance)
    box_y = jax.random.uniform(rng_box_y, (1,), minval=-0.5, maxval=-0.3)
    
    box_pos = jp.concatenate([box_x, box_y, jp.array([0.0]) + self._init_obj_pos[2:3]])

    # ========================================================
    # 2. POSIÇÃO DO ALVO (Lateral Esquerda - Para onde ele leva)
    # ========================================================
    # Mantemos o alvo separado para forçar o transporte.
    
    rng_target, rng_t_x, rng_t_y, rng_t_z = jax.random.split(rng_target, 4)

    # X positivo (Esquerda do robô)
    target_x = jax.random.uniform(rng_t_x, (1,), minval=0.25, maxval=0.55)
    # Y similar ao da caixa (na frente/lado)
    target_y = jax.random.uniform(rng_t_y, (1,), minval=-0.5, maxval=-0.3)
    target_z = jax.random.uniform(rng_t_z, (1,), minval=0.15, maxval=0.35)

    target_pos = jp.concatenate([target_x, target_y, target_z])

    target_quat = jp.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    if self._sample_orientation:
      # sample a random direction
      rng, rng_axis, rng_theta = jax.random.split(rng, 3)
      perturb_axis = jax.random.uniform(rng_axis, (3,), minval=-1, maxval=1)
      perturb_axis = perturb_axis / math.norm(perturb_axis)
      perturb_theta = jax.random.uniform(rng_theta, maxval=np.deg2rad(45))
      target_quat = math.axis_angle_to_quat(perturb_axis, perturb_theta)

    # initialize data
    init_q = (
        jp.array(self._init_q)
        .at[self._obj_qposadr : self._obj_qposadr + 3]
        .set(box_pos)
    )
    data = mjx_env.make_data(
        self._mj_model,
        qpos=init_q,
        qvel=jp.zeros(self._mjx_model.nv, dtype=float),
        ctrl=self._init_ctrl,
        impl=self._mjx_model.impl.value,
        nconmax=self._config.nconmax,
        njmax=self._config.njmax,
    )

    # set target mocap position
    data = data.replace(
        mocap_pos=data.mocap_pos.at[self._mocap_target, :].set(target_pos),
        mocap_quat=data.mocap_quat.at[self._mocap_target, :].set(target_quat),
    )

    # initialize env state and info
    metrics = {
        "out_of_bounds": jp.array(0.0, dtype=float),
        **{k: 0.0 for k in self._config.reward_config.scales.keys()},
    }
    info = {"rng": rng, "target_pos": target_pos, "reached_box": 0.0}
    obs = self._get_obs(data, info)
    reward, done = jp.zeros(2)
    state = State(data, obs, reward, done, metrics, info)
    return state

  def step(self, state: State, action: jax.Array) -> State:
    delta = action * self._action_scale
    ctrl = state.data.ctrl + delta
    ctrl = jp.clip(ctrl, self._lowers, self._uppers)

    data = mjx_env.step(self._mjx_model, state.data, ctrl, self.n_substeps)

    raw_rewards = self._get_reward(data, state.info)
    rewards = {
        k: v * self._config.reward_config.scales[k]
        for k, v in raw_rewards.items()
    }
    reward = jp.clip(sum(rewards.values()), -1e4, 1e4)
    box_pos = data.xpos[self._obj_body]
    out_of_bounds = jp.any(jp.abs(box_pos) > 1.0)
    out_of_bounds |= box_pos[2] < 0.0
    done = out_of_bounds | jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()
    done = done.astype(float)

    state.metrics.update(
        **raw_rewards, out_of_bounds=out_of_bounds.astype(float)
    )

    obs = self._get_obs(data, state.info)
    state = State(data, obs, reward, done, state.metrics, state.info)

    return state

#--- FUNÇÃO DE REWARD DENSA
  def _get_reward(self, data: mjx.Data, info: Dict[str, Any]) -> Dict[str, Any]:
        target_pos = info["target_pos"]
        box_pos = data.xpos[self._obj_body]
        gripper_pos = data.site_xpos[self._gripper_site]

        # --- Distâncias ---
        dist_to_box = jp.linalg.norm(box_pos - gripper_pos)
        pos_err = jp.linalg.norm(target_pos - box_pos)

        # Recompensa de Aproximação (Reach + Precision)
        # Reach (grosso): atrai de longe
        reach = 1 - jp.tanh(4.0 * dist_to_box)
        # Precision (fino): exige centralização perfeita (< 3cm)
        precision = (1 - jp.tanh(20.0 * dist_to_box)) * (dist_to_box < 0.1)
        gripper_box = reach + precision

        # Lógica de "Segurando"
        # Estamos segurando SE: (distancia < 3.5cm) E (dedos fazendo força para fechar)
        # RG2 fecha com valor positivo.
        is_holding = (dist_to_box < 0.035) * (data.ctrl[-1] > 0.1)
        is_holding = is_holding.astype(float)

        # Recompensa do Alvo (Box Target)
        # Se estiver segurando, a recompensa do alvo ativa com força total.
        # Se não estiver, damos 10% dela só para ele saber a direção.
        box_target_val = 1 - jp.tanh(5.0 * pos_err)
        box_target_reward = box_target_val * (0.1 + 0.9 * is_holding)

        # Lógica de Altura (Hoist)
        box_z = data.xpos[self._obj_body][2]
        # Recompensa suave por altura (não binária), encoraja subir até 20cm
        hoist_reward = 1 - jp.tanh(3.0 * jp.abs(box_z - 0.2))
        
        # Recalcula binários para estatísticas/log
        is_lifted = (box_z > 0.04).astype(float)

        # Postura (Robot Target QPos) - Encoraja posições próximas da inicial
        robot_target_qpos = 1 - jp.tanh(
            jp.linalg.norm(
                data.qpos[self._robot_arm_qposadr]
                - self._init_q[self._robot_arm_qposadr]
            )
        )

        # Colisões
        if self._floor_hand_found_sensor:
            hand_floor_collision = [
                data.sensordata[self._mj_model.sensor_adr[sensor_id]] > 0
                for sensor_id in self._floor_hand_found_sensor
            ]
            no_floor_collision = (1 - (sum(hand_floor_collision) > 0)).astype(float)
        else:
            no_floor_collision = jp.array(1.0, dtype=float)

        # Regularização
        joint_vel = jp.linalg.norm(data.qvel)
        ctrl_norm = jp.linalg.norm(data.ctrl)

        # Bônus Final: Perto do alvo E Estável
        is_stable_at_target = (pos_err < 0.05) * (joint_vel < 0.5)

        rewards = {
            "gripper_box": gripper_box,
            "box_target": box_target_reward,
            "no_floor_collision": no_floor_collision,
            "robot_target_qpos": robot_target_qpos,
            
            "is_lifted": is_lifted, 
            "hoist_reward": hoist_reward * is_holding, # Só ganha se estiver segurando
            
            "joint_vel": joint_vel,
            "ctrl_norm": ctrl_norm,
            "success_bonus": is_stable_at_target.astype(float) * 10.0,
        }
        return rewards

# --- FUNÇÃO DE REWARD ESPARSA
#   def _get_reward(self, data: mjx.Data, info: Dict[str, Any]) -> Dict[str, Any]:
#     target_pos = info["target_pos"]
#     box_pos = data.xpos[self._obj_body]
    
#     # Calcula apenas a distância final
#     dist_to_target = jp.linalg.norm(target_pos - box_pos)
    
#     # ----------------------------------------------------
#     # LÓGICA ESPARSA (Binária)
#     # ----------------------------------------------------
#     # Condição de Sucesso:
#     # 1. Cubo está a menos de 5cm do alvo (tolerância de erro)
#     # 2. (Opcional mas recomendado) Cubo não está no chão (Z > 3cm)
    
#     box_z = data.xpos[self._obj_body][2]
#     is_lifted = (box_z > 0.03).astype(float)
    
#     # Sucesso = Perto do alvo E Levantado
#     is_success = (dist_to_target < 0.05) * is_lifted
#     is_success = is_success.astype(float)

#     # Dicionário Limpo: Apenas o sucesso importa.
#     # Mantivemos apenas para evitar comportamentos destrutivos,
#     # mas você pode remover se quiser purismo total.
#     rewards = {
#         "sparse_success": is_success, 
#     }
    
#     return rewards

  def _get_obs(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
    gripper_pos = data.site_xpos[self._gripper_site]
    gripper_mat = data.site_xmat[self._gripper_site].ravel()
    target_mat = math.quat_to_mat(data.mocap_quat[self._mocap_target])
    obs = jp.concatenate([
        data.qpos,
        data.qvel,
        gripper_pos,
        gripper_mat[3:],
        data.xmat[self._obj_body].ravel()[3:],
        data.xpos[self._obj_body] - data.site_xpos[self._gripper_site],
        info["target_pos"] - data.xpos[self._obj_body],
        target_mat.ravel()[:6] - data.xmat[self._obj_body].ravel()[:6],
        data.ctrl - data.qpos[self._robot_qposadr],
        #data.ctrl - data.qpos[self._robot_qposadr[:-1]],
    ])

    return obs


class Ur3PickCubeOrientation(Ur3PickCube):
    """Bring a box to a target and orientation."""

    def __init__(
        self,
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ):
        # Pass sample_orientation=True to the base class
        super().__init__(config, config_overrides, sample_orientation=True)
