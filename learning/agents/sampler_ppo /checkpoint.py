# Copyright 2025 The Brax Authors.
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

"""Checkpointing for GMMPPO."""

from typing import Any, Union

from brax.training import checkpoint
from brax.training import types
from agents.gmmppo import networks as gmmppo_networks
from etils import epath
from ml_collections import config_dict

_CONFIG_FNAME = 'gmmppo_network_config.json'


def save(
    path: Union[str, epath.Path],
    step: int,
    params: Any,
    config: config_dict.ConfigDict,
):
  """Saves a checkpoint."""
  return checkpoint.save(path, step, params, config, _CONFIG_FNAME)


def load(
    path: Union[str, epath.Path],
):
  """Loads checkpoint."""
  return checkpoint.load(path)


def network_config(
    observation_size: types.ObservationSize,
    action_size: int,
    normalize_observations: bool,
    network_factory: types.NetworkFactory[gmmppo_networks.GMMPPONetworks],
) -> config_dict.ConfigDict:
  """Returns a config dict for re-creating a network from a checkpoint."""
  return checkpoint.network_config(
      observation_size, action_size, normalize_observations, network_factory
  )


def _get_gmmppo_network(
    config: config_dict.ConfigDict,
    network_factory: types.NetworkFactory[gmmppo_networks.GMMPPONetworks],
) -> gmmppo_networks.GMMPPONetworks:
  """Generates a GMMPPO network given config."""
  return checkpoint.get_network(config, network_factory)  # pytype: disable=bad-return-type


def load_config(
    path: Union[str, epath.Path],
) -> config_dict.ConfigDict:
  """Loads GMMPPO config from checkpoint."""
  path = epath.Path(path)
  config_path = path / _CONFIG_FNAME
  return checkpoint.load_config(config_path)


def load_policy(
    path: Union[str, epath.Path],
    network_factory: types.NetworkFactory[
        gmmppo_networks.GMMPPONetworks
    ] = gmmppo_networks.make_gmmppo_networks,
    deterministic: bool = True,
):
  """Loads policy inference function from GMMPPO checkpoint."""
  path = epath.Path(path)
  config = load_config(path)
  params = load(path)
  gmmppo_network = _get_gmmppo_network(config, network_factory)
  make_inference_fn = gmmppo_networks.make_inference_fn(gmmppo_network)

  return make_inference_fn(params, deterministic=deterministic)
