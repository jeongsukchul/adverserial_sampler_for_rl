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

"""SAMPLERPPO networks."""

from typing import Sequence, Tuple

from brax.training import distribution
from learning.module.gmmvi.network import create_gmm_network_and_state
from learning.module.normalizing_flow.simple_flow import make_realnvp_flow_networks
from module import networks
from brax.training import types
from brax.training.types import PRNGKey
import flax
from flax import linen
import jax


@flax.struct.dataclass
class SAMPLERPPONetworks:
  policy_network: networks.FeedForwardNetwork
  value_network: networks.FeedForwardNetwork
  gmm_network: networks.FeedForwardNetwork
  flow_network: networks.FeedForwardNetwork
  parametric_action_distribution: distribution.ParametricDistribution


def make_inference_fn(samplerppo_networks: SAMPLERPPONetworks):
  """Creates params and inference function for the SAMPLERPPO agent."""

  def make_policy(
      params: types.Params, deterministic: bool = False
  ) -> types.Policy:
    policy_network = samplerppo_networks.policy_network
    parametric_action_distribution = samplerppo_networks.parametric_action_distribution

    def policy(
        observations: types.Observation, key_sample: PRNGKey
    ) -> Tuple[types.Action, types.Extra]:
      param_subset = (params[0], params[1])  # normalizer and policy params
      logits = policy_network.apply(*param_subset, observations)
      if deterministic:
        return samplerppo_networks.parametric_action_distribution.mode(logits), {}
      raw_actions = parametric_action_distribution.sample_no_postprocessing(
          logits, key_sample
      )
      log_prob = parametric_action_distribution.log_prob(logits, raw_actions)
      postprocessed_actions = parametric_action_distribution.postprocess(
          raw_actions
      )
      return postprocessed_actions, {
          'log_prob': log_prob,
          'raw_action': raw_actions,
      }

    return policy

  return make_policy


def make_samplerppo_networks(
    observation_size: types.ObservationSize,
    action_size: int,
    dynamics_param_size : int,
    num_envs :int,
    batch_size : int,
    init_key : jax.random.PRNGKey,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    policy_hidden_layer_sizes: Sequence[int] = (32,) * 4,
    value_hidden_layer_sizes: Sequence[int] = (256,) * 5,
    activation: networks.ActivationFn = linen.swish,
    policy_obs_key: str = 'state',
    value_obs_key: str = 'state',
    bound_info : Tuple = None,
) -> SAMPLERPPONetworks:
  """Make SAMPLERPPO networks with preprocessor."""
  parametric_action_distribution = distribution.NormalTanhDistribution(
      event_size=action_size
  )
  policy_network = networks.make_policy_network(
      parametric_action_distribution.param_size,
      observation_size,
      preprocess_observations_fn=preprocess_observations_fn,
      hidden_layer_sizes=policy_hidden_layer_sizes,
      activation=activation,
      obs_key=policy_obs_key,
  )
  value_network = networks.make_value_network(
      observation_size,
      preprocess_observations_fn=preprocess_observations_fn,
      hidden_layer_sizes=value_hidden_layer_sizes,
      activation=activation,
      obs_key=value_obs_key,
  )
  print("num envs", num_envs)
  print('batch size', batch_size)
  init_gmmvi_state, gmm_network = create_gmm_network_and_state(dynamics_param_size, \
                                                               num_envs, batch_size, init_key,\
                                                               prior_scale=.5,
                                                                bound_info=bound_info)
  flow_network = make_realnvp_flow_networks(
    in_channels=dynamics_param_size)
  return SAMPLERPPONetworks(
      policy_network=policy_network,
      value_network=value_network,
      gmm_network=gmm_network,
      flow_network=flow_network,
      parametric_action_distribution=parametric_action_distribution,
  ), init_gmmvi_state
