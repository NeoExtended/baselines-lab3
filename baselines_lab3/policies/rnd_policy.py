import tensorflow as tf
from stable_baselines.common import tf_layers
from stable_baselines.common.policies import ActorCriticPolicy

from baselines_lab3.utils.tf_utils import build_cnn


class RndPolicy(ActorCriticPolicy):
    """
    Policy which resembles the actor CNN policy from the RND paper.
    """
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, **kwargs):
        super(RndPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=reuse, scale=True)

        with tf.variable_scope("model", reuse=reuse):
            activ = tf.nn.relu

            #extracted_features1 = nature_cnn(self.processed_obs, **kwargs)
            #extracted_features1 = tf.layers.flatten(extracted_features1)
            extracted_features = build_cnn(self.processed_obs, **kwargs)

            shared_layer = activ(tf_layers.linear(extracted_features, "fc_shared_1", 256))
            shared_layer = activ(tf_layers.linear(shared_layer, "fc_shared_2", 448))

            pi_latent = activ(tf_layers.linear(shared_layer, "fc_pi_1", 448))

            vf_latent = activ(tf_layers.linear(shared_layer, "fc_vf_1", 448))
            value_fn = tf.layers.dense(vf_latent, 1, name='vf')

            self._proba_distribution, self._policy, self.q_value = \
                self.pdtype.proba_distribution_from_latent(pi_latent, vf_latent, init_scale=0.01)

        self._value_fn = value_fn
        self._setup_init()

    def step(self, obs, state=None, mask=None, deterministic=False):
        if deterministic:
            action, value, neglogp = self.sess.run([self.deterministic_action, self.value_flat, self.neglogp],
                                                   {self.obs_ph: obs})
        else:
            action, value, neglogp = self.sess.run([self.action, self.value_flat, self.neglogp],
                                                   {self.obs_ph: obs})
        return action, value, self.initial_state, neglogp

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs})

    def value(self, obs, state=None, mask=None):
        return self.sess.run(self.value_flat, {self.obs_ph: obs})