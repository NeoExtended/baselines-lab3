from itertools import zip_longest

import numpy as np
import tensorflow as tf

from baselines_lab3.utils import tf_layers


def build_dynamic_cnn(input, arch, activ=tf.nn.leaky_relu, initializer=tf_layers.ortho_init(np.sqrt(2)), **kwargs):
    current_layer = input
    for idx, layer in enumerate(arch):
        if layer[0] == "conv":
            type, filters, filter_size, stride = layer
            current_layer = activ(tf_layers.conv(current_layer, "c{}".format(idx), n_filters=filters, filter_size=filter_size,
                                                 stride=stride, initializer=initializer, **kwargs))
        elif layer[0] == "pool":
            type, pool_size, stride = layer
            current_layer = tf.nn.max_pool2d(current_layer, pool_size, stride, 'VALID')
    to_fc = tf_layers.conv_to_fc(current_layer)
    return to_fc


def build_cnn(scaled_images, activ=tf.nn.leaky_relu, initializer=tf_layers.ortho_init(np.sqrt(2)), **kwargs):
    """
    CNN from Nature paper.

    :param scaled_images: (TensorFlow Tensor) Image input placeholder
    :param kwargs: (dict) Extra keywords parameters for the convolutional layers of the CNN
    :return: (TensorFlow Tensor) The CNN output layer
    """
    layer_1 = activ(tf_layers.conv(scaled_images, 'c1', n_filters=32, filter_size=8, stride=4, initializer=initializer, **kwargs))
    layer_2 = activ(tf_layers.conv(layer_1, 'c2', n_filters=64, filter_size=4, stride=2, initializer=initializer, **kwargs))
    layer_3 = activ(tf_layers.conv(layer_2, 'c3', n_filters=64, filter_size=3, stride=1, initializer=initializer, **kwargs))
    layer_3 = tf_layers.conv_to_fc(layer_3)
    return layer_3


def mlp_extractor(flat_observations, net_arch, act_fun, initializer=tf_layers.ortho_init(np.sqrt(2))):
    """
    Constructs an MLP that receives observations as an input and outputs a latent representation for the policy and
    a value network. The ``net_arch`` parameter allows to specify the amount and size of the hidden layers and how many
    of them are shared between the policy network and the value network. It is assumed to be a list with the following
    structure:

    1. An arbitrary length (zero allowed) number of integers each specifying the number of units in a shared layer.
       If the number of ints is zero, there will be no shared layers.
    2. An optional dict, to specify the following non-shared layers for the value network and the policy network.
       It is formatted like ``dict(vf=[<value layer sizes>], pi=[<policy layer sizes>])``.
       If it is missing any of the keys (pi or vf), no non-shared layers (empty list) is assumed.

    For example to construct a network with one shared layer of size 55 followed by two non-shared layers for the value
    network of size 255 and a single non-shared layer of size 128 for the policy network, the following layers_spec
    would be used: ``[55, dict(vf=[255, 255], pi=[128])]``. A simple shared network topology with two layers of size 128
    would be specified as [128, 128].

    :param flat_observations: (tf.Tensor) The observations to base policy and value function on.
    :param net_arch: ([int or dict]) The specification of the policy and value networks.
        See above for details on its formatting.
    :param act_fun: (tf function) The activation function to use for the networks.
    :return: (tf.Tensor, tf.Tensor) latent_policy, latent_value of the specified network.
        If all layers are shared, then ``latent_policy == latent_value``
    """
    latent = flat_observations
    policy_only_layers = []  # Layer sizes of the network that only belongs to the policy network
    value_only_layers = []  # Layer sizes of the network that only belongs to the value network

    # Iterate through the shared layers and build the shared parts of the network
    for idx, layer in enumerate(net_arch):
        if isinstance(layer, int):  # Check that this is a shared layer
            layer_size = layer
            latent = act_fun(tf_layers.linear(latent, "shared_fc{}".format(idx), layer_size, initializer=initializer))
        else:
            assert isinstance(layer, dict), "Error: the net_arch list can only contain ints and dicts"
            if 'pi' in layer:
                assert isinstance(layer['pi'], list), "Error: net_arch[-1]['pi'] must contain a list of integers."
                policy_only_layers = layer['pi']

            if 'vf' in layer:
                assert isinstance(layer['vf'], list), "Error: net_arch[-1]['vf'] must contain a list of integers."
                value_only_layers = layer['vf']
            break  # From here on the network splits up in policy and value network

    # Build the non-shared part of the network
    latent_policy = latent
    latent_value = latent
    for idx, (pi_layer_size, vf_layer_size) in enumerate(zip_longest(policy_only_layers, value_only_layers)):
        if pi_layer_size is not None:
            assert isinstance(pi_layer_size, int), "Error: net_arch[-1]['pi'] must only contain integers."
            latent_policy = act_fun(tf_layers.linear(latent_policy, "pi_fc{}".format(idx), pi_layer_size, initializer=initializer))

        if vf_layer_size is not None:
            assert isinstance(vf_layer_size, int), "Error: net_arch[-1]['vf'] must only contain integers."
            latent_value = act_fun(tf_layers.linear(latent_value, "vf_fc{}".format(idx), vf_layer_size, initializer=initializer))

    return latent_policy, latent_value


def lecun_normal(seed=None, **kwargs):
    return tf.keras.initializers.VarianceScaling(scale=1., mode="fan_in", distribution="truncated_normal", seed=seed, **kwargs)