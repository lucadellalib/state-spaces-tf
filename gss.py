"""Gated state space (GSS) layer."""

# Adapted from:
# https://github.com/lucidrains/gated-state-spaces-pytorch/blob/v0.0.16/gated_state_spaces_pytorch/gss.py

from typing import Optional

import tensorflow as tf

from dss import DSS


__all__ = ["GSS"]


class GSS(tf.Module):
    """Gated state space (GSS) layer.

    Parameters
    ----------
    input_size:
        The input size (i.e. the number of features).
    gating_size:
        The gating size (recommended `input_size * 4`).
    hidden_size:
        The hidden size (recommended `input_size / 4`).
    state_size:
        The state size.
    name:
        The module name.

    References
    ----------
    .. [1] H. Mehta, A. Gupta, A. Cutkosky, and B. Neyshabur.
           "Long Range Language Modeling via Gated State Spaces".
           In: International Conference on Learning Representations (ICLR). 2023.
           URL: https://arxiv.org/abs/2206.13947v3

    Examples
    --------
    >>> import tensorflow as tf
    >>>
    >>> from gss import GSS
    >>>
    >>>
    >>> batch_size = 4
    >>> seq_length = 2048
    >>> input_size = 64
    >>> model = GSS(input_size)
    >>> input = tf.random.normal([batch_size, seq_length, input_size])
    >>> output = model(input)

    """

    def __init__(
        self,
        input_size: "int",  # E
        gating_size: "int" = 256,  # F
        hidden_size: "int" = 16,  # H
        state_size: "int" = 512,  # N
        name: "Optional[str]" = None,
    ) -> "None":
        super().__init__(name=name)
        self.input_size = input_size
        self.gating_size = gating_size
        self.hidden_size = hidden_size
        self.state_size = state_size

        # Input normalization
        self.norm = tf.keras.layers.LayerNormalization(axis=-1, name="norm")

        # DSS
        self.dss = DSS(hidden_size, state_size, name="dss")

        # Dense layers
        self.to_v = tf.keras.layers.Dense(
            gating_size, activation="gelu", use_bias=False, name="to_v"
        )
        self.to_u = tf.keras.layers.Dense(
            hidden_size, activation="gelu", use_bias=False, name="to_u"
        )
        self.to_uc = tf.keras.layers.Dense(gating_size, use_bias=False, name="to_uc")
        self.to_o = tf.keras.layers.Dense(input_size, name="to_o")

    def __call__(self, input: "tf.Tensor") -> "tf.Tensor":
        """Forward pass.

        Parameters
        ----------
        input:
            The input, shape: ``[*batch_shape, seq_length, input_size]``.

        Returns
        -------
            The output, shape: ``[*batch_shape, seq_length, input_size]``.

        """
        x = input

        # [*batch_shape, seq_length, input_size]
        residual = x
        x = self.norm(x)
        # [*batch_shape, seq_length, gating_size]
        v = self.to_v(x)
        # [*batch_shape, seq_length, hidden_size]
        u = self.to_u(x)
        # [*batch_shape, seq_length, hidden_size]
        y = self.dss(u)
        # [*batch_shape, seq_length, gating_size]
        uc = self.to_uc(y)
        # [*batch_shape, seq_length, input_size]
        o = self.to_o(uc * v)
        output = o + residual

        return output


# Example
if __name__ == "__main__":
    batch_size = 4
    seq_length = 2048
    input_size = 64
    model = GSS(input_size)
    input = tf.random.normal([batch_size, seq_length, input_size])
    output = model(input)
    print(output.shape)
