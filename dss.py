"""Simplified diagonal state space (DSS) layer."""

# Adapted from:
# https://github.com/lucidrains/gated-state-spaces-pytorch/blob/v0.0.16/gated_state_spaces_pytorch/gss.py

from typing import Optional

import tensorflow as tf


__all__ = ["DSS"]


class DSS(tf.Module):
    """Simplified diagonal state space (DSS) layer.

    Parameters
    ----------
    input_size:
        The input size (i.e. the number of features).
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
    >>> from dss import DSS
    >>>
    >>>
    >>> batch_size = 4
    >>> seq_length = 2048
    >>> input_size = 64
    >>> model = DSS(input_size)
    >>> input = tf.random.normal([batch_size, seq_length, input_size])
    >>> output = model(input)

    """

    def __init__(
        self,
        input_size: "int",  # H
        state_size: "int" = 512,  # N
        name: "Optional[str]" = None,
    ) -> "None":
        super().__init__(name=name)
        self.input_size = input_size
        self.state_size = state_size

        # Input normalization
        self.norm = tf.keras.layers.LayerNormalization(axis=-1, name="norm")

        # Lambda
        self.Lambda_real = tf.Variable(
            tf.random.normal([state_size]), name="Lambda_real"
        )
        self.Lambda_imag = tf.Variable(
            tf.random.normal([state_size]), name="Lambda_imag"
        )

        # C
        self.C_real = tf.Variable(
            tf.random.normal([input_size, state_size]), name="C_real"
        )
        self.C_imag = tf.Variable(
            tf.random.normal([input_size, state_size]), name="C_imag"
        )

        # D
        self.D = tf.Variable(tf.random.normal([input_size]), name="D")

    def _build_kernel(self, seq_length: "int") -> "tf.Tensor":
        # [state_size]
        Lambda_real = -tf.exp(self.Lambda_real)
        Lambda_imag = tf.exp(self.Lambda_imag)
        Lambda = tf.complex(Lambda_real, Lambda_imag)

        # [input_size, state_size]
        C = tf.complex(self.C_real, self.C_imag) * (tf.exp(Lambda) - 1) / Lambda

        range = tf.range(0, seq_length, dtype=tf.float32)
        # [state_size, seq_length]
        S = tf.exp(Lambda[:, None] * tf.complex(range[None, :], 0.0))

        # [input_size, seq_length]
        K = tf.math.real(tf.matmul(C, S))

        return K

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
        u = input

        # Build kernel
        seq_length = u.shape[-2]  # L
        # [input_size, seq_length]
        K = self._build_kernel(seq_length)

        # [*batch_shape, seq_length, input_size]
        u = self.norm(u)

        # Learned weighted residual
        # [*batch_shape, seq_length, input_size]
        residual = u * self.D

        # Permute last 2 axes in order to correctly apply FFT
        rank = tf.rank(u)
        perm = tf.concat([tf.range(0, rank - 2), [rank - 1], [rank - 2]], axis=0)

        # Conv1D FFT (nlog(n))
        fft_length = 2 * seq_length
        # [*batch_shape, input_size, seq_length + 1]
        u_f = tf.signal.rfft(tf.transpose(u, perm), fft_length=[fft_length])
        # [input_size, seq_length + 1]
        K_f = tf.signal.rfft(K, fft_length=[fft_length])
        # [*batch_shape, input_size, seq_length]
        y = tf.signal.irfft(u_f * K_f, fft_length=[fft_length])[..., :seq_length]
        # [*batch_shape, seq_length, input_size]
        y = tf.transpose(y, perm)
        output = y + residual

        return output


# Example
if __name__ == "__main__":
    batch_size = 4
    seq_length = 2048
    input_size = 64
    model = DSS(input_size)
    input = tf.random.normal([batch_size, seq_length, input_size])
    output = model(input)
    print(output.shape)
