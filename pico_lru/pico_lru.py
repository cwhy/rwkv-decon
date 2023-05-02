from jax import numpy as np
from jax import vmap
from jax import lax


def forward(input_sequence, nu_log, theta_log, B_re, B_im, C_re, C_im, D, gamma_log):
    """Forward pass of the LRU layer. Output y and input_sequence are of shape (L, H)."""

    # Materializing the diagonal of Lambda and projections
    Lambda = np.exp(-np.exp(nu_log) + 1j * np.exp(theta_log))
    B_norm = (B_re + 1j * B_im) * np.expand_dims(np.exp(gamma_log), axis=-1)
    C = C_re + 1j * C_im

    # Running the LRU + output projection
    # For details on parallel scan, check discussion in Smith et al (2022).
    Lambda_elements = np.repeat(Lambda[None, ...], input_sequence.shape[0], axis=0)
    Bu_elements = vmap(lambda u: B_norm @ u)(input_sequence)
    elements = (Lambda_elements, Bu_elements)
    _, inner_states = lax.associative_scan(binary_operator_diag, elements)  # all x_k
    y = vmap(lambda x, u: (C @ x).real + D * u)(inner_states, input_sequence)

    return y
