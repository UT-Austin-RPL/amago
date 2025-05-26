Installation
============

.. code-block:: shell

    # download source
    git clone git@github.com:UT-Austin-RPL/amago.git
    # make a fresh conda environment with python 3.10+
    conda create -n amago python==3.10
    conda activate amago
    # install core agent
    pip install -e amago

- ``pip install -e amago[flash]``: The base Transformer policy uses `FlashAttention 2.0 <https://github.com/Dao-AILab/flash-attention>`_ by default. We recommend installing ``flash_attn`` if your GPU is compatible. Please refer to the `official installation instructions <https://github.com/Dao-AILab/flash-attention>`_ if you run into issues.

There are some optional installs for additional features:

- ``pip install -e amago[mamba]``: Enables `Mamba <https://arxiv.org/abs/2312.00752>`_ sequence model policies.

- ``pip install -e amago[envs]``: AMAGO comes with built-in support for a wide range of existing and custom meta-RL/generalization/memory domains (:py:mod:`amago.envs.builtin`) used in our experiments. This command installs (most of) the dependencies you'd need to run the `examples/ <examples/>`_.

.. note::

   AMAGO requires ``gymnasium`` <= 0.29. It is not compatible with the recent ``gymnasium`` 1.0 release. Please check your ``gymnasium`` version if you see environment-related error messages on startup.