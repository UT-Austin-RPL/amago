from setuptools import find_packages, setup

setup(
    name="amago",
    version="0.0.1",
    author="Jake Grigsby",
    author_email="grigsby@cs.utexas.edu",
    license="MIT",
    packages=find_packages(include=["amago"]),
    python_requires=">=3.10",
    install_requires=[
        "gymnasium>=0.26",
        "torch>=2.0",
        "matplotlib",
        "gin-config",
        "wandb",
        "einops",
        "tqdm",
        "gym",
    ],
    extras_require={
        "envs": [
            "popgym",
            "crafter",
            "opencv-python",
            # original metaworld
            "metaworld @ git+https://github.com/Farama-Foundation/Metaworld.git@04be337a12305e393c0caf0cbf5ec7755c7c8feb",
            # fix common metaworld install bug
            "cython<3",
            # deepmind alchemy
            "dm_env",
            "dm_alchemy @ git+https://github.com/deepmind/dm_alchemy.git",
        ],
        "flash": ["ninja", "packaging", "flash-attn"],
        "mamba": ["causal-conv1d>=1.1.0", "mamba-ssm"],
    },
)
