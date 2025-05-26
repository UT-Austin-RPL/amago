from setuptools import find_packages, setup

setup(
    name="amago",
    version="3.1.0",
    author="Jake Grigsby",
    author_email="grigsby@cs.utexas.edu",
    license="MIT",
    packages=find_packages(include=["amago", "amago.*"]),
    python_requires=">=3.10",
    install_requires=[
        "gymnasium>=0.26,<=0.29.1",
        "torch>=2.5",
        "numpy",
        "gin-config",
        "wandb",
        "einops",
        "tqdm",
        "gym",
        "accelerate>=1.0",
        "termcolor",
        "huggingface_hub",
    ],
    extras_require={
        "envs": [
            "metaworld @ git+https://github.com/Farama-Foundation/Metaworld.git@04be337a12305e393c0caf0cbf5ec7755c7c8feb",
            # fix common metaworld install bug
            "cython<3",
            "popgym",
            "minigrid",
            "stable-retro",
            # deepmind alchemy
            "dm_env",
            "dm_alchemy @ git+https://github.com/deepmind/dm_alchemy.git",
            "gymnax",
            "matplotlib",
            "opencv-python",
            "procgen",
            "minigrid",
            "ale_py>=0.10",
        ],
        "flash": ["ninja", "packaging", "flash-attn"],
        "mamba": ["causal-conv1d>=1.1.0", "mamba-ssm"],
    },
)
