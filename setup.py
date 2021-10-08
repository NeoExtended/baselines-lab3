from setuptools import find_packages, setup

setup(
    name="baselines_lab3",
    packages=[
        package for package in find_packages() if package.startswith("baselines_lab3")
    ],
    version="1.0.0",
    author="Matthias Konitzny",
    description="Adds additional environments to the OpenAI Gym package",
    install_requires=[
        "gym",
        "numpy",
        "stable-baselines3",
        "optuna",
        "imageio",
        "torch>=1.8.1",
        "pyyaml",
        "matplotlib",
    ],
    extras_require={"visualization": ["plotly", "scikit-learn"]},
    include_package_data=True,
    package_data={"": ["mapdata/*.csv"]},
)
