from setuptools import setup, find_packages

with open("README.md") as f:
    long_description = f.read()

setup(
    name="splitone-codec",
    version="0.3.1",
    description="A small RVQ neural codec for 24 kHz speech, built for downstream LM experiments.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Xu Mingrui",
    author_email="255718431+marived@users.noreply.github.com",
    url="https://github.com/marived/splitone-codec",
    license="Apache-2.0",
    packages=find_packages(exclude=["tests*", "examples*"]),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1",
        "torchaudio>=2.1",
        "einops>=0.7",
        "numpy>=1.24",
        "pyyaml>=6.0",
    ],
    extras_require={
        "train": ["accelerate>=0.25", "wandb>=0.16", "tqdm"],
        "dev":   ["pytest", "black", "ruff"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
    ],
)
