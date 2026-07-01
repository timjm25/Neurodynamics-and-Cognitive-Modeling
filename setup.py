from setuptools import setup, find_packages

setup(
    name="neurodynamics",
    version="1.0.0",
    description="Neurodynamics & Cognitive Modeling Research Platform",
    author="Neurodynamics Platform",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "matplotlib>=3.7.0",
        "networkx>=3.0",
        "fastapi>=0.100.0",
        "uvicorn>=0.23.0",
        "pydantic>=2.0.0",
        "httpx>=0.24.0",
    ],
    entry_points={
        "console_scripts": [
            "neurodynamics=neurodynamics.cli.main:main",
        ]
    },
)
