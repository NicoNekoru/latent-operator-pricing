from setuptools import setup, find_packages

setup(
    name="neural_operator_market",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch",
        "numpy",
        "pandas",
        "matplotlib",
        "plotly",
        "yfinance",
        "scipy",
        "kaleido"
    ],
)
