from setuptools import setup, find_packages

setup(
    name="sleep-staging-project",
    version="1.0.0",
    description="Multi-modal automatic sleep staging research project",
    author="Sleep Staging Team",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "mne>=1.0.0",
        "scikit-learn>=1.0.0",
    ],
    extras_require={
        "dev": ["pytest>=7.0.0", "black>=22.0.0", "flake8>=4.0.0"],
        "viz": ["matplotlib>=3.4.0", "seaborn>=0.11.0"],
    },
    python_requires=">=3.8",
)