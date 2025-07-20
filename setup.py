#!/usr/bin/env python3
"""
Setup script for Stock Advisor package.
"""

from setuptools import setup, find_packages
import os

# Read the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("configs/requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="stock-advisor",
    version="1.0.0",
    author="Stock Advisor Team",
    author_email="contact@stockadvisor.com",
    description="AI-Powered Stock Analysis and Prediction System",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/stock-advisor",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Office/Business :: Financial",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "black>=22.0",
            "flake8>=5.0",
            "mypy>=0.991",
        ],
        "ui": [
            "streamlit>=1.28.0",
            "plotly>=5.15.0",
        ],
        "ml": [
            "torch>=1.11.0",
            "transformers>=4.20.0",
            "openai>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "stock-advisor=scripts.main:main",
            "stock-advisor-ui=scripts.run_ui:main",
            "stock-advisor-predict=scripts.predict_3_days:main",
        ],
    },
    include_package_data=True,
    package_data={
        "stock_advisor": ["models/*.pkl", "data/*.csv"],
    },
    project_urls={
        "Bug Reports": "https://github.com/yourusername/stock-advisor/issues",
        "Source": "https://github.com/yourusername/stock-advisor",
        "Documentation": "https://stock-advisor.readthedocs.io/",
    },
)