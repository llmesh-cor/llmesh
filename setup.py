"""
LLMESH Network Setup
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="mesh-ai-network",
    version="1.0.0",
    author="LLLLMESH Network Contributors",
    author_email="hello@mesh-ai.network",
    description="Decentralized AI Infrastructure - No Center, All Connected",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mesh-ai-network/mesh",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "asyncio>=3.4.3",
        "aiohttp>=3.8.1",
        "cryptography>=3.4.8",
        "numpy>=1.21.0",
        "pydantic>=1.8.2",
        "websockets>=10.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.2.5",
            "pytest-asyncio>=0.16.0",
            "black>=21.12b0",
            "flake8>=4.0.1",
        ],
        "ml": [
            "onnx>=1.10.0",
            "onnxruntime>=1.9.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "mesh-node=mesh.cli:main",
        ],
    },
    project_urls={
        "Bug Reports": "https://github.com/mesh-ai-network/mesh/issues",
        "Source": "https://github.com/mesh-ai-network/mesh",
        "Documentation": "https://docs.mesh-ai.network",
    },
)
