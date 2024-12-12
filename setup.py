from setuptools import setup, find_packages

setup(
    name="mpae",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A description of your package",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/kalxed/multimodal-protein-ae",  # Optional
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        "torch "
    ],
)

