from setuptools import setup, find_packages
setup(
    name="art_attacks_plugin",
    version="0.4",
    author="MABADATA",
    author_email="mabadatabgu@gmail.com",
    description="new art attacks plugins",


    include_package_data=True,

    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    dependency_links=[
        'https://pypi.python.org/simple'
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.6",
)