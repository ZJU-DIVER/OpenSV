#!/usr/bin/env python
import os
import sys
from codecs import open

from setuptools import setup
from setuptools.command.test import test as TestCommand

# Version Check
CURRENT_PYTHON = sys.version_info[:2]
REQUIRED_PYTHON = (3, 10)

if CURRENT_PYTHON < REQUIRED_PYTHON:
    sys.stderr.write(
        """
==========================
Unsupported Python version
==========================
This version of OpenSV requires at least Python {}.{}, but
you're trying to install it on Python {}.{}. To resolve this,
consider upgrading to a supported Python version.
""".format(
            *(REQUIRED_PYTHON + CURRENT_PYTHON)
        )
    )
    sys.exit(1)

requires = """
transformers>=4.10.0
sentencepiece==0.1.96
# scikit-learn>=0.24.2
tqdm>=4.62.2
tensorboardX
nltk
yacs
dill
datasets
rouge==1.0.0
pyarrow
scipy
"""

# 'setup.py publish' shortcut.
if sys.argv[-1] == "publish":
    os.system("python setup.py sdist bdist_wheel")
    os.system("twine upload dist/*")
    sys.exit()

requires = {}
here = os.path.abspath(os.path.dirname(__file__))
with open("requirements.txt", "r", "utf-8") as f:
    requires = f.readlines()

about = {}
with open(os.path.join(here, "opensv", "__version__.py"), "r", "utf-8") as f:
    exec(f.read(), about)

with open("README.md", "r", "utf-8") as f:
    readme = f.read()

setup(
    name=about["__title__"],
    version=about["__version__"],
    description=about["__description__"],
    long_description=readme,
    long_description_content_type="text/markdown",
    author=about["__author__"],
    author_email=about["__author_email__"],
    url=about["__url__"],
    packages=["opensv"],
    package_data={"": ["LICENSE", "NOTICE"]},
    package_dir={"opensv": "opensv"},
    include_package_data=True,
    python_requires=">=3.10",
    install_requires=requires,
    license=about["__license__"],
    zip_safe=False,
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3 :: Only",
    ],
    extras_require={
        "security": [],
        "socks": ["PySocks>=1.5.6, !=1.5.7"],
        "use_chardet_on_py3": ["chardet>=3.0.2,<6"],
    },
    project_urls={
        "Documentation": "https://opensv.readthedocs.io",
        "Source": "https://github.com/ZJU-DIVER/OpenSV",
    },
)