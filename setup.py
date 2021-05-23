import os
import codecs
from setuptools import setup, find_packages

with open('requirements.txt', encoding="utf-8") as f:
    requirements = f.readlines()


def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")


# https://pypi.org/pypi?%3Aaction=list_classifiers
classifiers = [
    'Development Status :: 5 - Production/Stable',
    'Topic :: Software Development :: Libraries :: Python Modules',
    'License :: OSI Approved :: MIT License',
    "Environment :: Console",
    "Operating System :: OS Independent",
    "Intended Audience :: Science/Research",
    "Intended Audience :: Developers",
    "Intended Audience :: Financial and Insurance Industry",
    "Intended Audience :: Healthcare Industry",
    "Topic :: Scientific/Engineering",
    "Framework :: IPython",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.6",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
]

setup(name="data-purifier",
      version=get_version("datapurifier/__init__.py"),
      description="A Python library for Automated Exploratory Data Analysis, Automated Data Cleaning and Automated Data Preprocessing For Machine Learning and Natural Language Processing Applications in Python.",
      url="https://github.com/Elysian01/Data-Purifier",
      long_description=open("PYPI_README.md", encoding='utf-8').read(),
      long_description_content_type='text/markdown',
      keywords="automated eda exploratory-data-analysis data-cleaning data-preprocessing python jupyter ipython",
      author="Abhishek Manilal Gupta",
      author_email="abhig0209@gmail.com",
      license="MIT",
      classifiers=classifiers,
      python_requires=">=3.6",
      install_requires=requirements,
      extras_require={
          "notebook": [
              "jupyter-client>=6.0.0",
              "jupyter-core>=4.6.3",
              "ipywidgets>=7.5.1",
          ],
      },
      include_package_data=True,
      packages=find_packages()
      )

# for pypi => python setup.py sdist
# twine upload dist/*


# python setup.py bdist_wheel
# twine upload dist/*

# or upload specific version
# twine upload dist/data_purifier-0.2.9-py3-none-any.whl