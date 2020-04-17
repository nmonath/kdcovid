#!/usr/bin/env python

from distutils.core import setup

setup(name='kdcovid',
      version='0.07',
      packages=['kdcovid'],
      install_requires=[
          "nltk",
          "absl-py",
          "python-dateutil",
          "sent2vec @ git+git://github.com/nmonath/sent2vec.git"
      ],
      package_dir={'kdcovid': 'kdcovid'}
      )
