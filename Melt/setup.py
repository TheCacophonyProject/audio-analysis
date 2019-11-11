from distutils.core import setup

scripts = [
    'pex_entry.py',
    'cacophony_index.py',
    'common.py',
    'chain.py',
    'ensemble.py',
    'noise_reduction.py',
    'squawk.py',
]

reqs = [
    'numpy',
    'scipy',
    'tensorflow',
]

data = [
    ('bin/model', ['model/model_sd_aa.h5'])
]

setup(name='melt',
      version='0.12',
      description='A multitool for turning raw data, mostly audio, into structured information.',
      scripts=scripts,
      install_requires=reqs,
      data_files=data,
      )
