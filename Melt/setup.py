from distutils.core import setup

scripts = [
    "cacophony_index.py",
    "chain.py",
    "common.py",
    "identify_species.py",
    "identify_bird.py",
]

reqs = [
    "librosa",
    "numpy",
    "scipy",
    "tensorflow",
]

data = [("bin/model", ["model/model_sd_aa.h5"])]

setup(
    name="melt",
    version="0.13",
    description="A multitool for turning raw data, mostly audio, into structured information.",
    scripts=scripts,
    install_requires=reqs,
    data_files=data,
)
