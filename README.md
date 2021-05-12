# audio-analysis

Be aware that this repository contains compiled tensorflow model files.

Tensorflow model files are programs which can perform arbitrary operations on your computer.

More details here:
[https://github.com/tensorflow/tensorflow/blob/master/SECURITY.md](https://github.com/tensorflow/tensorflow/blob/master/SECURITY.md)

If you are running any branch or fork of Cacophony Project's audio-analysis code,
be sure to confirm the origin of any tensorflow model you run,
and take precautions accordingly..


# Development

Build the docker image
`docker build -t cacophony-audio .`

Then run it with the parameters:
* model_dir - model base directory
* base_dir - base directory
* audio-file - audio file relative to base_dir
* metadata-file- meta file relative to base_dir (not used at the moment)
* path to models- path to models directory contains (model1, model2, and model3) relative to model_dir

`docker run -it-v {model_dir}:/model -v {base_dir}:/io cacophony-audio /io/{audio-file} /io{metadata-file} /model/"."`


# Release
Make a tag on GitHub
This will push a docker image to docker.io to use this image update cacophony-processing tag version
To run on docker image use the following command
