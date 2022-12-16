import librosa
import numpy as np
import os
import tensorflow as tf

frequency_min = 600
frequency_max = 1200
num_bands = int((frequency_max - frequency_min) / 10)
slices_per_second = 20
seconds_per_sample = 3.0
slices_per_sample = int(slices_per_second * seconds_per_sample)
sample_slide_seconds = 1.0
sample_slide_slices = int(sample_slide_seconds * slices_per_second)
activation_threshold = 1.0

model_file_name = "saved_model.pb"


def _load_sample(path):
    frames, sr = librosa.load(path, sr=None)

    # generate spectrogram
    nfft = int(sr / 10)
    stft = librosa.stft(frames, n_fft=nfft, hop_length=int(nfft / 2))
    npspec = np.abs(stft)[int(frequency_min / 10) : int(frequency_max / 10)]

    return sr, npspec


def _model_paths(basepath):
    namelist = os.listdir(basepath)
    pathlist = list()
    for name in namelist:
        namepath = os.path.join(basepath, name)
        if os.path.isdir(namepath):
            pathlist = pathlist + _model_paths(namepath)
        elif namepath.endswith(model_file_name):
            pathlist.append(basepath)
    return pathlist


def _find_likely_span(liklihoods, start_times, first, last):
    """
    Find the likelihood of a morepork call, and the actual time span, corresponding to a span of consecutive samples
    with morepork predicted. We're not currently predicting the actual time of a particular morepork call, but we can
    make inferences based on the assumption that every sample containing an entire morepork call will give a positive
    prediction. This uses heuristics to handle the common cases of two, three, or more samples.
    :param liklihoods: percentage liklihoods for all samples
    :type liklihoods: list(float)
    :param start_times: start time for each sample (normally same interval, but last may be shorter)
    :type start_times: list(float)
    :param first: first sample index in range with morepork predicted
    :type first: int
    :param last: last sample index in range with morepork predicted
    :type last: int
    :return: liklihood, start_time, end_time
    :rtype: float, float, float
    """
    count = last - first
    first_start_time = start_times[first]
    last_end_time = start_times[last] + seconds_per_sample
    if count == 0:
        # single isolated sample, just return the liklihood and time span for that sample
        return liklihoods[first], first_start_time, last_end_time
    elif count == 1:
        # two consecutive samples, assume call in the overlap span and return maximum liklihood with that span
        liklihood = max(liklihoods[first], liklihoods[last])
        return (
            liklihood,
            first_start_time + sample_slide_seconds,
            first_start_time + seconds_per_sample,
        )
    elif count == 2:
        # three consecutive samples, probably two calls if max likelihood are the two end values
        max_liklihood = max(liklihoods[first : last + 1])
        min_liklihood = min(liklihoods[first : last + 1])
        if max_liklihood == liklihoods[first + 1]:
            # maximum liklihood is middle sample, assume that's where the call actually is
            return (
                max_liklihood,
                start_times[first + 1],
                start_times[first + 1] + seconds_per_sample,
            )
        elif min_liklihood == liklihoods[first]:
            # lowest liklihood is the first sample, so assume call probably in overlap and perhaps a second one present
            return max_liklihood, start_times[first + 1], last_end_time
        elif min_liklihood == liklihoods[last]:
            # lowest liklihood is the last sample, so assume call probably in first and perhaps a second one present
            return (
                max_liklihood,
                first_start_time,
                start_times[first + 1] + seconds_per_sample,
            )
        else:
            # no good guessing, just return the full span
            return max_liklihood, first_start_time, last_end_time
    else:
        # more than three consecutive samples, just see if we can safely trim the non-overlapping end spans
        max_liklihood = max(liklihoods[first : last + 1])
        if max_liklihood > liklihoods[first]:
            if max_liklihood > liklihoods[last]:
                # first and last not highest likelihood, trim off the non-overlapping end spans
                return (
                    max_liklihood,
                    start_times[first + 1],
                    start_times[last - 1] + seconds_per_sample,
                )
            else:
                # last is highest likelihood, just trim off non-overlapping start
                return max_liklihood, start_times[first + 1], last_end_time
        elif max_liklihood > liklihoods[last]:
            # first is highest likelihood, last is not, just trim off non-overlapping end
            return (
                max_liklihood,
                first_start_time,
                start_times[last - 1] + seconds_per_sample,
            )
        else:
            # first and last both highest likelihood, just return the entire time
            return max_liklihood, first_start_time, last_end_time


def build_entry(begin, end, species, activation):
    entry = {}
    entry["begin_s"] = begin
    entry["end_s"] = end
    entry["species"] = species
    entry["likelihood"] = round(activation * 0.01, 2)
    return entry


def identify_species(recording, metadata, models):

    # get spectrogram to be checked
    sr, npspec = _load_sample(recording)

    # divide recording into samples of appropriate length
    samples = []
    start_times = []
    for base in range(0, npspec.shape[1], sample_slide_slices):
        limit = base + slices_per_sample
        if limit > npspec.shape[1]:
            limit = npspec.shape[1]
        start = limit - slices_per_sample
        start_times.append(start / slices_per_second)
        sample = npspec[:, start:limit]

        sample = librosa.amplitude_to_db(sample, ref=np.max)
        if sample.min() != 0:
            sample = sample / abs(sample.min()) + 1.0
        samples.append(sample.reshape(sample.shape + (1,)))
    samples = np.array(samples)

    # accumulate results from all models
    activations_sum = np.zeros(len(samples))
    model_paths = _model_paths(models)
    for path in model_paths:
        model = tf.keras.models.load_model(path)
        activations = model.predict(samples, verbose=0).flatten()
        activations_sum += activations

    # generate labels from summed activations
    labels = []
    liklihoods = [round(v * 100 / len(model_paths)) for v in activations_sum]
    first_index = -1
    for i in range(len(samples)):
        if activations_sum[i] >= activation_threshold:
            # only collect sample ranges where the summed activations are above the threshold value
            if first_index < 0:
                first_index = i
            last_index = i
        elif first_index >= 0:
            # just past the end of a sample range with activations, record it and clear
            liklihood, start_time, end_time = _find_likely_span(
                liklihoods, start_times, first_index, last_index
            )
            labels.append(build_entry(start_time, end_time, "morepork", liklihood))
            first_index = -1
    if first_index >= 0:
        # record final sample range with activations
        liklihood, start_time, end_time = _find_likely_span(
            liklihoods, start_times, first_index, last_index
        )
        labels.append(build_entry(start_time, end_time, "morepork", liklihood))
    return labels
