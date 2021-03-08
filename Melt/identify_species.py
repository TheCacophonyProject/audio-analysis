
import librosa
import numpy as np
import os
import tensorflow as tf
print(tf.__version__)

frequency_min = 600
frequency_max = 1200
num_bands = int((frequency_max - frequency_min) / 10)
slices_per_second = 20
seconds_per_sample = 3.0
slices_per_sample = int(slices_per_second * seconds_per_sample)
sample_slide_seconds = 1.0
sample_slide_slices = int(sample_slide_seconds * slices_per_second)
accept_threshold = 0.5

model_file_name = 'saved_model.pb'


def _load_sample(path):
    frames, sr = librosa.load(path, sr=None)

    # generate spectrogram
    nfft = int(sr / 10)
    stft = librosa.stft(frames, n_fft=nfft, hop_length=int(nfft / 2))
    npspec = np.abs(stft)[int(frequency_min / 10):int(frequency_max / 10)]

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

def _find_likely_span(liklihoods, first, last):
    count = last - first
    if count == 0:
        end = first + seconds_per_sample
        return liklihoods[first], first, end
    elif count == 1:
        liklihood = max(liklihoods[first], liklihoods[last])
        end = first + seconds_per_sample
        return liklihood, first + 1, end
    elif count == 2:
        max_liklihood = max(liklihoods[first:last + 1])
        min_liklihood = min(liklihoods[first:last + 1])
        if max_liklihood == liklihoods[first + 1]:
            start = first + 1
            end = start + seconds_per_sample
            return max_liklihood, start, end
        elif min_liklihood == liklihoods[first]:
            start = first + 1
            end = last + seconds_per_sample
            return max_liklihood, start, end
        elif min_liklihood == liklihoods[last]:
            end = first + 1 + seconds_per_sample
            return max_liklihood, first, end
        else:
            end = last + seconds_per_sample
            return max_liklihood, first, end
    else:
        max_liklihood = max(liklihoods[first:last + 1])
        if max_liklihood > liklihoods[first]:
            start = first + 1
            if max_liklihood > liklihoods[last]:
                end = last - 1 + seconds_per_sample
                return max_liklihood, start, end
            else:
                end = last + seconds_per_sample
                return max_liklihood, start, end
        elif max_liklihood > liklihoods[last]:
            end = last - 1 + seconds_per_sample
            return max_liklihood, first, end
        else:
            start = first + 1
            end = last - 1 + seconds_per_sample
            return max_liklihood, start, end


def identify_species(recording, metadata, models):

    # get spectrogram to be checked
    sr, npspec = _load_sample(recording)

    # divide recording into samples of appropriate length
    samples = []
    for base in range(0, npspec.shape[1], sample_slide_slices):
        limit = base + slices_per_sample
        if limit > npspec.shape[1]:
            limit = npspec.shape[1]
        start = limit - slices_per_sample
        sample = npspec[:, start:limit]
        sample = librosa.power_to_db(sample, ref=np.max)
        sample = sample / abs(sample.min()) + 1.0
        samples.append(sample.reshape(sample.shape + (1,)))
    samples = np.array(samples)

    # accumulate results from all models
    activations_sum = np.zeros(len(samples))
    model_paths = _model_paths(models)
    for path in model_paths:
        model = tf.keras.models.load_model(path)
        activations = model.predict(samples).flatten()
        activations_sum += activations

    def entry(begin, end, species, activation):
        entry = {}
        entry['begin_s'] = begin * sample_slide_seconds
        entry['end_s'] = (i - 1) * sample_slide_seconds + seconds_per_sample
        entry['species'] = species
        entry['liklihood'] = round(activation * 0.01, 2)
        return entry

    # generate labels from summed activations
    labels = []
    liklihoods = [round(v * 33.33333333) for v in activations_sum]
    first = -1
    for i in range(len(samples)):
        if activations_sum[i] >= 1.0:
            if first < 0:
                first = i
            last = i
        elif first >= 0:
            liklihood, start, end = _find_likely_span(liklihoods, first, last)
            labels.append(entry(start, end, 'morepork', liklihood))
            first = -1
    if first >= 0:
        liklihood, start, end = _find_likely_span(liklihoods, first, last)
        labels.append(entry(start, end, 'morepork', liklihood))
    return labels