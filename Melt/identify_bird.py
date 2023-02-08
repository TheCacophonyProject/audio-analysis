from pathlib import Path
import librosa
import tensorflow as tf
import numpy as np
import logging
import sys
import json

SEG_LENGTH = 3
SEG_STRIDE = 1

fmt = "%(process)d %(thread)s:%(levelname)7s %(message)s"

logging.basicConfig(
    stream=sys.stderr, level=logging.INFO, format=fmt, datefmt="%Y-%m-%d %H:%M:%S"
)


def load_samples(path):
    frames, sr = librosa.load(path, sr=None)
    mels = []
    i = 0
    n_fft = sr // 10
    hop_length = 640  # feature frame rate of 75

    mel_all = librosa.feature.melspectrogram(
        y=frames,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        fmin=50,
        fmax=11000,
        n_mels=80,
    )
    mel_all = librosa.power_to_db(mel_all, ref=np.max)
    mel_sample_size = int(1 + SEG_LENGTH * sr / hop_length)
    jumps_per_stride = int(mel_sample_size / 3.0)

    length = mel_all.shape[1]
    end = 0
    mel_samples = []
    i = 0
    while end < length:
        start = int(jumps_per_stride * (i * SEG_STRIDE))
        end = start + mel_sample_size
        mel = mel_all[:, start:end].copy()
        mel_m = tf.reduce_mean(mel, axis=1)
        mel_m = tf.expand_dims(mel_m, axis=1)
        mel = mel - mel_m
        if mel.shape[1] != 226:
            # pad with zeros
            empty = np.zeros(((80, 226)))
            empty[:, : mel.shape[1]] = mel
            mel = empty

        mel_samples.append(mel)
        i += 1
    return np.array(mel_samples), len(frames) / sr


def load_model(model_path):
    logging.debug("Loading %s", model_path)
    model_path = Path(model_path)
    model = tf.keras.models.load_model(model_path)
    model.load_weights(model_path / "val_accuracy").expect_partial()
    meta_file = model_path / "metadata.txt"
    with open(meta_file, "r") as f:
        meta = json.load(f)
    return model, meta


def classify(file, model_file):
    global SEG_LENGTH, SEG_STRIDE
    samples, length = load_samples(file)
    model, meta = load_model(model_file)
    labels = meta.get("labels")
    predictions = model.predict(samples, verbose=0)

    track = None
    tracks = []
    start = 0
    for prediction in predictions:
        best_i = np.argmax(prediction)
        best_p = prediction[best_i]
        label = labels[best_i]
        if best_p > 0.7:
            if track is None:
                track = Track(label, start, start + SEG_LENGTH, best_p)
            elif track.label != label:
                track.end = start
                tracks.append((track))
                track = Track(label, start, start + SEG_LENGTH, best_p)
            else:
                track.confidences.append(best_p)
        elif track is not None:
            track.end = start + (SEG_LENGTH / 2 - SEG_STRIDE)
            tracks.append((track))
            track = None

        start += SEG_STRIDE

    if track is not None:
        track.end = length
        track.confidences.append(best_p)
        tracks.append((track))

    return [t.get_meta() for t in tracks]


class Track:
    def __init__(self, label, start, end, confidence):
        self.start = start
        self.label = label
        self.end = end
        self.confidences = [confidence]

    def get_meta(self):
        meta = {}
        meta["begin_s"] = self.start
        meta["end_s"] = self.end
        meta["species"] = self.label
        likelihood = float(round((100 * np.mean(np.array(self.confidences))), 2))
        meta["likelihood"] = likelihood
        return meta
