from pathlib import Path
import librosa
import tensorflow as tf
import numpy as np
import logging
import sys
import json


fmt = "%(process)d %(thread)s:%(levelname)7s %(message)s"

logging.basicConfig(
    stream=sys.stderr, level=logging.INFO, format=fmt, datefmt="%Y-%m-%d %H:%M:%S"
)


def load_samples(path, segment_length, stride, hop_length=640):
    frames, sr = librosa.load(path, sr=None)
    mels = []
    i = 0
    n_fft = sr // 10
    # hop_length = 640  # feature frame rate of 75

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
    mel_sample_size = int(1 + segment_length * sr / hop_length)
    jumps_per_stride = int(mel_sample_size / segment_length)

    length = mel_all.shape[1]
    end = 0
    mel_samples = []
    i = 0
    while end < length:
        start = int(jumps_per_stride * (i * stride))
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
    model, meta = load_model(model_file)
    labels = meta.get("labels")
    multi_label = meta.get("multi_label")
    segment_length = meta.get("segment_length", 3)
    segment_stride = meta.get("segment_stride", 1.5)

    segment_stride = meta.get("hop_length", 640)
    samples, length = load_samples(file, segment_length, segment_stride, hop_length)
    predictions = model.predict(samples, verbose=0)

    tracks = []
    start = 0
    active_tracks = {}
    for prediction in predictions:
        results = []
        track_labels = []
        if multi_label:
            for i, p in enumerate(prediction):
                if p > 0.7:
                    label = labels[i]
                    results.append((p, label))
                    track_labels.append(label)
        else:
            best_i = np.argmax(prediction)
            best_p = prediction[best_i]
            if best_p > 0.7:
                label = labels[best_i]
                results.append((best_p, label))
                track_labels.append(label)

        # remove tracks that have ended
        existing_tracks = list(active_tracks.keys())
        for existing in existing_tracks:
            track = active_tracks[existing]
            if track.label not in track_labels:
                track.end = track.end - segment_stride
                del active_tracks[track.label]

        for r in results:
            label = r[1]
            track = active_tracks.get(label, None)
            if track is None:
                track = Track(label, start, start + segment_length, r[0])
                tracks.append(track)
                active_tracks[label] = track
            else:
                track.end = start + segment_length
                track.confidences.append(r[0])
            # else:

        # elif track is not None:
        #     track.end = start + (segment_length / 2 - segment_stride)
        #     tracks.append((track))
        #     track = None

        start += segment_stride

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
