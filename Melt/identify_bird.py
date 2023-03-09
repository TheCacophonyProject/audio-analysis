from pathlib import Path
import librosa
import tensorflow as tf
import numpy as np
import logging
import sys
import json
import audioread.ffdec  # Use ffmpeg decoder
import math

fmt = "%(process)d %(thread)s:%(levelname)7s %(message)s"

logging.basicConfig(
    stream=sys.stderr, level=logging.INFO, format=fmt, datefmt="%Y-%m-%d %H:%M:%S"
)
PROB_THRESH = 0.8


def load_recording(file, resample=48000):
    # librosa.load(file) giving strange results
    aro = audioread.ffdec.FFmpegAudioFile(file)
    frames, sr = librosa.load(aro)
    aro.close()
    if resample is not None and resample != sr:
        frames = librosa.resample(frames, orig_sr=sr, target_sr=resample)
        sr = resample
    return frames, sr


def load_samples(path, segment_length, stride, hop_length=640, mean_sub=False):
    logging.debug(
        "Loading samples with length %s stride %s hop length %s and mean_sub %s",
        segment_length,
        stride,
        hop_length,
        mean_sub,
    )
    frames, sr = load_recording(path)
    mels = []
    i = 0
    n_fft = sr // 10
    # hop_length = 640  # feature frame rate of 75

    sample_size = int(sr * segment_length)
    jumps_per_stride = int(sr * stride)
    length = len(frames) / sr
    end = segment_length
    mel_samples = []
    i = 0
    while end < (length + stride):
        if end > length:
            # always use end ofr last sample
            data = frames[-sample_size:]
        else:
            data = frames[i * jumps_per_stride : i * jumps_per_stride + sample_size]
        if len(data) != sample_size:
            sample = np.zeros((sample_size))
            sample[: len(data)] = data
            data = sample
        end += stride
        # /start = int(jumps_per_stride * (i * stride))
        mel = librosa.feature.melspectrogram(
            y=data,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            fmin=50,
            fmax=11000,
            n_mels=80,
        )
        half = mel[:, 75:]
        if np.amax(half) == np.amin(half):
            # noting usefull here stop early
            strides_per = math.ceil(segment_length / 2.0 / stride) + 1
            mel_samples = mel_samples[:-strides_per]
            break
        mel = librosa.power_to_db(mel)
        # end = start + sample_size
        if mean_sub:
            mel_m = tf.reduce_mean(mel, axis=1)
            mel_m = tf.expand_dims(mel_m, axis=1)
            mel = mel - mel_m

        mel_samples.append(mel)
        i += 1
    return np.array(mel_samples), len(frames) / sr


def load_model(model_path):
    logging.debug("Loading %s", model_path)
    model_path = Path(model_path)
    model = tf.keras.models.load_model(
        str(model_path),
        compile=False,
    )
    # model.load_weights(model_path / "val_binary_accuracy").expect_partial()
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
    hop_length = meta.get("hop_length", 640)
    mean_sub = meta.get("mean_sub", False)
    model_name = meta.get("name", False)

    samples, length = load_samples(
        file, segment_length, segment_stride, hop_length, mean_sub=mean_sub
    )
    predictions = model.predict(samples, verbose=0)
    tracks = []
    start = 0
    active_tracks = {}
    for prediction in predictions:
        # last sample always ends at length of audio rec
        if start + segment_length > length:
            start = length - segment_length
        specific_bird = False
        results = []
        track_labels = []
        if multi_label:
            for i, p in enumerate(prediction):
                if p >= PROB_THRESH:
                    label = labels[i]
                    results.append((p, label))
                    track_labels.append(label)
                    specific_bird = specific_bird or label not in [
                        "human",
                        "noise",
                        "bird",
                    ]

        else:
            best_i = np.argmax(prediction)
            best_p = prediction[best_i]
            if best_p >= PROB_THRESH:
                label = labels[best_i]
                results.append((best_p, label))
                track_labels.append(label)
                specific_bird = label not in ["human", "noise", "bird"]

        # remove tracks that have ended
        existing_tracks = list(active_tracks.keys())
        for existing in existing_tracks:
            track = active_tracks[existing]
            if track.label not in track_labels or (
                track.label == "bird" and specific_bird
            ):
                if specific_bird:
                    track.end = start
                else:
                    track.end = min(length, track.end - segment_length / 2)
                del active_tracks[track.label]

        for r in results:
            label = r[1]
            if specific_bird and label == "bird":
                continue
            track = active_tracks.get(label, None)
            if track is None:
                track = Track(label, start, start + segment_length, r[0], model_name)
                tracks.append(track)
                active_tracks[label] = track
            else:
                track.end = min(length, start + segment_length)
                track.confidences.append(r[0])
            # else:

        # elif track is not None:
        #     track.end = start + (segment_length / 2 - segment_stride)
        #     tracks.append((track))
        #     track = None

        start += segment_stride
    return [t.get_meta() for t in tracks], length


class Track:
    def __init__(self, label, start, end, confidence, model_name):
        self.start = start
        self.label = label
        self.end = end
        self.confidences = [confidence]
        self.model = model_name

    def get_meta(self):
        meta = {}
        meta["model"] = self.model
        meta["begin_s"] = self.start
        meta["end_s"] = self.end
        meta["species"] = self.label
        likelihood = float(round((100 * np.mean(np.array(self.confidences))), 2))
        meta["likelihood"] = likelihood
        return meta
