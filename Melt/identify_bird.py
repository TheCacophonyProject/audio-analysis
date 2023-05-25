from pathlib import Path
import librosa
import tensorflow as tf
import numpy as np
import logging
import sys
import json
import audioread.ffdec  # Use ffmpeg decoder
import math
from custommel import mel_spec
import cv2

fmt = "%(process)d %(thread)s:%(levelname)7s %(message)s"

logging.basicConfig(
    stream=sys.stderr, level=logging.INFO, format=fmt, datefmt="%Y-%m-%d %H:%M:%S"
)
CALL_LENGTH = 1

DEFAULT_SPECIES = ["kiwi", "whistler", "morepork"]

DEFAULT_BIRDS = ["bird"]
DEFAULT_BIRDS.extend(DEFAULT_SPECIES)


def load_recording(file, resample=48000):
    # librosa.load(file) giving strange results
    aro = audioread.ffdec.FFmpegAudioFile(file)
    frames, sr = librosa.load(aro)
    aro.close()
    if resample is not None and resample != sr:
        frames = librosa.resample(frames, orig_sr=sr, target_sr=resample)
        sr = resample
    return frames, sr


def load_samples(
    path,
    segment_length,
    stride,
    hop_length=640,
    mean_sub=False,
    use_mfcc=False,
    mel_break=1750,
    htk=False,
    n_mels=80,
    fmin=50,
    fmax=11000,
    channels=1,
    power=2,
    db_scale=True,
):
    logging.debug(
        "Loading samples with length %s stride %s hop length %s and mean_sub %s mfcc %s break %s htk %s n mels %s fmin %s fmax %s",
        segment_length,
        stride,
        hop_length,
        mean_sub,
        use_mfcc,
        mel_break,
        htk,
        n_mels,
        fmin,
        fmax,
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
    while i == 0 or end < (length + stride):
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
        if not htk:
            mel = librosa.feature.melspectrogram(
                y=data,
                sr=sr,
                n_fft=n_fft,
                hop_length=hop_length,
                fmin=50,
                fmax=11000,
                n_mels=n_mels,
            )
        else:
            spectogram = np.abs(librosa.stft(data, n_fft=n_fft, hop_length=hop_length))
            mel = mel_spec(
                spectogram,
                sr,
                n_fft,
                hop_length,
                n_mels,
                fmin,
                fmax,
                mel_break,
                power=power,
            )
        half = mel[:, 75:]
        if np.amax(half) == np.amin(half):
            # noting usefull here stop early
            strides_per = math.ceil(segment_length / 2.0 / stride) + 1
            mel_samples = mel_samples[:-strides_per]
            break
        if db_scale:
            mel = librosa.power_to_db(mel, ref=np.max)
        mel = tf.expand_dims(mel, axis=2)

        if use_mfcc:
            mfcc = librosa.feature.mfcc(
                y=data,
                sr=sr,
                hop_length=hop_length,
                htk=True,
                fmin=50,
                fmax=11000,
                n_mels=80,
            )
            mfcc = tf.image.resize_with_pad(mfcc, *mel.shape)
            mel = tf.concat((mel, mfcc), axis=0)
        # end = start + sample_size
        if mean_sub:
            mel_m = tf.reduce_mean(mel, axis=1)
            mel_m = tf.expand_dims(mel_m, axis=1)
            mel = mel - mel_m
        if channels > 1:
            mel = tf.repeat(mel, channels, axis=2)
        mel_samples.append(mel)
        i += 1
    return frames, sr, np.array(mel_samples), len(frames) / sr


def load_model(model_path):
    model_path = Path(model_path)
    logging.debug("Loading %s", str(model_path))
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
    use_mfcc = meta.get("use_mfcc", False)
    n_mels = meta.get("n_mels", 80)
    mel_break = meta.get("mel_break", 1750)
    htk = meta.get("htk", False)
    fmin = meta.get("fmin", 50)
    fmax = meta.get("fmax", 11000)
    power = meta.get("power", 2)
    db_scale = meta.get("db_scale", True)
    bird_labels = meta.get("bird_labels", DEFAULT_BIRDS)
    bird_species = meta.get("bird_species", DEFAULT_SPECIES)
    channels = meta.get("channels", 1)
    prob_thresh = meta.get("threshold", 0.7)
    frames, sr, samples, length = load_samples(
        file,
        segment_length,
        segment_stride,
        hop_length,
        mean_sub=mean_sub,
        use_mfcc=use_mfcc,
        htk=htk,
        mel_break=mel_break,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        channels=channels,
        power=power,
        db_scale=db_scale,
    )
    if len(samples) == 0:
        return [], length
    predictions = model.predict(samples, verbose=0)
    tracks = []
    start = 0
    active_tracks = {}
    for i, prediction in enumerate(predictions):
        # last sample always ends at length of audio rec
        if start + segment_length > length:
            start = length - segment_length
        specific_bird = False
        results = []
        track_labels = []
        if multi_label:
            for i, p in enumerate(prediction):
                if p >= prob_thresh:
                    label = labels[i]
                    results.append((p, label))
                    track_labels.append(label)
                    specific_bird = specific_bird or label in bird_species

        else:
            best_i = np.argmax(prediction)
            best_p = prediction[best_i]
            if best_p >= prob_thresh:
                label = labels[best_i]
                results.append((best_p, label))
                track_labels.append(label)
                specific_bird = label in bird_species

        # remove tracks that have ended
        existing_tracks = list(active_tracks.keys())
        for existing in existing_tracks:
            track = active_tracks[existing]
            if track.label not in track_labels or (
                track.label == "bird" and specific_bird
            ):
                track.end = start + CALL_LENGTH
                del active_tracks[track.label]

        for r in results:
            label = r[1]
            if specific_bird and label == "bird":
                continue
            track = active_tracks.get(label, None)
            if track is None:
                t_s = start
                t_e = start + segment_length
                if start > 0:
                    t_s = start - segment_stride + segment_length - CALL_LENGTH

                if (i + 1) < len(predictions):
                    t_e = start + segment_stride + CALL_LENGTH
                t_e = max(t_s, t_e)
                track = Track(label, t_s, t_e, r[0], model_name)
                tracks.append(track)
                active_tracks[label] = track
            else:
                track.end = start + segment_stride + CALL_LENGTH
                track.confidences.append(r[0])
            # else:

        # elif track is not None:
        #     track.end = start + (segment_length / 2 - segment_stride)
        #     tracks.append((track))
        #     track = None

        start += segment_stride
    tracks = [t for t in tracks if t.end > t.start]
    signals, noise = signal_noise(frames, sr, hop_length)
    signals = join_signals(signals, max_gap=0.2)
    sorted_tracks = [t for t in tracks if t.label in bird_labels]
    sorted_tracks = sorted(
        sorted_tracks,
        key=lambda track: track.start,
    )
    chirps = 0
    last_end = 0
    track_index = 0
    for s in signals:
        if track_index >= len(sorted_tracks):
            break
        while track_index < len(sorted_tracks):
            t = sorted_tracks[track_index]
            start = t.start
            end = t.end
            if start < last_end:
                start = last_end
                end = max(start, end)
            # overlap
            if ((end - start) + (s[1] - s[0])) > max(end, s[1]) - min(start, s[0]):
                # print("Have track", t, " for ", s, t.start, t.end, t.label)
                if t.label in bird_labels:
                    chirps += 1
                if end > s[1]:
                    # check next signal
                    break
            elif start > s[1]:
                break
            last_end = end
            track_index += 1
    return [t.get_meta() for t in tracks], length, chirps


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


def signal_noise(frames, sr, hop_length=281):
    # frames = frames[:sr]
    n_fft = sr // 10
    # frames = frames[: sr * 3]
    spectogram = np.abs(librosa.stft(frames, n_fft=n_fft, hop_length=hop_length))

    a_max = np.amax(spectogram)
    spectogram = spectogram / a_max
    row_medians = np.median(spectogram, axis=1)
    column_medians = np.median(spectogram, axis=0)
    rows, columns = spectogram.shape

    column_medians = column_medians[np.newaxis, :]
    row_medians = row_medians[:, np.newaxis]
    row_medians = np.repeat(row_medians, columns, axis=1)
    column_medians = np.repeat(column_medians, rows, axis=0)

    signal = (spectogram > 3 * column_medians) & (spectogram > 3 * row_medians)
    noise = (spectogram > 2.5 * column_medians) & (spectogram > 2.5 * row_medians)
    noise[signal == noise] = 0
    noise = noise.astype(np.uint8)
    signal = signal.astype(np.uint8)
    kernel = np.ones((4, 4), np.uint8)
    signal = cv2.morphologyEx(signal, cv2.MORPH_OPEN, kernel)
    noise = cv2.morphologyEx(noise, cv2.MORPH_OPEN, kernel)
    # plot_spec(spectogram)

    signal_indicator_vector = np.amax(signal, axis=0)
    noise_indicator_vector = np.amax(noise, axis=0)

    signal_indicator_vector = signal_indicator_vector[np.newaxis, :]
    signal_indicator_vector = cv2.dilate(
        signal_indicator_vector, np.ones((4, 1), np.uint8)
    )
    signal_indicator_vector = np.where(signal_indicator_vector > 0, 1, 0)
    signal_indicator_vector = signal_indicator_vector * 255

    noise_indicator_vector = noise_indicator_vector[np.newaxis, :]
    noise_indicator_vector = cv2.dilate(
        noise_indicator_vector, np.ones((4, 1), np.uint8)
    )
    noise_indicator_vector = np.where(noise_indicator_vector > 0, 1, 0)

    noise_indicator_vector = noise_indicator_vector * 128

    indicator_vector = np.concatenate(
        (signal_indicator_vector, noise_indicator_vector), axis=0
    )
    i = 0
    indicator_vector = np.uint8(indicator_vector)
    s_start = -1
    noise_start = -1
    signals = []
    noise = []
    for c in indicator_vector.T:
        # print("indicator", c)
        if c[0] == 255:
            if s_start == -1:
                s_start = i
        elif s_start != -1:
            signals.append((s_start * 281 / sr, (i - 1) * 281 / sr))
            s_start = -1
        if c[1] == 128:
            if noise_start == -1:
                noise_start = i
        elif noise_start != -1:
            noise.append((noise_start * 281 / sr, (i - 1) * 281 / sr))
            noise_start = -1

        i += 1
    if s_start != -1:
        signals.append((s_start * 281 / sr, (i - 1) * 281 / sr))
    if noise_start != -1:
        noise.append((noise_start * 281 / sr, (i - 1) * 281 / sr))
    return signals, noise


# join signals that are close toghetr
def join_signals(signals, max_gap=0.1):
    new_signals = []
    prev_s = None
    for s in signals:
        if prev_s is None:
            prev_s = s
        else:
            if s[0] < prev_s[1] + max_gap:
                # combine them
                prev_s = (prev_s[0], s[1])
            else:
                new_signals.append(prev_s)
                prev_s = s
    if prev_s is not None:
        new_signals.append(prev_s)
    #
    # print("spaced have", len(new_signals))
    # for s in new_signals:
    #     print(s)
    return new_signals
