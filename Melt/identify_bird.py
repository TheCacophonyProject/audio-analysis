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
    frames, sr = librosa.load(aro, sr=None)
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
        third = int(mel.shape[1] / 3)
        half = mel[:, third:]
        if np.amax(half) == np.amin(half):
            # noting usefull here stop early
            strides_per = math.ceil(segment_length / 3.0 / stride) + 1
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


def get_chirp_samples(rec_data, sr=32000, stride=1, length=5):
    start = 0

    samples = []
    while True:
        sr_s = start * sr
        sr_e = (start + length) * sr
        sr_s = int(sr_s)
        sr_e = int(sr_e)
        s = rec_data[sr_s:sr_e]
        start += stride
        if len(s) < length * sr:
            s = np.pad(s, (0, int(length * sr - len(s))))
        samples.append(s)
        if sr_e >= len(rec_data):
            break
    return np.array(samples)


def chirp_embeddings(file, stride=5):
    import tensorflow_hub as hub

    rec_data, sr = load_recording(file, resample=32000)
    samples = get_chirp_samples(rec_data, sr=sr, stride=stride)
    # Load the model.
    model = hub.load("https://tfhub.dev/google/bird-vocalization-classifier/1")

    embeddings = []
    for s in samples:
        logits, embedding = model.infer_tf(s[np.newaxis, :])
        embeddings.append(embedding[0])
    return np.array(embeddings), len(rec_data) / sr


def yamn_embeddings(file, stride=1):
    import tensorflow_hub as hub

    rec_data, sr = load_recording(file, resample=16000)
    samples = get_chirp_samples(rec_data, sr=sr, stride=stride, length=3)
    # Load the model.
    model = hub.load("https://tfhub.dev/google/yamnet/1")
    # model = hub.load("https://tfhub.dev/google/bird-vocalization-classifier/1")

    embeddings = []
    for s in samples:
        logits, embedding, _ = model(s)
        embeddings.append(embedding)
    return np.array(embeddings), len(rec_data) / sr


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
    if model_name == "embeddings":
        samples, length = chirp_embeddings(file, segment_stride)
    elif model_name == "yamn-embeddings":
        samples, length = yamn_embeddings(file, segment_stride)
    else:
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
        return [], length, 0

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
                track.end = min(start + segment_length - segment_stride, track.end)
                track.end = min(length, track.end)
                if start >= track.end:
                    # with smaller strides may have overlaps
                    del active_tracks[track.label]

        for r in results:
            label = r[1]
            if specific_bird and label == "bird":
                continue
            track = active_tracks.get(label, None)
            if track is None:
                track = Track(label, start, start + segment_length, r[0], model_name)
                track.end = min(track.end, length)

                tracks.append(track)
                active_tracks[label] = track
            else:
                track.end = start + segment_length
                if track.end > length:
                    track.end = length
                track.confidences.append(r[0])
            # else:

        # elif track is not None:
        #     track.end = start + (segment_length / 2 - segment_stride)
        #     tracks.append((track))
        #     track = None

        start += segment_stride
    chirps = 0
    tracks = [t for t in tracks if t.end > t.start]
    signal_length = len(samples) * segment_stride + segment_length
    signals = signal_noise(frames[: int(sr * signal_length)], sr, hop_length)
    sorted_tracks = [t for t in tracks if t.label in bird_labels]
    sorted_tracks = sorted(
        sorted_tracks,
        key=lambda track: track.start,
    )
    last_end = 0
    track_index = 0
    # overlapping signals with bird tracks
    for t in sorted_tracks:
        start = t.start
        end = t.end
        if start < last_end:
            start = last_end
            end = max(start, end)
        for s in signals:
            if ((end - start) + (s[1] - s[0])) > max(end, s[1]) - min(start, s[0]):
                chirps += 1
            elif s[0] > start:
                break
        last_end = t.end

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

    signal = signal.astype(np.uint8)
    kernel = np.ones((4, 4), np.uint8)
    signal = cv2.morphologyEx(signal, cv2.MORPH_OPEN, kernel)

    min_width = 0.1
    min_width = min_width * sr / hop_length
    min_width = int(min_width)
    width = 0.25  # seconds
    width = width * sr / hop_length
    width = int(width)
    freq_range = 1000
    height = 0
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    for i, f in enumerate(freqs):
        if f > freq_range:
            height = i + 1
            break

    signal = cv2.dilate(signal, np.ones((height, width), np.uint8))
    signal = cv2.erode(signal, np.ones((height // 10, width), np.uint8))

    components, small_mask, stats, _ = cv2.connectedComponentsWithStats(signal)
    stats = stats[1:]
    stats = sorted(stats, key=lambda stat: stat[0])
    stats = [s for s in stats if s[2] > min_width]

    i = 0
    # indicator_vector = np.uint8(indicator_vector)
    s_start = -1
    signals = []

    bins = len(freqs)
    for s in stats:
        max_freq = min(len(freqs) - 1, s[1] + s[3])
        freq_range = (freqs[s[1]], freqs[max_freq])
        start = s[0] * 281 / sr
        end = (s[0] + s[2]) * 281 / sr
        signals.append((start, end, freq_range[0], freq_range[1]))

    return signals
