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
    frames,
    sr,
    tracks,
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
    mels = []
    i = 0
    n_fft = sr // 10
    # hop_length = 640  # feature frame rate of 75

    sample_size = int(sr * segment_length)
    jumps_per_stride = int(sr * stride)
    length = len(frames) / sr
    end = segment_length
    mel_samples = []
    for t in tracks:
        start = t.start
        end = start + segment_length
        end = min(end, t.end)
        while True:
            data = frames[int(start * sr) : int(end * sr)]
            if len(data) != sample_size:
                sample = np.zeros((sample_size))
                sample[: len(data)] = data
                data = sample
            spect = get_spect(
                data,
                sr,
                hop_length,
                mean_sub,
                use_mfcc,
                mel_break,
                htk,
                n_mels,
                fmin,
                fmax,
                n_fft,
                power,
                db_scale,
                channels,
            )
            t.spects.append(spect)
            start = start + stride
            end = start + segment_length
            # always take 1 sample
            if end > t.end:
                break
    return tracks


def get_spect(
    data,
    sr,
    hop_length,
    mean_sub,
    use_mfcc,
    mel_break,
    htk,
    n_mels,
    fmin,
    fmax,
    n_fft,
    power,
    db_scale,
    channels=1,
):
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
    return mel


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


def get_end(frames, sr):
    hop_length = 281
    spectogram = np.abs(librosa.stft(frames, n_fft=sr // 10, hop_length=hop_length))
    mel = mel_spec(
        spectogram,
        sr,
        sr // 10,
        hop_length,
        120,
        50,
        11000,
        1750,
        power=1,
    )
    start = 0
    chunk_length = sr // hop_length
    # this is roughtly a third of our spectogram used for classification
    end = start + chunk_length
    file_length = len(frames) / sr
    while end < mel.shape[1]:
        data = mel[:, start:end]
        if np.amax(data) == np.amin(data):
            # end of data
            return start * hop_length // sr
        start = end
        end = start + chunk_length
    return file_length


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
        frames, sr = load_recording(file)
        end = get_end(frames, sr)
        signals = signal_noise(frames[: int(sr * end)], sr, hop_length)
        tracks = get_tracks_from_signals(signals)
        length = len(frames) / sr
        tracks = load_samples(
            frames,
            sr,
            tracks,
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
    if len(tracks) == 0:
        return [], length, 0
    for t in tracks:
        predictions = model.predict(np.array(t.spects), verbose=0)
        prediction = np.mean(predictions, axis=0)
        p_labels = []
        for i, p in enumerate(prediction):
            if p >= prob_thresh:
                label = labels[i]
                p_labels.append(label)
        t.labels = p_labels
        t.confidences = predictions
        t.model = model_name
    sorted_tracks = []
    for t in tracks:
        for l in t.labels:
            if l in bird_labels:
                sorted_tracks.append(t)
                continue
    # sorted_tracks = [t for t in tracks if t.label in bird_labels]
    sorted_tracks = sorted(
        sorted_tracks,
        key=lambda track: track.start,
    )
    last_end = 0
    track_index = 0
    chirps = 0
    # overlapping signals with bird tracks
    for t in sorted_tracks:
        start = t.start
        end = t.end
        if start < last_end:
            start = last_end
            end = max(start, end)
        for s in signals:
            if segment_overlap((start, end), (s.start, s.end)):
                chirps += 1
            elif s.start > start:
                break
        last_end = t.end

    return [t.get_meta() for t in tracks], length, chirps


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
    freq_range = 100
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
        signals.append(Signal(start, end, freq_range[0], freq_range[1]))

    return signals


def segment_overlap(first, second):
    return (
        (first[1] - first[0])
        + (second[1] - second[0])
        - (max(first[1], second[1]) - min(first[0], second[0]))
    )


def mel_freq(f):
    return 2595.0 * np.log10(1.0 + f / 700.0)


# an attempt at getting frequency based tracks
# try and merge signals that are close together in time and frequency


def get_tracks_from_signals(signals):
    unique_signals = []
    f_overlap = 200
    for s in signals:
        merged = False
        for u_i, u in enumerate(unique_signals):
            overlap = s.time_overlap(u)
            mel_overlap = s.mel_freq_overlap(u)
            range = u.mel_freq_range
            range *= 0.75
            if overlap > u.length * 0.75 and mel_overlap > u.mel_freq_range * 0.5:
                u.merge(s)
                merged = True
                break
            elif mel_overlap > u.mel_freq_range * 0.75 and (s.start - u.end) <= 1:
                u.merge(s)
                merged = True
                break
        if not merged:
            unique_signals.append(s)
    to_delete = []
    min_length = 0.5
    for s in unique_signals:
        s.enlarge(1.4)
    for s in unique_signals:
        if s in to_delete:
            continue
        if s.length < min_length:
            to_delete.append(s)
            continue
        for s2 in unique_signals:
            if s2 in to_delete:
                continue
            if s == s2:
                continue
            overlap = s.time_overlap(s2)
            engulfed = overlap >= 0.9 * s2.length
            f_overlap = s.mel_freq_overlap(s2)
            range = s2.mel_freq_range
            range *= 0.7
            if f_overlap > range and engulfed:
                to_delete.append(s2)
    for s in to_delete:
        unique_signals.remove(s)
    return unique_signals


class Signal:
    def __init__(self, start, end, freq_start, freq_end):
        self.start = start
        self.end = end
        self.freq_start = freq_start
        self.freq_end = freq_end

        self.mel_freq_start = mel_freq(freq_start)
        self.mel_freq_end = mel_freq(freq_end)
        self.spects = []

        self.model = None
        self.labels = None
        self.confidences = None

    def time_overlap(self, other):
        return segment_overlap(
            (self.start, self.end),
            (other.start, other.end),
        )

    def mel_freq_overlap(self, other):
        return segment_overlap(
            (self.mel_freq_start, self.mel_freq_end),
            (other.mel_freq_start, other.mel_freq_end),
        )

    def freq_overlap(s, s2):
        return segment_overlap(
            (self.mel_freq_start, self.mel_freq_end),
            (other.mel_freq_start, other.mel_freq_end),
        )

    @property
    def mel_freq_range(self):
        return self.mel_freq_end - self.mel_freq_start

    @property
    def freq_range(self):
        return self.freq_end - self.freq_start

    @property
    def length(self):
        return self.end - self.start

    def enlarge(self, scale):
        new_length = self.length * scale
        extension = (new_length - self.length) / 2
        self.start = self.start - extension
        self.end = self.end + extension
        self.start = max(self.start, 0)

    def merge(self, other):
        self.start = min(self.start, other.start)
        self.end = max(self.end, other.end)
        self.freq_start = min(self.freq_start, other.freq_start)
        self.freq_end = max(self.freq_end, other.freq_end)
        self.mel_freq_start = mel_freq(self.freq_start)
        self.mel_freq_end = mel_freq(self.freq_end)

    def __str__(self):
        return f"Signal: {self.start}-{self.end} f: {self.freq_start}-{self.freq_end}"

    def get_meta(self):
        meta = {}
        meta["model"] = self.model
        meta["begin_s"] = self.start
        meta["end_s"] = self.end
        meta["species"] = self.labels
        meta["freq_start"] = self.freq_start
        meta["freq_end"] = self.freq_end
        likelihood = float(round((100 * np.mean(np.array(self.confidences))), 2))
        meta["likelihood"] = likelihood
        return meta
