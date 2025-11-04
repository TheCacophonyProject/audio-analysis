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

CALL_LENGTH = 1

DEFAULT_SPECIES = ["kiwi", "whistler", "morepork"]
NON_BIRD = ["human", "noise", "insect"]
SPECIFIC_NOISE = ["insect"]

DEFAULT_BIRDS = ["bird"]
DEFAULT_BIRDS.extend(DEFAULT_SPECIES)
SIGNAL_WIDTH = 0.25
MAX_FRQUENCY = 48000 / 2


# tensorflow refuces to load without this
@tf.keras.utils.register_keras_serializable(package="MyLayers", name="MagTransform")
class MagTransform(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(MagTransform, self).__init__(**kwargs)
        self.a = self.add_weight(
            initializer=tf.keras.initializers.Constant(value=0.0),
            name="a-power",
            dtype="float32",
            shape=(),
            trainable=True,
        )

    def call(self, inputs):
        c = tf.math.pow(inputs, tf.math.sigmoid(self.a))
        return c


# roughly the max possible chirps
# assuming no more than 3 birds at any given moment
def get_max_chirps(length):
    return int(length / (SIGNAL_WIDTH + 0.01))


def load_recording(file, resample=48000):
    try:
        # librosa.load(file) giving strange results
        aro = audioread.ffdec.FFmpegAudioFile(file)
        frames, sr = librosa.load(aro, sr=None)
        aro.close()
        if resample is not None and resample != sr:
            frames = librosa.resample(frames, orig_sr=sr, target_sr=resample)
            sr = resample
        return frames, sr
    except:
        logging.error("Could not load %s", file, exc_info=True)
        # for some reason the original exception causes docker to hang
        raise Exception(f"Could not load {file}")


def load_samples(
    frames,
    sr,
    tracks,
    segment_length,
    stride,
    hop_length=281,
    mean_sub=False,
    use_mfcc=False,
    mel_break=1000,
    htk=True,
    n_mels=160,
    fmin=50,
    fmax=11000,
    channels=1,
    power=2,
    db_scale=False,
    filter_freqs=False,
    filter_below=None,
    normalize=True,
    n_fft=4096,
    pad_short_tracks=False,
):
    logging.debug(
        "Loading samples with length %s stride %s hop length %s and mean_sub %s mfcc %s break %s htk %s n mels %s fmin %s fmax %s filtering freqs %s filter below %s n_fft %s pad short tracks %s",
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
        filter_freqs,
        filter_below,
        n_fft,
        pad_short_tracks,
    )
    mels = []
    i = 0
    # hop_length = 640  # feature frame rate of 75

    sample_size = int(sr * segment_length)
    jumps_per_stride = int(sr * stride)
    length = len(frames) / sr
    end = segment_length
    mel_samples = []
    for t in tracks:
        track_data = []
        if t.freq_start > fmax or t.freq_end < fmin:
            mel_samples.append(track_data)
            # no need to id these tracks
            continue
        start = 0
        end = start + segment_length

        sr_end = int(t.end * sr)
        sr_start = int(sr * t.start)

        if pad_short_tracks:
            end = min(end, t.length)
            track_frames = frames[sr_start:sr_end]
        else:
            missing = sample_size - (sr_end - sr_start)
            if missing > 0:
                offset = np.random.randint(0, missing)
                sr_start = sr_start - offset

                if sr_start <= 0:
                    sr_start = 0
                    sr_end = sr_start + sample_size
                    sr_end = min(sr_end, len(frames))
                else:
                    end_offset = sr_end + missing - offset
                    if end_offset > len(frames):
                        end_offset = len(frames)
                        sr_start = end_offset - sample_size
                        sr_start = max(sr_start, 0)
                    sr_end = end_offset
                assert sr_end - sr_start == sample_size

            track_frames = frames[sr_start:sr_end]

        sr_start = 0
        sr_end = min(sr_end, sample_size)
        if filter_freqs:
            track_frames = butter_bandpass_filter(
                track_frames, t.freq_start, t.freq_end, sr
            )
        elif filter_below and t.freq_end < filter_below:
            logging.info(
                "Filter freq below %s %s %s", filter_below, t.freq_start, t.freq_end
            )
            track_frames = butter_bandpass_filter(
                track_frames, t.freq_start, t.freq_end, sr
            )
        while True:
            data = track_frames[sr_start:sr_end]
            if len(data) != sample_size:
                extra_frames = sample_size - len(data)
                offset = np.random.randint(0, extra_frames)
                data = np.pad(data, (offset, extra_frames - offset))
            if normalize:
                data = normalize_data(data)
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
                # pass_freqs=[t.freq_start, t.freq_end],
            )

            track_data.append(spect)
            start = start + stride
            end = start + segment_length
            sr_start = int(start * sr)
            sr_end = min(int(end * sr), sr_start + sample_size)
            # always take 1 sample
            if end > t.length:
                break

        mel_samples.append(track_data)
    return mel_samples


def normalize_data(x):
    min_v = np.min(x, -1, keepdims=True)
    x = x - min_v
    max_v = np.max(x, -1, keepdims=True)
    x = x / max_v + 0.000001
    x = x - 0.5
    x = x * 2
    return x


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
    pass_freqs=None,
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
        # if pass_freqs is not None:
        #     data = butter_bandpass_filter(data, pass_freqs[0], pass_freqs[1], sr)

        spectogram = np.abs(librosa.stft(data, n_fft=n_fft, hop_length=hop_length))
        # bins = 1 + n_fft / 2
        # max_f = sr / 2
        # gap = max_f / bins
        # if low_pass is not None:
        #     min_bin = low_pass // gap
        #     spectogram[: int(min_bin)] = 0
        #
        # if high_pass is not None:
        #     max_bin = high_pass // gap
        #     spectogram[int(max_bin) :] = 0
        mel = mel_spec(
            spectogram,
            sr,
            n_fft,
            hop_length,
            n_mels,
            50 if fmin is None else fmin,
            11000 if fmin is None else fmax,
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


def load_model_meta(model_path):
    if model_path.is_file():
        meta_file = model_path.parent / "metadata.txt"
    else:
        meta_file = model_path / "metadata.txt"

    with open(meta_file, "r") as f:
        meta = json.load(f)
    return meta


def load_model(model_path, meta):
    try:
        #     if model_path.is_file():
        #         meta_file = model_path.parent / "metadata.txt"
        #     else:
        #         meta_file = model_path / "metadata.txt"

        #     with open(meta_file, "r") as f:
        #         meta = json.load(f)

        # tensorflow being difficult about custom layers
        if meta.get("magv2", True):
            from magtransformv2 import MagTransform
        else:
            from magtransform import MagTransform

        model_path = Path(model_path)
        logging.info("Loading %s", str(model_path))
        model = tf.keras.models.load_model(
            str(model_path),
        )

    except Exception as e:
        logging.info("Could not load model", exc_info=True)
        raise e
    return model


def get_chirp_samples(rec_data, tracks, sr=32000, stride=1, length=5):
    start = 0

    samples = []
    sr_length = int(length * sr)
    sr_stride = stride * sr
    for track in tracks:
        track_samples = []
        start = track.start * sr
        start = int(start)
        while True:
            end = start + sr_length
            s = rec_data[start:end]
            if len(s) < length * sr:
                s = np.pad(s, (0, int(length * sr - len(s))))
            start += sr_stride
            track_samples.append(s)
            if end / sr > track.end:
                break
        samples.append(track_samples)
    return samples


def chirp_embeddings(file, tracks, stride=5):
    import tensorflow_hub as hub

    rec_data, sr = load_recording(file, resample=32000)
    samples = get_chirp_samples(rec_data, tracks, sr=sr, stride=stride)
    # Load the model.
    model = hub.load("https://tfhub.dev/google/bird-vocalization-classifier/1")

    embeddings = []
    for track_sample in samples:
        track_embeddings = []
        for s in track_sample:
            logits, embedding = model.infer_tf(s[np.newaxis, :])
            track_embeddings.append(embedding[0])
        embeddings.append(track_embeddings)
    return embeddings


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


def classify(file, models, analyse_tracks, meta_data=None):
    frames, sr = load_recording(file)
    raw_length = len(frames) / sr
    length = get_end(frames, sr)
    signals = signal_noise(frames[: int(sr * length)], sr, 281)
    # want to use signals for chrips
    if analyse_tracks:
        if meta_data is None:
            return None
        meta_tracks = [t for t in meta_data["Tracks"]]
        tracks = []
        for t in meta_tracks:
            freq_start = t.get("minFreq", 0)
            freq_end = t.get("maxFreq", MAX_FRQUENCY)
            # add to signals also???
            signal = Signal(t["start"], t["end"], freq_start, freq_end)
            signal.track_id = t["id"]
            tracks.append(signal)
    else:
        tracks = [s.copy() for s in signals]

        tracks = get_tracks_from_signals(tracks, length)
    if len(tracks) == 0:
        return [], length, [], raw_length, []
    track_data = None
    mel_data = None
    bird_labels = set()

    pre_models = []
    mean_models = []
    for model_file in models:
        meta = load_model_meta(Path(model_file))
        if meta.get("pre_model", False):
            pre_models.append((model_file, meta))
        else:
            mean_models.append((model_file, meta))

    grouped_models = [mean_models]
    if len(pre_models) > 0:
        grouped_models.append(pre_models)
    for model_group in grouped_models:
        predict_models = []
        if len(model_group) > 1:
            logging.info("Meaning predictions as have multiple models")
        for model_f in model_group:
            meta = model_f[1]
            model = load_model(Path(model_f[0]), meta)
            predict_models.append((model, meta))

        meta = predict_models[0][1]
        filter_freqs = meta.get("filter_freq", False)
        filter_below = meta.get("filter_below", None)

        ebird_ids = meta.get("ebird_ids")
        labels = meta.get("labels")
        multi_label = meta.get("multi_label")
        segment_length = meta.get("segment_length", 3)
        segment_stride = meta.get("segment_stride", 1.5)
        hop_length = meta.get("hop_length", 640)
        mean_sub = meta.get("mean_sub", False)
        model_name = meta.get("name", False)
        use_mfcc = meta.get("use_mfcc", False)
        n_mels = meta.get("n_mels", 160)

        pad_short = meta.get("pad_short_tracks", False)
        mel_break = meta.get("break_freq", 1750)
        htk = meta.get("htk", False)
        fmin = meta.get("fmin", 50)
        fmax = meta.get("fmax", 11000)
        power = meta.get("power", 2)
        db_scale = meta.get("db_scale", True)
        model_bird_labels = meta.get("bird_labels", DEFAULT_BIRDS)
        bird_species = meta.get("bird_species", DEFAULT_SPECIES)
        channels = meta.get("channels", 1)
        prob_thresh = meta.get("threshold", 0.7)
        bird_thresh = meta.get("bird_thresh", 0.5)
        n_fft = meta.get("n_fft", 4096)
        pre_model = meta.get("pre_model", False)

        bird_labels.update(model_bird_labels)
        if n_fft is None:
            n_fft = 4096
        normalize = meta.get("normalize", True)
        if model_name == "embeddings":
            data = chirp_embeddings(file, tracks, segment_stride)
        else:
            if track_data is None:
                track_data = load_samples(
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
                    filter_freqs=filter_freqs,
                    filter_below=filter_below,
                    normalize=normalize,
                    n_fft=n_fft,
                    pad_short_tracks=pad_short,
                )
            else:
                logging.info(
                    "Re using track data this will cuase problems if the STFT settings are not the same for multiple models"
                )
            data = track_data
        if len(data) == 0:
            return [], length, [], raw_length, []

        bird_indexes = []
        for i, l in enumerate(labels):
            bird_indexes.append(l not in NON_BIRD)
        for d, t in zip(data, tracks):
            if len(d) == 0:
                continue
            if "efficientnet" in model_name.lower():
                d = np.repeat(d, 3, -1)

            all_predictions = []
            for model, _ in predict_models:
                predictions = model.predict(np.array(d))
                all_predictions.append(predictions)

            if len(all_predictions) > 0:
                predictions = np.mean(all_predictions, axis=0)
            else:
                predictions = all_predictions[0]
            prediction = np.mean(predictions, axis=0)
            max_p = None
            result = ModelResult(model_name, pre_model)
            t.results.append(result)
            bird_prob = 0

            # just sum up confidences of all bird species as add generic tag
            # if threshold is met, probably will use seperate model for this in future
            if not multi_label and "bird" not in labels:
                for p in predictions:
                    max_i = np.argmax(p)
                    if bird_indexes[max_i]:
                        bird_prob += p[max_i]
                if len(predictions) > 0:
                    bird_prob = bird_prob / len(predictions)
                if bird_prob > bird_thresh:
                    result.add_prediction("bird", bird_prob, None)
                    # result.labels.append("bird")
                    # if ebird_ids is not None:
                    #     result.ebird_ids.append([])
                    # result.confidences.append(round(bird_prob * 100))

            for i, p in enumerate(prediction):
                if max_p is None or p > max_p[1]:
                    max_p = (i, p)
                if p >= prob_thresh:
                    ebird_id = None
                    if ebird_ids is not None:
                        ebird_id = ebird_ids[i]
                    result.add_prediction(labels[i], p, ebird_id)

            if len(result.predictions) == 0:
                # use max prediction
                ebird_id = None
                if ebird_ids is not None:
                    ebird_id = ebird_ids[max_p[0]]
                result.raw_prediction = Prediction(labels[max_p[0]], max_p[1], ebird_id)

    return tracks, length, signals, raw_length, list(bird_labels)


# sure prediction from master model
# sure prediction from pre model
# unsure prediction from master model
# unsure prediction from pre model
def get_master_tag(track):
    pre_model = None
    other_model = []
    raw_preds = []
    for model_result in track.results:
        if model_result.pre_model:
            pre_model = model_result
            continue
        for p in model_result.predictions:
            if p.filtered:
                continue
            other_model.append((p, model_result.model))
        if model_result.raw_prediction is not None:
            raw_preds.append((model_result.raw_prediction, model_result.model))
    # if other model is sure choose this first
    if len(other_model) > 0:
        ordered = sorted(
            other_model,
            key=lambda prediction: (prediction[0].confidence),
            reverse=True,
        )

        first_specific = None
        for p in ordered:
            if p[0].what == "bird":
                continue
            first_specific = p
            break
        if first_specific is None:
            first_specific = ordered[0]
        return *first_specific, False
    if pre_model is not None:
        if len(pre_model.predictions) > 0:
            pre_prediction = pre_model.predictions[0]
            if not pre_prediction.filtered:
                return pre_prediction, pre_model.model, False

    # should we set raw prediction as master tag...
    if len(raw_preds) > 0:
        ordered = sorted(
            raw_preds,
            key=lambda raw_pred: raw_pred[0].confidence,
            reverse=True,
        )
        return *ordered[0], True
    elif pre_model is not None and pre_model.raw_prediction is not None:
        return pre_model.raw_prediction, pre_model.model, True
    return None


def signal_noise(frames, sr, hop_length=281):
    # frames = frames[:sr]
    n_fft = 4096
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

    width = SIGNAL_WIDTH * sr / hop_length
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
    min_width = 0.65 * width
    min_height = height - height // 10
    stats = [s for s in stats if s[2] > min_width and s[3] > min_height]

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


def merge_signals(signals):
    unique_signals = []
    to_delete = []
    something_merged = False
    i = 0

    signals = sorted(signals, key=lambda s: s.mel_freq_end, reverse=True)
    signals = sorted(signals, key=lambda s: s.start)

    for s in signals:
        if s in to_delete:
            continue
        merged = False
        for u_i, u in enumerate(signals):
            if u in to_delete:
                continue
            if u == s:
                continue
            in_freq = u.mel_freq_end < 1500 and s.mel_freq_end < 1500
            in_freq = in_freq or u.mel_freq_end > 1500 and s.mel_freq_end > 1500
            # ensure both are either below 1500 or abov
            if not in_freq:
                continue
            overlap = s.time_overlap(u)
            if s.mel_freq_start > 1000 and u.mel_freq_start > 1000:
                freq_overlap = 0.1
                freq_overlap_time = 0.5
            else:
                freq_overlap = 0.5
                freq_overlap_time = 0.75
            if s.start > u.end:
                time_diff = s.start - u.end
            else:
                time_diff = u.start - s.end
            mel_overlap = s.mel_freq_overlap(u)
            if overlap > u.length * 0.75 and mel_overlap > -20:
                s.merge(u)
                merged = True

                break
            elif overlap > 0 and mel_overlap > u.mel_freq_range * freq_overlap_time:
                # time overlaps at all with more freq overlap
                s.merge(u)
                merged = True

                break

            elif mel_overlap > u.mel_freq_range * freq_overlap_time and time_diff <= 2:
                if u.mel_freq_end > s.mel_freq_range:
                    range_overlap = s.mel_freq_range / u.mel_freq_range
                else:
                    range_overlap = u.mel_freq_range / s.mel_freq_range
                if range_overlap < 0.75:
                    continue
                # freq range similar
                s.merge(u)
                merged = True

                break

        if merged:
            something_merged = True
            to_delete.append(u)

    for s in to_delete:
        signals.remove(s)

    return signals, something_merged


def get_tracks_from_signals(signals, end):
    # probably a much more efficient way of doing this
    # just keep merging until there are no more valid merges
    merged = True
    min_mel_range = 50
    while merged:
        signals, merged = merge_signals(signals)

    to_delete = []
    min_length = 0.35
    min_track_length = 0.7
    for s in signals:
        if s in to_delete:
            continue
        if s.length < min_length:
            to_delete.append(s)
            continue

        s.enlarge(1.4, min_track_length=min_track_length)
        s.end = min(end, s.end)

        for s2 in signals:
            if s2 in to_delete:
                continue
            if s == s2:
                continue

            overlap = s.time_overlap(s2)
            mel_overlap = s.freq_overlap(s2)
            min_length = min(s.length, s2.length)
            # 2200 chosen on testing some files may be too leniant
            # was also filtering by  and abs(mel_overlap) < 2200:
            if overlap > 0.7 * min_length:
                s.merge(s2)
                to_delete.append(s2)

    for s in to_delete:
        signals.remove(s)
    to_delete = []
    for s in signals:
        # doing earlier now
        # s.enlarge(1.4, min_track_length=min_track_length)
        # s.end = min(end, s.end)
        if s.mel_freq_range < min_mel_range:
            to_delete.append(s)
    for s in to_delete:
        signals.remove(s)
    return signals


class Prediction:
    def __init__(self, what, confidence, ebird_id, normalize_confidence=True):
        self.what = what
        if normalize_confidence:
            self.confidence = round(100 * confidence)
        else:
            self.confidence = confidence
        self.ebird_id = ebird_id
        self.filtered = False

    def get_meta(self):
        meta = {}
        meta["what"] = self.what
        meta["confidence"] = self.confidence
        meta["filtered"] = self.filtered
        meta["ebird_id"] = self.ebird_id
        return meta


class ModelResult:
    def __init__(self, model, pre_model):
        self.model = model
        self.pre_model = pre_model
        self.raw_prediction = None
        # self.raw_tag = None
        # self.raw_confidence = None
        self.predictions = []

    def add_prediction(self, what, confidence, ebird_ids, normalize_confidence=True):
        eid = ebird_ids
        if ebird_ids is not None and len(ebird_ids) == 0:
            eid = None
        p = Prediction(what, confidence, eid, normalize_confidence)
        self.predictions.append(p)

    def get_meta(self):
        meta = {}
        meta["model"] = self.model
        meta["pre_model"] = self.pre_model

        meta["predictions"] = [p.get_meta() for p in self.predictions]
        if self.raw_prediction is not None:
            meta["raw_prediction"] = self.raw_prediction.get_meta()

        # meta["filtered_species"] = self.filtered_labels
        # meta["species"] = self.labels
        # meta["likelihood"] = self.confidences
        # # used when no actual tag
        # if self.raw_tag is not None:
        #     meta["raw_tag"] = self.raw_tag
        #     meta["raw_confidence"] = self.raw_confidence
        #     meta["raw_ebird_ids"] = self.ebird_ids
        # else:
        #     meta["ebird_ids"] = self.ebird_ids

        return meta


class Signal:
    def __init__(self, start, end, freq_start, freq_end):
        self.start = start
        self.end = end
        self.freq_start = freq_start
        self.freq_end = freq_end

        self.mel_freq_start = mel_freq(freq_start)
        self.mel_freq_end = mel_freq(freq_end)
        self.results = []
        self.master_tag = None
        self.master_model = None
        self.master_below_thresh = True
        self.track_id = None
        # self.model = None
        # self.labels = None
        # self.confidences = None
        # self.raw_tag = None
        # self.raw_confidence = None

    def set_master_tag(self):
        master_tag = get_master_tag(self)
        if master_tag is None:
            return
        master_tag, model, below_thresh = master_tag
        self.master_tag = master_tag
        self.master_model = model
        self.master_below_thresh = below_thresh

    def to_array(self, decimals=1):
        a = [self.start, self.end, self.freq_start, self.freq_end]
        if decimals is not None:
            a = list(
                np.round(
                    np.array(a),
                    decimals,
                )
            )
        return a

    def copy(self):
        return Signal(self.start, self.end, self.freq_start, self.freq_end)

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

    def freq_overlap(self, other):
        return segment_overlap(
            (self.freq_start, self.freq_end),
            (other.freq_start, other.freq_end),
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

    def enlarge(self, scale, min_track_length):
        new_length = self.length * scale
        if new_length < min_track_length:
            new_length = min_track_length
        extension = (new_length - self.length) / 2
        self.start = self.start - extension
        self.end = self.end + extension
        self.start = max(self.start, 0)

        # also enlarge freq
        new_length = (self.freq_end - self.freq_start) * scale
        extension = (new_length - (self.freq_end - self.freq_start)) / 2
        self.freq_start = self.freq_start - extension
        self.freq_end = int(self.freq_end + extension)
        self.freq_start = int(max(self.freq_start, 0))

        self.mel_freq_start = mel_freq(self.freq_start)
        self.mel_freq_end = mel_freq(self.freq_end)

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
        meta["begin_s"] = self.start
        meta["end_s"] = self.end
        meta["freq_start"] = self.freq_start
        meta["freq_end"] = self.freq_end
        if self.master_tag is not None:
            meta["master_tag"] = {
                "below_thresh": self.master_below_thresh,
                "prediction": self.master_tag.get_meta(),
                "model": self.master_model,
            }
        meta["model_results"] = [r.get_meta() for r in self.results]
        if self.track_id is not None:
            meta["track_id"] = self.track_id
        return meta


from scipy.signal import butter, sosfilt, sosfreqz, freqs


def butter_bandpass(lowcut, highcut, fs, order=2):
    nyq = 0.5 * fs
    btype = "lowpass"
    freqs = []
    if lowcut > 0:
        btype = "bandpass"
        low = lowcut / nyq
        freqs.append(low)
    high = highcut / nyq
    freqs.append(high)
    sos = butter(order, freqs, analog=False, btype=btype, output="sos")
    return sos


def butter_bandpass_filter(data, lowcut, highcut, fs, order=2):
    sos = butter_bandpass(lowcut, highcut, fs, order=order)
    filtered = sosfilt(sos, data)
    return filtered
