SOCKET_NAME = "/var/run/classifier"
import json
import sys
import socket
import argparse
import logging
import time
from classifyservice import ClassifyJob
from logs import init_logging
from pathlib import Path


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--service_socket",
        default="/etc/cacophony/audio-classifier",
        help="Socket name",
    )

    parser.add_argument(
        "--ready",
        action="store_true",
        default=False,
        help="Check if classify service is up and running",
    )

    parser.add_argument(
        "--analyse-tracks",
        action="store_true",
        default=False,
        help="Classify human made tracks marked with classify flag, in metadata file",
    )

    parser.add_argument(
        "source",
        help="an audio file to classify",
    )

    parser.add_argument(
        "-o",
        "--meta-to-stdout",
        action="store_true",
        help="Print metadata to stdout instead of saving to file.",
    )
    args = parser.parse_args()
    return args


def test_socket(sock, address):
    try:
        sock.connect(address)
        logging.info("Classify service is ready")
        return True
    except:
        logging.error("Classify service is loading", exc_info=True)
        return False


def main():
    args = parse_args()
    # init_logging()
    logging.info("Classifying %s", args.source)
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    results = ""
    if args.ready:
        return test_socket(sock, args.service_socket)
    try:
        count = 0
        retries = 1
        while True:
            try:
                sock.connect(args.service_socket)
                logging.info("Connected to %s ", args.service_socket)
                break
            except Exception as ex:
                if count >= retries:
                    raise ex
                count += 1
                logging.warning(
                    "Could not connect to %s retrying in 10s error was %s",
                    args.service_socket,
                    ex,
                )
                time.sleep(10)

        data = ClassifyJob(file=args.source, analyse_tracks=args.analyse_tracks)
        sock.send(json.dumps(data.as_dict()).encode())

        results = read_all(sock).decode()
        meta_data = json.loads(str(results))
        if "error" in meta_data:
            logging.error("Error classifying %s %s", args.source, meta_data["error"])
            raise Exception(meta_data["error"])

    except socket.error as msg:
        print(msg)
        sys.exit(1)
    finally:
        # Clean up the connection
        sock.close()
    if args.meta_to_stdout:
        print(str(results))
    else:
        audio_file = Path(args.source)
        metadata_file = audio_file.with_suffix(".txt")
        logging.info("Writing metadata to %s", metadata_file)

        if metadata_file.exists():
            with metadata_file.open("r") as f:
                metadata = json.load(f)
        else:
            metadata = {}
        metadata["analysis_result"] = meta_data["analysis_result"]
        with metadata_file.open("w") as f:
            json.dump(metadata, f, sort_keys=True, indent=4)


def read_all(socket):
    size = 4096
    data = bytearray()

    while True:
        packet = socket.recv(size)
        if packet:
            data.extend(packet)
        else:
            break
    return data


if __name__ == "__main__":
    main()
