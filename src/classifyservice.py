import threading
import socket
import logging
import os
import argparse
import json
import traceback

from logs import init_logging


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--service_socket",
        default="/etc/cacophony/audio-classifier",
        help="Socket name",
    )

    parser.add_argument(
        "--bird-model",
        type=str,
        action="append",
        help="Path to bird model",
    )

    args = parser.parse_args()
    if args.bird_model is None or len(args.bird_model) == 0:
        args.bird_model = [
            "/models/pre-model/audioModel.keras",
            "/models/bird-model-v2m/audioModel.keras",
        ]

    return args


# Links to socket and continuously waits for 1 connection
def main():
    init_logging()
    args = parse_args()

    service = ClassifyService(args.service_socket, args.bird_model)
    try:
        service.run()
    except KeyboardInterrupt:
        logging.info("Keyboard interupt closing down")
    except PermissionError:
        logging.error("Error with permissions", exc_info=True)
    except:
        logging.error("Error with service restarting", exc_info=True)


class ClassifyService:
    def __init__(self, socket_path, model_files):
        from model import Model

        self.socket_path = socket_path
        self.models = []
        for model_file in model_files:
            model = Model(model_file)
            self.models.append(model)

    def run(self):
        logging.info("Running on %s", self.socket_path)
        try:
            os.unlink(self.socket_path)
        except OSError:
            if os.path.exists(self.socket_path):
                raise

        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.bind(self.socket_path)
        sock.listen(1)

        while True:
            logging.info("waiting for jobs")
            connection, client_address = sock.accept()
            t = threading.Thread(
                target=classify_job,
                args=(self.models, connection, client_address),
            )
            t.start()


def classify_job(models, clientsocket, addr):
    from analyse import species_identify, RoundFloats

    try:
        job = read_all(clientsocket).decode()
        logging.info("Received job %s", job)
        if len(job) == 0:
            logging.error("Client disconnected")
            return
        args = json.loads(job)
        try:
            job = ClassifyJob.from_dict(**args)
        except Exception as e:
            logging.error("Could not parse job", exc_info=True)
            clientsocket.sendall(
                json.dumps({"error": f"Could not parse job {e}"}).encode()
            )
            return
        logging.info("Classifying %s", job)
        metadata = {}
        metadata["analysis_result"] = species_identify(
            job.file, models, job.analyse_tracks
        )

        response = clientsocket.sendall(json.dumps(metadata, cls=RoundFloats).encode())
        if response:
            logging.error("Error sending data to socket %s", response)
    except BrokenPipeError:
        logging.error(
            "Error sending metadata for job %s too %s",
            job.file,
            addr,
            exc_info=True,
        )
    except Exception as e:
        logging.error("Error classifying job %s", job.file, exc_info=True)
        clientsocket.sendall(
            json.dumps(
                {"error": f"Error classifying {traceback.format_exc()}"}
            ).encode()
        )
        raise e
    finally:
        try:
            clientsocket.close()
        except:
            pass


def read_all(socket):
    size = 4096
    data = bytearray()

    while size > 0:
        packet = socket.recv(size)
        data.extend(packet)
        if len(packet) < size:
            break
    return data


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


import attr


@attr.s
class ClassifyJob:
    file = attr.ib()
    analyse_tracks = attr.ib()

    def as_dict(self):
        return attr.asdict(self)

    @classmethod
    def from_dict(cls, file, analyse_tracks):
        return cls(file=file, analyse_tracks=analyse_tracks)


if __name__ == "__main__":
    main()
