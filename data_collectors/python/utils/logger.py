# import seqlog
# import logging
import logging
import sys


# def initialize_logger():
#     seqlog.log_to_seq(
#         server_url="http://localhost:5341/",
#         level=logging.NOTSET,
#         batch_size=10,
#         auto_flush_timeout=10,  # seconds
#         override_root_logger=True,
#     )


class OwnLogger:
    def __init__(self, is_verbose=False):
        if is_verbose:
            self.log_level = logging.DEBUG
        else:
            self.log_level = logging.INFO

        log_format = logging.Formatter("[%(asctime)s] [%(levelname)s] - %(message)s")
        self.log = logging.getLogger(__name__)
        self.log.setLevel(self.log_level)

        # writing to stdout
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(self.log_level)
        handler.setFormatter(log_format)
        self.log.addHandler(handler)
