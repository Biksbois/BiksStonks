import seqlog
import logging


def initialize_logger():
    seqlog.log_to_seq(
        server_url="http://localhost:5341/",
        level=logging.NOTSET,
        batch_size=10,
        auto_flush_timeout=10,  # seconds
        override_root_logger=True,
    )
