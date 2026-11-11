# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Launch one independent live-generation worker on one visible GPU."""

import sys

from common.shared import read_configuration
from models.live import run_worker


if __name__ == "__main__":
    question_file, results_root, shard, num_shards, deadline, revision = sys.argv[1:]
    configuration = read_configuration()
    configuration["model"]["revision"] = revision
    run_worker(question_file, results_root, configuration, int(shard),
               int(num_shards), float(deadline))
