import os
import shutil

import jax
from absl import app, flags, logging
from clu import platform
from ml_collections import config_flags

import train


FLAGS = flags.FLAGS

flags.DEFINE_string("workdir", None, "Directory to store model data.")
config_flags.DEFINE_config_file(
    "config",
    "configs/default.py",
    "File path to the training hyperparameter configuration.",
    lock_config=True,
)
flags.mark_flags_as_required(["workdir"])


def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    logging.info("JAX process: %d / %d", jax.process_index(), jax.process_count())
    logging.info("JAX local devices: %r", jax.local_devices())

    # Add a note so that we can tell which task is which JAX host.
    # (Depending on the platform task 0 is not guaranteed to be host 0)
    platform.work_unit().set_task_status(
        f"process_index: {jax.process_index()}, process_count: {jax.process_count()}"
    )
    platform.work_unit().create_artifact(
        platform.ArtifactType.DIRECTORY, FLAGS.workdir, "workdir"
    )

    if os.path.exists(FLAGS.workdir):
        logging.info(f"Workdir {FLAGS.workdir} already exists. Deleting...")
        shutil.rmtree(FLAGS.workdir)
    os.makedirs(FLAGS.workdir, exist_ok=True)

    train.train_and_evaluate(FLAGS.config, FLAGS.workdir)


if __name__ == "__main__":
    jax.config.config_with_absl()
    app.run(main)
