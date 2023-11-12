# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import asyncio
import logging
import subprocess
from typing import TypeVar

from submitit import Job

logger = logging.getLogger("modelling_utils")

JT = TypeVar("JT")


# useful until we get submitit#1622 lands
async def awaitable_job(job: Job[JT], poll_s: int = 1) -> JT:
    while not job.done():
        await asyncio.sleep(poll_s)
    return job.result()


async def awaitable_cleanup_job(job: Job[JT], poll_s: int = 1) -> JT:
    result = await awaitable_job(job)
    # if job is scheduled with cancel_at_deletion, `del job` will cancel it
    del job
    return result


def execute_in_shell(command, shell=True, dry_run=False, quiet=True):
    """Execute commands in the shell

    Args:
        command ([type]): str or list commands (type needs to correspond to the value of the shell)
        shell (bool, optional): controls the command type (True: str, False: list). Defaults to True.
        dry_run (bool, optional): print out commands without real execution. Defaults to False.
        quiet (bool, optional): controls whether to print information. Defaults to True.
    """
    if dry_run:
        if not quiet:
            logger.info(f"dry run command: {command}")
    else:
        with subprocess.Popen(command, stdout=subprocess.PIPE, shell=shell) as proc:
            if not quiet:
                logger.info(proc.stdout.read().decode("utf-8"))
