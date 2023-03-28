# (C) Copyright 2022 CEA LIST. All Rights Reserved.
# Contributor(s): Nicolas Granger <nicolas.granger@cea.fr>
#
# This software is governed by the CeCILL-C license under French law and
# abiding by the rules of distribution of free software.  You can  use,
# modify and/ or redistribute the software under the terms of the CeCILL-C
# license as circulated by CEA, CNRS and INRIA at the following URL
# "http://www.cecill.info".
# As a counterpart to the access to the source code and  rights to copy,
# modify and redistribute granted by the license, users are provided only
# with a limited warranty  and the software's author,  the holder of the
# economic rights,  and the successive licensors  have only  limited
# liability.
# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL-C license and that you accept its terms.

import argparse
import asyncio
import collections
import configparser
import logging
import os
import shutil
import signal
import sys
import time
from os import path

# Globals
logger = logging.getLogger("scratch_manager")
stats = collections.OrderedDict()
total_cache_reads = 0


class Stat:
    def __init__(self, window=600, reads=None):
        self.window = window
        self.reads = [] if reads is None else [(time.monotonic(), reads)]

    def __repr__(self) -> str:
        return f"Stat({self.throughput()}B/s)"

    def throughput(self):
        if len(self.reads) == 0:
            return 0

        t1, r1 = self.reads[-1]

        t0, r0 = self.reads[0]
        for t, r in self.reads:
            if t < t1 - self.window:
                t0, r0 = t, r

        # divisor lower bound to discard inaccurate initial readings
        return (r1 - r0) / max(t1 - t0, self.window / 2)

    def update(self, reads):
        t1 = time.monotonic()

        self.reads.append((t1, reads))

        before_window = 0
        for t, _ in self.reads:
            if t < t1 - self.window:
                before_window += 1

        self.reads = self.reads[max(0, before_window - 1) :]


def goodby(signum, frame):
    logger.info("Exiting")
    log_cache_reads(None, None)
    sys.exit(0)


def log_cache_reads(signum, frame):
    total = total_cache_reads
    for _, sc in stats.values():
        if len(sc.reads) > 0:
            total += sc.reads[-1][1]

    logger.info(f"total cache_reads : {total // 1024 ** 3}GB")


def knapsack(values, weights, capacity):
    """https://gist.github.com/KaiyangZhou/71a473b1561e0ea64f97d0132fe07736"""
    n_items = len(values)

    # Normalize
    weights = [w * 1000 // capacity for w in weights]
    capacity = 1000

    table = [[0 for _ in range(capacity + 1)] for _ in range(n_items + 1)]
    keep = [[0 for _ in range(capacity + 1)] for _ in range(n_items + 1)]

    for i in range(1, n_items + 1):
        for w in range(0, capacity + 1):
            wi = weights[i - 1]  # weight of current item
            vi = values[i - 1]  # value of current item
            if (wi <= w) and (vi + table[i - 1][w - wi] > table[i - 1][w]):
                table[i][w] = vi + table[i - 1][w - wi]
                keep[i][w] = 1
            else:
                table[i][w] = table[i - 1][w]

    picks = []
    K = capacity

    for i in range(n_items, 0, -1):
        if keep[i][K] == 1:
            picks.append(i)
            K -= weights[i - 1]

    picks = [x - 1 for x in picks]  # change to 0-index

    return picks


def diskstats():
    """Return (backing_file, deleted, reads) for each loop device."""

    out = []

    with open("/proc/diskstats") as f:
        for line in f:
            _, _, dev, _, _, reads, *_ = line.split()

            try:
                sector_size = (
                    open(f"/sys/block/{dev}/queue/hw_sector_size").read().strip()
                )
            except OSError:
                continue

            reads = int(reads) * int(sector_size)

            try:
                backing_file = (
                    open(f"/sys/block/{dev}/loop/backing_file").read().strip()
                )
                if backing_file.endswith(" (deleted)"):
                    backing_file = backing_file[:-10]
                    deleted = True
                else:
                    deleted = False
            except OSError:
                continue

            out.append((backing_file, deleted, reads))

    return out


def mounts(filter_mountpoint=None):
    """Return (backing_file, deleted, mountpoint) for each mounted loopback file."""
    out = []

    with open("/proc/mounts") as f:
        for line in f:
            dev, mountpoint, *_ = line.split()

            if filter_mountpoint is not None and mountpoint != filter_mountpoint:
                continue

            dev = path.basename(dev)

            try:
                backing_file = (
                    open(f"/sys/block/{dev}/loop/backing_file").read().strip()
                )
                if backing_file.endswith(" (deleted)"):
                    backing_file = backing_file[:-10]
                    deleted = True
                else:
                    deleted = False
            except OSError:
                continue

            out.append((backing_file, deleted, mountpoint))

    return out


async def cmd(command, *args, nolog=False):
    if not nolog:
        logger.debug(f"> {command} {' '.join(args)}")

    proc = await asyncio.create_subprocess_exec(
        command, *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT
    )
    stdout = (await proc.stdout.read()).decode()
    code = await proc.wait()
    if code != 0:
        raise RuntimeError(f"command failed: {command} {' '.join(args)}: {stdout}")

    return stdout


async def update_dataset(dataset, data, cache, mnt, should_cache):
    global total_cache_reads, stats

    data_file = path.abspath(path.join(data, dataset))
    cache_file = path.abspath(path.join(cache, dataset))
    mountpoint = path.abspath(path.join(mnt, dataset.split(".")[0]))

    # rmed data file
    if not path.exists(data_file):
        logger.info(f"removing {dataset}")

        # umount all
        for _ in [m for _, _, m in mounts() if m == mountpoint]:
            await cmd("umount", "-l", mountpoint)

        # rm cache if it exists
        try:
            os.remove(cache_file)
        except OSError:
            if path.exists(cache_file):
                raise

        return False

    # create mount point
    if not path.exists(mountpoint):
        await cmd("mkdir", mountpoint)

    # remove stale cache file
    if (
        should_cache
        and (
            await cmd(
                "rsync",
                "-an",
                "--out-format='%f'",
                data_file,
                cache_file,
                nolog=True,
            )
        )
        != ""
        and path.exists(cache_file)
    ):
        logger.info(f"removing stale cached {dataset}")
        os.remove(cache_file)
        should_cache = False  # abandon caching for this update

    # drop from cache
    if not should_cache and path.exists(cache_file):
        logger.info(f"dropping {dataset} from cache")
        try:
            os.remove(cache_file)
        except OSError:
            if path.exists(cache_file):
                raise

    # clean up spurious or unordered mounts
    mnt_list = mounts(mountpoint)
    if len(mnt_list) > 0 and mnt_list[0] != (data_file, False, mountpoint):
        logger.warning(f"cleaning mount for {dataset}")
        while len(mounts(mountpoint)) > 0:
            await cmd("umount", "-l", mountpoint)

        stats[dataset][0].reads.clear()

    mnt_list = mounts(mountpoint)
    if (
        should_cache
        and len(mnt_list) > 1
        and mnt_list[1] != (cache_file, False, mountpoint)
    ):
        logger.warning(f"cleaning mount for cached {dataset}")
        while len(mounts(mountpoint)) > 1:
            await cmd("umount", "-l", mountpoint)

        total_cache_reads += stats[dataset][1].reads[-1][1]
        stats[dataset][1].reads.clear()

    mnt_list = mounts(mountpoint)
    if not should_cache and len(mnt_list) > 1:
        logger.warning(f"cleaning mount for {dataset}")
        while len(mounts(mountpoint)) > 1:
            await cmd("umount", "-l", mountpoint)

    # mount data file
    if (data_file, False, mountpoint) not in mounts():
        logger.info(f"mounting {dataset}")
        await cmd("mount", data_file, mountpoint)

    # sync and mount cache file
    if should_cache:
        if not path.exists(cache_file):
            logger.info(f"caching {dataset}")
            await cmd("rsync", "-a", data_file, cache_file)

        if (cache_file, False, mountpoint) not in mounts():
            logger.info(f"mounting cached {dataset}")
            await cmd("mount", cache_file, mountpoint)

    return True


async def loop(data_dir, cache_dir, mnt_dir, cache_capacity, window):
    """Main program loop"""
    global total_cache_reads, stats
    data_dir = path.abspath(data_dir)
    cache_dir = path.abspath(cache_dir)
    mnt_dir = path.abspath(mnt_dir)
    blacklist = set()

    while True:
        t0 = time.monotonic()

        cached = set()

        # list datasets
        datasets = [d for d in os.listdir(data_dir) if d not in blacklist]

        for d in datasets:
            if d not in stats:
                logger.info(f"new dataset: {d}")
                stats[d] = Stat(window=window), Stat(window=window)

        # update stats
        for backing_file, _, reads in diskstats():
            prefix = path.dirname(backing_file)
            d = path.basename(backing_file)

            if d in datasets and prefix == data_dir:
                stats[d][0].update(reads)

            elif d in datasets and prefix == cache_dir:
                stats[d][1].update(reads)
                cached.add(d)

        # clean up spurious cached files
        for f in os.listdir(cache_dir):
            if f not in datasets:
                try:
                    await cmd("rm", "-rf", path.join(cache_dir, f))
                except RuntimeError as e:
                    if path.exists(path.join(cache_dir, f)):
                        logger.error(f"failed to remove spurious cache files: {e}")
                        logger.error("exiting")
                        sys.exit(1)

        # clean up left-over mounts
        mount_names = {path.join(mnt_dir, d.split(".")[0]) for d in datasets}
        for backing_file, deleted, mountpoint in mounts()[::-1]:
            if (
                path.commonprefix([mountpoint, mnt_dir]) == mnt_dir
                and mountpoint not in mount_names
            ):
                try:
                    logger.info(f"cleaning up leftover mount {mountpoint}")
                    await cmd("umount", "-l", mountpoint)
                except RuntimeError as e:
                    if (backing_file, deleted, mountpoint) in mounts():
                        logger.error(f"failed to remove spurious mount: {e}")
                        logger.error("exiting")
                        sys.exit(1)

        # update cache list
        throughputs = [s[0].throughput() + s[1].throughput() for s in stats.values()]

        try:
            sizes = [
                os.stat(path.join(data_dir, d)).st_size if d in d in datasets else 0
                for d in stats.keys()
            ]
        except OSError:  # dataset deleted since listdir?
            await asyncio.sleep(max(0, 5 - time.monotonic() + t0))
            continue

        cached = os.listdir(cache_dir)
        for i, d in enumerate(stats.keys()):
            if d in cached:
                throughputs[i] += 1024**2  # favor already cached dataset

        to_cache = knapsack(throughputs, sizes, cache_capacity)
        to_cache = [i in to_cache for i in range(len(datasets))]

        # update datasets
        update_coroutines = [
            update_dataset(d, data_dir, cache_dir, mnt_dir, c)
            for d, c in zip(stats.keys(), to_cache)
        ]

        updates = await asyncio.gather(*update_coroutines, return_exceptions=True)
        for d, u in zip(list(stats.keys()), updates):
            if isinstance(u, Exception):
                blacklist.add(d)
                logger.error(f"{u}")
                logger.error(f"blacklisting {d}")
                _, sc = stats.pop(d)
                if len(sc.reads) > 0:
                    total_cache_reads += sc.reads[-1][1]
            elif not u:
                logger.info(f"removed {d}")
                _, sc = stats.pop(d)
                if len(sc.reads) > 0:
                    total_cache_reads += sc.reads[-1][1]

        # pause before looping
        await asyncio.sleep(max(0, 5 - time.monotonic() + t0))


def run_daemon():
    argparser = argparse.ArgumentParser(
        description="Mount, monitor and cache datasets stored as disk images."
    )
    argparser.add_argument(
        "--config", "-c",
        default="/etc/scratch_manager.conf",
        help="configuration file"
    )
    argparser.add_argument(
        "--datadir", help="directory where dataset images are stored"
    )
    argparser.add_argument(
        "--cachedir",
        help="cache directory where dataset images are mirrored",
    )
    argparser.add_argument(
        "--mountdir",
        help="mount directory where dataset images are mounted",
    )
    argparser.add_argument("--capacity", help="cache capacity in MB, GB, TB or %")
    argparser.add_argument(
        "--period", type=int, default=1200, help="cache content update period"
    )
    argparser.add_argument(
        "--verbose", "-v", action="store_true", help="verbose logger"
    )
    args = argparser.parse_args()

    # Logging
    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(logging.WARNING)
    logger.addHandler(handler)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    handler.addFilter(lambda record: record.levelno <= logging.INFO)
    logger.addHandler(handler)

    if args.verbose:
        logger.setLevel(logging.DEBUG)
        logger.debug("loglevel set to debug")

    # Config
    if args.config is not None:
        if (
            args.datadir is not None
            or args.cachedir is not None
            or args.mountdir is not None
            or args.capacity is not None
        ):
            logger.error("arguments cannot be used when using config file")

        config = configparser.ConfigParser(interpolation=None)
        try:
            with open(args.config) as f:
                config.read_string("[config]\n" + f.read())
        except FileNotFoundError:
            logger.error(f"config file {args.config} does not exist")
            sys.exit(1)

        try:
            datadir = config.get("config", "datadir")
            cachedir = config.get("config", "cachedir")
            mountdir = config.get("config", "mountdir")
            capacity = config.get("config", "capacity")
            period = config.getint("config", "period", fallback=1200)
        except KeyError:
            logger.error("missing config value")
            sys.exit(1)

    else:
        if (
            args.datadir is None
            or args.cachedir is None
            or args.mountdir is None
            or args.capacity is None
        ):
            logger.error("either --config or other arguments must be provided")
            argparser.error()

        datadir = args.datadir
        cachedir = args.cachedir
        mountdir = args.mountdir
        capacity = args.capacity
        period = args.period

    if capacity.lower().endswith("mb"):
        capacity = int(capacity[:-2]) * 1024**2
    elif capacity.lower().endswith("gb"):
        capacity = int(capacity[:-2]) * 1024**3
    elif capacity.lower().endswith("tb"):
        capacity = int(capacity[:-2]) * 1024**4
    elif capacity.endswith("%"):
        capacity = int(capacity[:-1]) * shutil.disk_usage(cachedir).total // 100
    else:
        logger.error("failed to parse capacity")
        argparser.error()

    signal.signal(signal.SIGTERM, goodby)
    signal.signal(signal.SIGUSR1, log_cache_reads)

    try:
        sys.exit(asyncio.run(loop(datadir, cachedir, mountdir, capacity, period)))
    except KeyboardInterrupt:
        pass
