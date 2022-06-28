#!/usr/bin/env python3

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
import logging
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time


logger = logging.getLogger('scratch_manager')
logger.setLevel(logging.INFO)

handler = logging.StreamHandler(sys.stderr)
handler.setLevel(logging.WARN)
logger.addHandler(handler)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
logger.addHandler(handler)

stats = {}
sizes = {}
total_cache_reads = 0


def goodby(signum, frame):
    logger.info('Exiting')
    log_cache_reads(None, None)
    sys.exit(0)


def log_cache_reads(signum, frame):
    logger.info(f"total cache_reads : {(total_cache_reads + sum(s.reads[-1][1] for s in stats.values())) // 1024 ** 3}GB")


def mb(s):
    return round(s / 1024 ** 2, 2)


def gb(s):
    return round(s / 1024 ** 3)


def is_synced(a, b):
    sta = os.stat(a)
    stb = os.stat(b)
    return sta.st_size == stb.st_size and sta.st_ctime == stb.st_ctime


def knapsack(values, weights, capacity):
    '''https://gist.github.com/KaiyangZhou/71a473b1561e0ea64f97d0132fe07736'''
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

    picks.sort()
    picks = [x - 1 for x in picks]  # change to 0-index

    return picks


class Stat:
    def __init__(self, window=600, reads=None):
        self.window = window

        self.reads = [] if reads is None else [(time.monotonic(), reads)]

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

        self.reads = self.reads[max(0, before_window - 1):]


def dataset_name(filename, data_dir, cache_dir):
    """Parse loopback backing file path.

    Returns a tuple containing:
      - the dataset name
      - whether the file is in the cache
      - whether the file has been deleted.
    """
    if filename.endswith(' (deleted)'):
        filename = filename[:-len(' (deleted)')]
        deleted = True
    else:
        deleted = False

    if filename.endswith('.squashfs'):
        filename = filename[:-len('.squashfs')]
    else:
        return None, None, None

    if os.path.samefile(os.path.dirname(filename), data_dir):
        return os.path.basename(filename), False, deleted
    elif os.path.samefile(os.path.dirname(filename), cache_dir):
        return os.path.basename(filename), True, deleted
    else:
        return None, None, None


def diskstats(data_dir, cache_dir):
    """Iterate over mounted images stats.

    Yields tuples containing:
      - the dataset name
      - whether is cached
      - number of bytes read since mounted
      - whether the backing file has been deleted
    """
    with open("/proc/diskstats") as f:
        for line in f:
            _, _, dev, _, _, reads, *_ = line.split()
            reads = int(reads) * 512

            if not os.path.exists(f"/sys/block/{dev}/loop/backing_file"):
                continue

            backing_file = open(f"/sys/block/{dev}/loop/backing_file").read().strip()
            dataset, cached, deleted = dataset_name(backing_file, data_dir, cache_dir)
            if dataset is None:
                continue

            yield dataset, cached, reads, deleted


def run(cmd, log_prefix):
    """Run a command in a subprocess.

    :arg cmd: the command line
    :arg log_prefix: a description of the command for logging
    :return: True if the process exited with code 0, False otherwise
    """
    r = subprocess.run(
        shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if r.returncode:
        out = r.stdout.decode().strip('\n')
        logger.error(f"{log_prefix} failed: {out}")
        return False
    else:
        logger.debug(f"{log_prefix} succeeded")
        return True


def loop(data_dir, cache_dir, mnt_dir, cache_capacity, window):
    """Main program loop"""
    global total_cache_reads, stats, sizes

    t0 = time.monotonic()

    while True:
        # Update statistics and check for deleted files
        mounted = set()
        deleted = set()

        for dataset, cached, reads, deleted_ in diskstats(data_dir, cache_dir):
            mounted.add((dataset, cached))

            if (dataset, cached) in stats:
                stats[(dataset, cached)].update(reads)
            else:
                stats[(dataset, cached)] = Stat(window, reads)

            if deleted_:
                deleted.add((dataset, cached))

        # Clean-up after externally deleted files
        for dataset, cached in deleted:
            if cached:
                logger.info(f"backing file for cached {dataset} was deleted")

                # Umount dataset
                ok = run(f"umount -l {os.path.join(mnt_dir, dataset)}", f"unmounting cached {dataset}")
                if ok:
                    mounted.remove((dataset, True))
                    s = stats.pop((dataset, True))
                    total_cache_reads += s.reads[-1][1]
                    logger.info("total cache_reads : {total_cache_reads // 1024 ** 3}GB")

            else:
                logger.info(f"{dataset} was deleted")

                if (dataset, True) in mounted:
                    # Umount cached dataset first
                    ok = run(f"umount -l {os.path.join(mnt_dir, dataset)}", f"unmounting cached {dataset}")

                    if ok:
                        mounted.remove((dataset, True))
                        stats.pop((dataset, True))
                        _, reads, _ = stats.pop((dataset, True))
                        total_cache_reads += reads
                        logger.info("total cache_reads : {total_cache_reads // 1024 ** 3}GB")

                        # Remove cached dataset file
                        if os.path.exists(os.path.join(cache_dir, dataset + ".squashfs")):
                            run(f"rm {os.path.join(cache_dir, dataset)}.squashfs", f"removing cached {dataset}")

                        # Umount dataset
                        ok = run(f"umount -l {os.path.join(mnt_dir, dataset)}", f"unmounting {dataset}")
                        if ok:
                            mounted.remove((dataset, False))
                            stats.pop((dataset, False))

                else:  # Umount dataset directly
                    ok = run(f"umount -l {os.path.join(mnt_dir, dataset)}", f"unmounting {dataset}")
                    if ok:
                        mounted.remove((dataset, False))
                        stats.pop((dataset, False))

        assert set(stats.keys()) == mounted

        # Unmount if cache file was somehow mounted without the original
        for dataset, cached in mounted.copy():
            if cached and ((dataset, False) not in mounted):
                logger.warn(f"dropping cached version of deleted {dataset}")

                ok = run(f"umount -l {os.path.join(mnt_dir, dataset)}", f"unmounting cached {dataset}")
                if ok:
                    mounted.remove((dataset, True))
                    s = stats.pop((dataset, True))
                    total_cache_reads += s.reads[-1][1]
                    logger.info("total cache_reads : {total_cache_reads // 1024 ** 3}GB")

        # Mount new datasets
        for dataset in os.listdir(data_dir):
            if not dataset.endswith(".squashfs"):
                continue

            dataset = dataset[:-len(".squashfs")]

            if (dataset, False) not in mounted:
                logger.info(f"new dataset: {dataset}")

                os.makedirs(os.path.join(mnt_dir, dataset), exist_ok=True)
                ok = run(f"mount {os.path.join(data_dir, dataset)}.squashfs {os.path.join(mnt_dir, dataset)}", f"mounting {dataset}")
                if ok:
                    mounted.add((dataset, False))
                    stats[(dataset, False)] = Stat(window)

        # Optimize mounts
        throughputs = {}
        for (d, c), s in stats.items():
            throughputs[d] = throughputs.get(d, 0) + s.throughput() + c * 1024 ** 2

        throughputs = list(throughputs.items())
        datasets, throughputs = [d for d, _ in throughputs], [t for _, t in throughputs]

        sizes = [os.stat(os.path.join(data_dir, d + ".squashfs")).st_size for d in datasets]

        to_cache = [datasets[i] for i in knapsack(throughputs, sizes, cache_capacity)]

        # unmount discarded cached datasets
        for d, c in list(stats.keys()):
            if c and d not in to_cache:
                logger.info(f"dropping {d} from cache")

                ok = run(f"umount -l {os.path.join(mnt_dir, d)}", f"unmounting {d}")
                if ok:
                    mounted.remove((d, True))
                    s = stats.pop((d, True))
                    total_cache_reads += s.reads[-1][1]
                    logger.info("total cache_reads : {total_cache_reads // 1024 ** 3}GB")

        # remove deprecated cached images (or any garbage file liying around)
        for f in os.listdir(cache_dir):
            d = f[:-len(".squashfs")]
            if (d, True) not in mounted:
                run(f"rm {os.path.join(cache_dir, f)}", f"removing cached {d}")

        # cache and mount
        for d in to_cache:
            if (d, True) in mounted:
                continue

            if stats[(d, False)].throughput() < 2 * 1024 ** 2:
                continue

            logger.info(f"adding {d} to cache ({stats[(d, False)].throughput() / 1024 ** 2:.0f}MB/s)")

            ok = run(f"rsync {os.path.join(data_dir, d )}.squashfs {os.path.join(cache_dir, d)}.squashfs", f"transfering {d}")
            if ok:
                run(f"mount {os.path.join(cache_dir, d)}.squashfs {os.path.join(mnt_dir, d)}", f"mounting cached {d}")
                mounted.add((d, True))
                stats[(d, True)] = Stat(window)

        if set(stats.keys()) != mounted:
            raise RuntimeError()

        time.sleep(max(0, t0 + 5 - time.monotonic()))
        t0 = time.monotonic()


def main():
    argparser = argparse.ArgumentParser(description="""
        This script will do the following:
        1- Monitor datasets stored as image files (squashfs) in datadir and mounted in mountdir.
        2- Move datasets images with high read thoughput to cachedir and mount them over the existing mountpoint.
        3- Unmount and drop inactive cached datasets.
""")
    argparser.add_argument('--datadir', help='directory where dataset images are stored')
    argparser.add_argument('--cachedir', help='cache directory where dataset images are mirrored')
    argparser.add_argument('--mountdir', help='mount directory where dataset images are mounted')
    argparser.add_argument('--capacity', help='cache capacity in GB, TB or %')
    argparser.add_argument('--period', type=int, default=1200, help='cache content update period')
    argparser.add_argument('--verbose', '-v', action='store_true', help='verbose logger')
    args = argparser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)
        logger.debug("loglevel set to debug")

    if args.capacity.endswith("GB"):
        args.capacity = int(args.capacity[:-2]) * 1024 ** 3
    elif args.capacity.endswith("TB"):
        args.capacity = int(args.capacity[:-2]) * 1024 ** 4
    elif args.capacity.endswith("%"):
        args.capacity = int(args.capacity[:-1]) * shutil.disk_usage(args.cachedir).total // 100
    else:
        logger.info("assuming capacity unit is GB")
        args.capacity = int(args.capacity) * 1024 ** 3

    signal.signal(signal.SIGTERM, goodby)
    signal.signal(signal.SIGUSR1, log_cache_reads)

    loop(args.datadir, args.cachedir, args.mountdir, args.capacity, args.period)


if __name__ == '__main__':
    main()
