# Stage 0 — Storage preflight

## Purpose

The raw and processed archives live on the network-mounted
`/mnt/data-r2` filesystem. A directory can exist while the CIFS mount is
disconnected, redirected to the local root filesystem, read-only, full, or
returning stale/corrupt data. This stage proves basic storage integrity before
any expensive or mutating process starts.

## Method

For every distinct configured directory—`LOCAL_DATA_DIR`, `RAW_DATA_DIR`,
and `PROCESSED_DATA_DIR`—the probe:

1. Parses `/proc/self/mountinfo` and chooses the longest mount point containing
   the configured path.
2. Rejects a path below `/mnt` if its nearest mount is only `/`; this catches
   a missing network mount before accidentally writing to the local disk.
3. Reads filesystem capacity with `statvfs` and requires at least
   `--minimum-free-gib` (5 GiB by default).
4. Generates 1 MiB of cryptographically random bytes.
5. Writes the payload, flushes userspace buffers, and calls `fsync`.
6. Atomically renames the temporary file.
7. Reads it back and verifies

   \[
   \operatorname{SHA256}(x_{written}) =
   \operatorname{SHA256}(x_{read}).
   \]

8. Removes both possible probe filenames in a `finally` block.

The report includes mount point, filesystem type, free GiB, end-to-end latency,
write status, checksum status, and any exception. A probe is healthy only when
the mount is identified, the durable write succeeds, the checksum matches, and
cleanup reports no error.

## Resources and side effects

The check writes only a uniquely named 1 MiB temporary file in each configured
directory and immediately deletes it. It requires no GPU and negligible RAM.
Network latency is measured but not used as a failure threshold because CIFS
latency varies; it remains an operational diagnostic.

## Output and next-stage contract

The command is:

```bash
python -m src.utils.storage --minimum-free-gib 5
```

When orchestrated, the result is written to
`outputs/v2_clean/<target>/validation/storage.json`. A failure terminates the
pipeline before acquisition or data access. A success proves storage was usable
at that instant; it cannot guarantee that a network connection will remain
available for an entire multi-hour run, so downstream I/O errors still remain
fatal and logged.
