# Test data for the integration tests

The `Integration` workflow (`.github/workflows/integration.yml`) runs MRIQC end to end
on small datasets assembled from public S3 buckets. Nothing is re-hosted: each manifest
pins its files by S3 object version and MD5, and the workflow keeps the assembled
dataset in the Actions cache, fetching it again whenever the cache is evicted.

| Manifest | Contents | Licence |
| --- | --- | --- |
| `dwi.json` | Two Healthy Brain Network DWI runs (Siemens TrioTim and Prisma), each trimmed to its b=0 and 16 directions per shell (b = 1000 and 2000) | CC BY 4.0 |
| `flair.json` | Three 2D FLAIR runs with a reduced field of view, from OpenNeuro ds007486 | CC0 |

## Building a dataset locally

```bash
uv run --no-project --with nibabel --with numpy \
    python .github/testdata/fetch.py .github/testdata/flair.json /tmp/testdata/flair
```

`fetch.py` downloads every file, checks its MD5, cuts the DWI runs down to the volumes
listed under `subsets`, and writes a minimal BIDS dataset with a README that records
the sources and their licences. Running it twice gives byte-identical files.

## Adding or changing a dataset

1. List the files in a manifest, with the URL, the S3 version ID (`x-amz-version-id`
   of a `HEAD` request) and the MD5. For objects uploaded in one part, the ETag is the MD5.
2. For a DWI run, compute the volumes to keep and paste them into `subsets`:

   ```bash
   uv run --no-project --with nibabel --with numpy \
       python .github/testdata/fetch.py .github/testdata/dwi.json --select <path-without-extension> 16
   ```

3. Update `.github/integration/outputs-<modality>.txt`, and the reference IQMs in
   `mriqc/data/testdata/group_<modality>.tsv`, from a green run of the workflow.

The cache key hashes the manifest and `fetch.py`, so any change to either fetches the
data again.
