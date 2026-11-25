# GeneWeaver: CPU vs Dask Benchmark

## Configuration
- Contigs: 2
- Total genome size: 500,000 bp
- Chunk size: 10,000 bp
- Dask workers: 2
- Repeated runs: 3
- Execution environment: Local machine

## Results

| Measurement | CPU | Dask |
|---|---:|---:|
| Run 1 | 0.0333s | 0.2640s |
| Run 2 | 0.0053s | 0.1996s |
| Run 3 | 0.0053s | 0.1234s |
| Median | 0.0053s | 0.1996s |

- CPU/Dask time ratio: 0.03x
- Output verification: Passed for every run
- Note: Dask uses local workers; this is not a multi-machine benchmark.
- Dask startup and scheduling overhead are included in its timings.
