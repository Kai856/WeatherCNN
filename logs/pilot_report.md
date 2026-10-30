# Stage B: 7-day pilot report

Pilot: 2025-07-14 00:00:00 .. 2025-07-20 23:00:00 (168 hours processed, 168 fully valid)

## Storage

- Stored inputs: 1.88 MB/hour (raw float32 would be 12.7 MB; 6.8x smaller)
- Same fields as compressed float32: 3.92 MB/hour
- Projected 3 years (26304 hours): inputs 49.4 GB + per-hour arrays 0.00 GB + static/coords 0.001 GB + provenance 0.05 GB = **49.5 GB**
- If every channel were float32: 103.0 GB
- Download volume: 43.2 MB/hour, 1.14 TB over 3 years (streamed, not kept)

Alternative encodings measured on a pilot subsample (MB/hour):

| Encoding | MB/hour | 3-year GB |
|---|---|---|
| float32 (no filter) | 3.91 | 103 |
| int16 bounds-step (no filter) | 3.80 | 100 |
| int16 bounds-step + delta | 3.13 | 82 |
| int16 source-step (no filter) | 2.46 | 65 |
| int16 source-step + delta [chosen] | 1.88 | 49 |

## Precision (int16 vs float32 6 km fields)

Rule: max abs error <= half the largest GRIB packing step seen (plus floating-point representation tolerance: 0.1% for decimal steps and 2 float32 ulps at the channel bound, since the float32 reference itself carries that error). Failing channels: none
Secondary (reported only) RMSE/std < 1e-4 fails for 46/49 channels; the GRIB source itself is quantized at a comparable level, see logs/pilot_precision.csv.

| Channel | int16 step | GRIB step (max) | max error | RMSE/std |
|---|---|---|---|---|
| t2m | 0.0625 | 0.0625 | 0.03119 | 3.7e-03 |
| d2m | 0.0625 | 0.0625 | 0.03119 | 4.0e-03 |
| rh2m | 0.1 | 0.1 | 0.05 | 2.2e-03 |
| u10 | 0.0625 | 0.0625 | 0.03123 | 7.8e-03 |
| v10 | 0.0625 | 0.0625 | 0.03124 | 6.3e-03 |
| gust | 0.0625 | 0.0625 | 0.03118 | 5.7e-03 |
| sp | 10 | 10 | 5 | 1.3e-03 |
| skt | 0.0625 | 0.0625 | 0.03122 | 3.2e-03 |
| mslp | 1 | 1 | 0.5 | 8.6e-04 |
| cape | 10 | 10 | 5 | 2.6e-03 |
| cin | 1 | 10 | 0.5 | 5.0e-03 |
| pwat | 0.125 | 0.125 | 0.06222 | 2.9e-03 |
| dswrf | 0.1 | 1 | 0.05005 | 8.0e-05 |
| apcp1h | 0.01 | 0.01 | 0.005011 | 8.8e-04 |
| t1000 | 0.0625 | 0.0625 | 0.03122 | 4.6e-03 |
| t925 | 0.0625 | 0.0625 | 0.03125 | 5.1e-03 |
| t850 | 0.0625 | 0.0625 | 0.03119 | 5.7e-03 |
| t700 | 0.0625 | 0.0625 | 0.03119 | 6.1e-03 |
| t500 | 0.0625 | 0.0625 | 0.03125 | 7.0e-03 |
| t300 | 0.0625 | 0.0625 | 0.0312 | 6.0e-03 |
| t250 | 0.0625 | 0.0625 | 0.03123 | 7.2e-03 |
| dpt1000 | 0.0625 | 0.0625 | 0.03122 | 4.0e-03 |
| dpt925 | 0.0625 | 0.0625 | 0.03122 | 3.7e-03 |
| dpt850 | 0.0625 | 0.0625 | 0.03125 | 3.4e-03 |
| dpt700 | 0.0625 | 0.0625 | 0.03122 | 2.1e-03 |
| dpt500 | 0.0625 | 0.0625 | 0.03104 | 1.9e-03 |
| dpt300 | 0.0625 | 0.0625 | 0.03104 | 2.1e-03 |
| dpt250 | 0.0625 | 0.0625 | 0.03104 | 2.0e-03 |
| u1000 | 0.0625 | 0.0625 | 0.03124 | 6.5e-03 |
| u925 | 0.0625 | 0.0625 | 0.03121 | 4.4e-03 |
| u850 | 0.0625 | 0.0625 | 0.03122 | 4.4e-03 |
| u700 | 0.0625 | 0.0625 | 0.03115 | 3.8e-03 |
| u500 | 0.0625 | 0.0625 | 0.03118 | 2.3e-03 |
| u300 | 0.0625 | 0.0625 | 0.03124 | 1.5e-03 |
| u250 | 0.0625 | 0.0625 | 0.03114 | 1.3e-03 |
| v1000 | 0.0625 | 0.0625 | 0.03104 | 5.0e-03 |
| v925 | 0.0625 | 0.0625 | 0.03124 | 3.8e-03 |
| v850 | 0.0625 | 0.0625 | 0.03118 | 4.0e-03 |
| v700 | 0.0625 | 0.0625 | 0.03125 | 4.5e-03 |
| v500 | 0.0625 | 0.0625 | 0.03123 | 3.8e-03 |
| v300 | 0.0625 | 0.0625 | 0.03117 | 2.3e-03 |
| v250 | 0.0625 | 0.0625 | 0.03118 | 1.9e-03 |
| gh1000 | 0.03125 | 0.03125 | 0.01562 | 2.8e-04 |
| gh925 | 0.03125 | 0.03125 | 0.01562 | 2.7e-04 |
| gh850 | 0.03125 | 0.03125 | 0.01562 | 2.5e-04 |
| gh700 | 0.03125 | 0.03125 | 0.01562 | 1.9e-04 |
| gh500 | 0.03125 | 0.03125 | 0.01562 | 1.3e-04 |
| gh300 | 0.03125 | 0.03125 | 0.01562 | 8.5e-05 |
| gh250 | 0.03125 | 0.03125 | 0.01562 | 7.6e-05 |

## Gate

Projected total 49.5 GB vs gate 70 GB (budget 100 GB, leaving >= 30 GB for temp files and checkpoints); precision failures: 0. **PASS**
