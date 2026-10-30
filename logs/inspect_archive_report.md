# Stage A: archive inspection

Sample hours: 2023-10-01, 2025-01-15, 2025-04-01, 2025-07-15, 2026-09-30 at [0, 18] UTC

Channels present in every sample: 51/51

Source product per channel: apcp1h=wrfsfcf01, cape=wrfsfcf00, cin=wrfsfcf00, d2m=wrfsfcf00, dpt1000=wrfsfcf00, dpt250=wrfprsf00, dpt300=wrfprsf00, dpt500=wrfsfcf00, dpt700=wrfsfcf00, dpt850=wrfsfcf00, dpt925=wrfsfcf00, dswrf=wrfsfcf00, gh1000=wrfsfcf00, gh250=wrfprsf00, gh300=wrfprsf00, gh500=wrfsfcf00, gh700=wrfsfcf00, gh850=wrfsfcf00, gh925=wrfprsf00, gust=wrfsfcf00, mslp=wrfsfcf00, pwat=wrfsfcf00, rh2m=wrfsfcf00, skt=wrfsfcf00, sp=wrfsfcf00, static:land=wrfsfcf00, static:orography=wrfsfcf00, t1000=wrfsfcf00, t250=wrfprsf00, t2m=wrfsfcf00, t300=wrfprsf00, t500=wrfsfcf00, t700=wrfsfcf00, t850=wrfsfcf00, t925=wrfsfcf00, u10=wrfsfcf00, u1000=wrfsfcf00, u250=wrfsfcf00, u300=wrfsfcf00, u500=wrfsfcf00, u700=wrfsfcf00, u850=wrfsfcf00, u925=wrfsfcf00, v10=wrfsfcf00, v1000=wrfsfcf00, v250=wrfsfcf00, v300=wrfsfcf00, v500=wrfsfcf00, v700=wrfsfcf00, v850=wrfsfcf00, v925=wrfsfcf00

All decoded valid times equal the requested valid hour: True
apcp1h accumulation hours (f01 of previous init): [1.0], step ranges: ['0-1']
APCP in f00 files: step ranges ['0'], max value 0.0 (zero-length accumulation, not used)
DSWRF f00 max per sample: 2023-10-01 00Z=416, 2023-10-01 18Z=972, 2025-01-15 00Z=355, 2025-01-15 18Z=803, 2025-04-01 00Z=549, 2025-04-01 18Z=1074, 2025-07-15 00Z=692, 2025-07-15 18Z=1063, 2026-09-30 00Z=421, 2026-09-30 18Z=938
Bytes downloaded per valid hour (49 channels): 40.5-45.4 MB
Mean full file sizes: wrfprsf00=404 MB, wrfsfcf00=147 MB, wrfsfcf01=156 MB
Max GRIB packing step per channel: apcp1h=0.01, cape=10, cin=10, d2m=0.0625, dpt1000=0.0625, dpt250=0.0625, dpt300=0.0625, dpt500=0.0625, dpt700=0.0625, dpt850=0.0625, dpt925=0.0625, dswrf=1, gh1000=0.03125, gh250=0.03125, gh300=0.03125, gh500=0.03125, gh700=0.03125, gh850=0.03125, gh925=0.03125, gust=0.0625, mslp=1, pwat=0.125, rh2m=0.1, skt=0.0625, sp=10, static:land=1, static:orography=0.03125, t1000=0.0625, t250=0.0625, t2m=0.0625, t300=0.0625, t500=0.0625, t700=0.0625, t850=0.0625, t925=0.0625, u10=0.0625, u1000=0.0625, u250=0.0625, u300=0.0625, u500=0.0625, u700=0.0625, u850=0.0625, u925=0.0625, v10=0.0625, v1000=0.0625, v250=0.0625, v300=0.0625, v500=0.0625, v700=0.0625, v850=0.0625, v925=0.0625

Crop window (native): rows 536-981, cols 1217-1798 (446 x 582)
6 km grid: 223 x 291 (lat 34.72..49.98, lon -86.55..-62.09); 43553/64893 cell centers inside bbox
Fraction of bbox edge outside HRRR domain: 0.009
Projection vs GRIB lat/lon max error: 0.0000 km
Tufts cell: j=777 i=1613 (42.4061, -71.1206), 0.19 km away; great-circle check agrees: True
Wind rotation at Tufts: projection -16.422 deg, cone formula 16.422 deg (same magnitude; the two use opposite sign conventions)


Availability scan: 26304 valid hours, 26304 complete, 0 with at least one required file missing (see metadata/missing_timestamps.csv)
  wrfsfcf00_init_bytes: mean 141 MB, missing 0
  wrfprsf00_init_bytes: mean 394 MB, missing 0
  wrfsfcf01_init_prev_bytes: mean 150 MB, missing 0
