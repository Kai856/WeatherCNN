# Validation report

Store: /Users/kaikaplinsky/Desktop/DLProjects/hrrr_cnn/data/processed/hrrr_ne_6km.zarr
Processed hours: 168 / 26304

| Check | Result | Detail |
|---|---|---|
| Arrays have expected spatial dimensions | PASS | 223 x 291; mismatches: none |
| All variables use the same native grid | PASS | 8232 decoded messages checked against grid_metadata.json; 0 mismatches |
| Timestamps aligned | PASS | store time axis == hourly period: True; GRIB valid time == record time: True; init + lead == valid: True |
| Provenance covers every channel of every processed hour | PASS | 168 hours with provenance / 168 processed |
| Target offset is exactly +24 h | PASS | 144 samples |
| Precipitation interval is exactly 1 h ending at the valid time | PASS | 168 APCP messages: accum [1.0] h, step ['0-1'] |
| No input sample uses data from after its valid time | PASS | inputs: init <= valid and GRIB valid == t for all messages (0 violations); targets come from records valid at t+24 (precip from init t+23) |
| No duplicated timestamps | PASS | time axis 0, provenance 0, samples 0 |
| Missing values quantified | PASS | 168 processed hours, 0 not fully valid (0.00%); worst channels: t2m 0.00%, d2m 0.00%, rh2m 0.00%; unprocessed hours in period: 26136 |
| Temperature, humidity, wind, precipitation within plausible ranges | PASS | targets: t2m [287.54, 309.16]; rh2m [28.30, 93.10]; u10 [-2.94, 5.90]; v10 [-5.34, 5.70]; gust [0.76, 15.02]; precip_1h [0.00, 0.12]; rain_binary [0.00, 0.00]; rain_binary consistent: True; inputs in 48 sampled hours within channel bounds and NaN-free: True [] |
| Nearest Tufts grid cell is correct | PASS | native (j=777, i=1613) at (42.4061, -71.1206), 0.192 km from the requested point (brute-force search over 11x11 neighborhood) |
| Train, validation, and test periods do not overlap | PASS | period bounds ordered and disjoint: True; every input and target in its sample's split: True; sample sets disjoint: True; counts {'train': 144} |
| Output stays within the 100 GB budget | PASS | data root currently 1.00 GB; projected 3-year total from pilot 49.5 GB |

Overall: **PASS**
