# CNN data audit

- Samples after physical deduplication and conflict quarantine: 146,471.
- Input shape on disk: `[N,10,10]`; dynamic model input: `[B,3,10,10]`.
- Input classes: Air, N38 permanent magnet, Pure Iron.
- Missing or non-finite targets: 0.
- Duplicate physical topologies in the training catalog: 0.
- Tavg min/median/mean/max: 0.000102 / 2.257438 / 2.044681 / 3.782920.
- DeltaT min/median/mean/max: 0.060751 / 0.776579 / 0.786798 / 4.381166.
- Material cells Air/PM/Iron: 1,861,395 / 4,026,291 / 8,759,414.
- Repair-related topology groups: 145,741; maximum group size: 4.
- Performance stratification uses seven Tavg bands and DeltaT quartiles inside each band.
- Figures include target histograms, class counts, repair-group sizes, 20 topology examples and a diagnostic mirror expansion.
