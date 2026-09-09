# Deduplication and label-conflict report

- Raw four-state unique: 170,251; conflicts: 334.
- Corrected four-state unique: 159,921; conflicts: 3,754.
- Raw/corrected exact union: 187,775; intersection: 142,397.
- Corrected three-class physical unique: 150,422; conflicts: 3,951.
- Clean supervised physical dataset: 146,471.
- Conflict tolerance: absolute Tavg and DeltaT range must each be <= 1e-6.
- Fitvalue disagreement is reported but does not quarantine a physical label.
- Repeated consistent labels are represented by their group median.
