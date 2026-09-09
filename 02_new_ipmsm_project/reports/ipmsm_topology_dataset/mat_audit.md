# workspace_600.mat audit

- Historical records: 302,904 = 601 states × 504 individuals.
- Array order: MATLAB `[individual, bit, state]`; exported order is state-major.
- `population_noChange_all` is the pre-repair gene; `population_all` is the corrected FEMM input.
- Final `population == population_all[:,:,600]`; scalar final arrays equal history column 600.
- `Material` decodes from pre-repair genes; `Material_change` decodes from corrected genes.
- VolumePM mismatch: 445 records, all confined to state 0; states 1..600 align exactly.
- Historical MAT contains scalar Tavg/DeltaT, not angle-by-angle torque curves.
