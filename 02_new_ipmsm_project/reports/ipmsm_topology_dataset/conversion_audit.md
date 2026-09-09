# Conversion audit

1. Every 200-bit chromosome decodes stably to 100 four-state cells.
2. Adjacent pairs are MSB-first: `code = 2*bit[2i] + bit[2i+1]`.
3. Exported 10×10 axes are row=radius and column=angle.
4. MaterialPosition verifies centres [1.125, 3.375, 5.625, 7.875, 10.125, 12.375, 14.625, 16.875, 19.125, 21.375] degrees.
5. Verified physical mapping is code 0=Air, code 1=N38 PM, codes 2/3=Pure Iron.
6. The default CNN dataset collapses 2/3 and dynamically returns three one-hot channels.
7. Four-state files remain available for traceability and an ablation experiment.
8. Raw-only genes are not assigned corrected FEMM labels; doing so would create false supervision.
9. Default usable physical samples: 146,471.
10. No resize, interpolation, PNG storage, or FEMM rerun was performed.
