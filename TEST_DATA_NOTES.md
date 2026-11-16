# Test dataset verification notes

Archive: `2026-0822_473nm_optotest_cosine.zip` (uploaded separately; excluded from public GitHub).

Verified from its archive and metadata:

- 44 archive entries; ~326.7 MB uncompressed.
- Session path: `2026-0822_473nm_optotest_cosine/2026-08-22_02-27-46/Record Node 112/experiment1/recording1/`.
- Neuropix-PXI-104.ProbeA: 384 channels, 30,000 Hz, 384,656 frames, 12.821867 s, ~295.4 MB `continuous.dat`.
- NI-DAQmx-106.PCIe-6259: 40,000 Hz, 513,024 frames, 12.825600 s.
- Both streams include real `timestamps.npy` and `sample_numbers.npy` arrays; sample numbers increase by exactly 1 across each stream and timestamps have no backward steps or NaNs.
- `structure.oebin`, TTL events, and recording metadata are present.
- Archived `nam_analysis_config.json` contains a **stale absolute path**. The prepared notebook's 0.2 cell rewrites the config for the local extracted path.

This is an **input-level check**, not a claim that SpikeInterface loading, preprocessing or Kilosort4 ran successfully here. Kilosort4 needs Docker and GPU. A 12.82-second recording can exercise the software workflow, but it cannot establish reliable spike-sorting quality.
