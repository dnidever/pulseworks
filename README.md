# PulseWorks

**PulseWorks** is an interactive Streamlit laboratory for ASTR 372 (Stellar Astrophysics), based on the `Pulsating_Stars.ipynb` assignment. Students investigate pulsating stars using OGLE I-band photometry.

## Activities

1. Examine an observed light curve.
2. Find candidate periods with a Lomb–Scargle periodogram.
3. Refine the pulsation period by phase folding.
4. Classify the star as an RR Lyrae or Cepheid (optionally compare with Layden templates).
5. Estimate its absolute magnitude and distance.

## Installation and launch

```bash
pip install -r requirements.txt
streamlit run app.py
```

The included `OGLE-LMC-RRLYR-00001.dat` is loaded by default. Students may upload another three-column, whitespace-separated light curve (MJD, I magnitude, magnitude uncertainty).

**Optional template file:** The original notebook references `layden.txt`, which was not provided. Upload that file in the app, or place it in the same directory as `app.py`, to enable the RRb/Cepheid template overlay. Expected columns: `phase`, `RRb`, `Cep`.

The period–luminosity relations and default V−I = 0.683 come from the assignment. The distance calculation does **not** include an extinction correction.
