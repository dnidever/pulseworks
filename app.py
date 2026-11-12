"""ASTR 372 — Pulsating Stars interactive laboratory."""
from pathlib import Path
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from astropy.timeseries import LombScargle

st.set_page_config(page_title='ASTR 372 | Pulsating Stars', layout='wide')
st.title('Pulsating Stars')
st.caption('ASTR 372 · Interactive lab · OGLE I-band photometry')
st.markdown('Explore a variable-star light curve, measure its period, classify the star, and estimate its distance.')

HERE = Path(__file__).resolve().parent

def read_photometry(source):
    df = pd.read_csv(source, sep=r'\s+', comment='#', header=None, names=['MJD','IMAG','IERR'], usecols=[0,1,2])
    for col in ['MJD','IMAG','IERR']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['MJD','IMAG'])
    df = df.sort_values('MJD')
    if len(df) < 5 or df['MJD'].nunique() < 5:
        raise ValueError('At least five observations at distinct times are needed.')
    return df

@st.cache_data(show_spinner=False)
def periodogram(t, mag, fmin, fmax, spp, use_errors, errors):
    tt = np.array(t, dtype=float)
    mm = np.array(mag, dtype=float)
    ee = np.array(errors, dtype=float) if use_errors else None
    freq, power = LombScargle(tt, mm, dy=ee).autopower(minimum_frequency=fmin, maximum_frequency=fmax, samples_per_peak=spp)
    return freq, power

st.sidebar.header('Data')
upload = st.sidebar.file_uploader('Optional: upload another OGLE-style data file', type=['dat','txt','csv'])
try:
    data = read_photometry(io.BytesIO(upload.getvalue()) if upload else HERE / 'OGLE-LMC-RRLYR-00001.dat')
except Exception as exc:
    st.error(f'Unable to load the photometry: {exc}')
    st.stop()

mjd = data['MJD'].to_numpy()
mag = data['IMAG'].to_numpy()
err = data['IERR'].to_numpy()
valid_err = np.all(np.isfinite(err) & (err > 0))
with st.sidebar:
    st.metric('Observations', len(data))
    st.metric('Time baseline', f'{np.ptp(mjd):,.1f} d')
    show_errors = st.checkbox('Show photometric error bars', value=False)
    st.caption('Astronomical magnitudes: smaller numbers are brighter.')

st.header('1. Plot the observed light curve')
fig, ax = plt.subplots(figsize=(9,3.6))
if show_errors and valid_err:
    ax.errorbar(mjd, mag, yerr=err, fmt='.', ms=4, alpha=.75, lw=.6)
else:
    ax.scatter(mjd, mag, s=13, alpha=.75)
ax.set(xlabel='MJD (days)', ylabel='I magnitude', title='OGLE light curve')
ax.invert_yaxis()
ax.grid(alpha=.2)
st.pyplot(fig)
plt.close(fig)
st.info('**Think:** Why is it difficult to estimate the pulsation period directly from this plot?')

st.header('2. Create a Lomb–Scargle periodogram')
a,b,c = st.columns(3)
with a: fmin = st.number_input('Minimum frequency (cycles/day)', min_value=.001, max_value=9.0, value=.01, format='%.3f')
with b: fmax = st.number_input('Maximum frequency (cycles/day)', min_value=.01, max_value=30.0, value=10.0, format='%.2f')
with c: spp = st.slider('Samples per peak', 3, 20, 10)
if fmin >= fmax:
    st.error('Maximum frequency must exceed minimum frequency.')
    st.stop()
weighted = st.checkbox('Use measurement uncertainties in periodogram', value=False, disabled=not valid_err,
                       help='The notebook uses an unweighted Lomb–Scargle periodogram by default.')
freq, power = periodogram(tuple(mjd), tuple(mag), fmin, fmax, spp, weighted, tuple(err) if weighted else ())
peak = int(np.argmax(power))
best_period = float(1 / freq[peak])
max_period_display = st.slider('Show periods up to (days)', .5, 20.0, 4.0, .5)
order = np.argsort(1/freq)
periods = 1/freq
fig, ax = plt.subplots(figsize=(9,3.6))
mask = periods <= max_period_display
ax.plot(periods[order][np.isin(order, np.flatnonzero(mask))], power[order][np.isin(order, np.flatnonzero(mask))], lw=1)
if best_period <= max_period_display:
    ax.axvline(best_period, ls='--', color='tab:red', label=f'Peak: {best_period:.7f} d')
    ax.legend()
ax.set(xlabel='Period (days)', ylabel='Lomb–Scargle power', title='Periodogram')
ax.grid(alpha=.2)
st.pyplot(fig)
plt.close(fig)
st.metric('Highest-power period', f'{best_period:.7f} days')
st.info('**Think:** Are all the peaks real pulsation periods? How might daily sampling produce aliases?')

st.header('3. Refine the period using the phase-folded light curve')
if 'period' not in st.session_state:
    st.session_state.period = round(best_period, 7)
if st.button('Reset trial period to periodogram maximum'):
    st.session_state.period = round(best_period, 7)
trial = st.number_input('Trial period (days)', min_value=.000001, max_value=1000.0, step=.0000001,
                        format='%.7f', key='period')
phase_offset_display = st.slider('Phase shift for display (cycles)', -1.0, 1.0, 0.0, .01)
phase = (mjd / trial + phase_offset_display) % 1
fig, ax = plt.subplots(figsize=(9,3.7))
if show_errors and valid_err:
    ax.errorbar(phase, mag, yerr=err, fmt='.', ms=4, alpha=.65, lw=.5)
else:
    ax.scatter(phase, mag, s=14, alpha=.7)
ax.set(xlabel='Phase', ylabel='I magnitude', title=f'Folded light curve · P = {trial:.7f} days', xlim=(0,1))
ax.invert_yaxis()
ax.grid(alpha=.2)
st.pyplot(fig)
plt.close(fig)
st.info('**Try:** Compare periods of 0.6347000 and 0.6347364 days, as in the notebook. Which produces a cleaner light curve? Why does such a small change matter over a long observing baseline?')

st.header('4. Classify the light curve')
st.markdown('The notebook uses Layden templates named **RRb** and **Cep**. Upload the original `layden.txt` to compare them with the data.')
template_upload = st.file_uploader('Layden template file (optional)', type=['txt','dat','csv'])
template_path = HERE / 'layden.txt'
template_source = io.BytesIO(template_upload.getvalue()) if template_upload else (template_path if template_path.exists() else None)
mean_i = float(np.mean(mag))
observed_range = float(np.percentile(mag,95) - np.percentile(mag,5))
st.write(f'Observed mean I magnitude: **{mean_i:.3f}**; approximate 5–95% magnitude range: **{observed_range:.3f} mag**.')
if template_source is not None:
    try:
        templates = pd.read_csv(template_source, sep=r'\s+', comment='#')
        # Support files without a header, or header lines commented out.
        if not {'phase','RRb','Cep'}.issubset(templates.columns):
            if hasattr(template_source, 'seek'): template_source.seek(0)
            templates = pd.read_csv(template_source, sep=r'\s+', comment='#', header=None)
            if templates.shape[1] < 3: raise ValueError('Expected at least three columns: phase, RRb, Cep.')
            templates = templates.iloc[:, :3]
            templates.columns = ['phase','RRb','Cep']
        templates = templates.apply(pd.to_numeric, errors='coerce').dropna(subset=['phase','RRb','Cep'])
        if templates.empty: raise ValueError('No numeric template rows found.')
        template_name = st.selectbox('Template', ['RRb','Cep'])
        x,y,z = st.columns(3)
        with x: amp = st.slider('Template amplitude (mag)', .05, 2.5, max(.05,min(2.5,round(observed_range,2))), .01)
        with y: offset = st.slider('Template phase offset (cycles)', -1.0, 1.0, 0.0, .01)
        with z: base = st.number_input('Template magnitude offset', value=round(mean_i,3), step=.01, format='%.3f')
        fig, ax = plt.subplots(figsize=(9,3.7))
        ax.scatter(phase,mag,s=12,alpha=.55,label='OGLE photometry')
        template_phase = (templates['phase'].to_numpy()+offset)%1
        template_mag = amp*templates[template_name].to_numpy()+base
        # Repeat at phase boundaries for continuity.
        idx = np.argsort(template_phase)
        ax.plot(template_phase[idx],template_mag[idx],color='tab:red',lw=2,label=template_name)
        ax.set(xlabel='Phase',ylabel='I magnitude',xlim=(0,1))
        ax.invert_yaxis();ax.legend();ax.grid(alpha=.2)
        st.pyplot(fig);plt.close(fig)
    except Exception as exc:
        st.warning(f'Could not parse template file: {exc}')
else:
    st.warning('Layden templates not provided. Template comparison is disabled; all other activities remain available.')
classification = st.radio('Your classification', ['RR Lyrae', 'Cepheid', 'Not sure yet'], horizontal=True)
st.caption('Notebook guidance: RR Lyrae periods ~0.5–1 day, amplitudes ~0.4–1.2 mag; Cepheids periods ~3–100 days, amplitudes around 0.7 mag. Use both shape and period/amplitude.')
st.text_area('Explain your classification using the observed period and light-curve shape:')

st.header('5. Calculate absolute magnitude from the period–luminosity relation')
st.latex(r'M_V(\mathrm{RR\ Lyrae}) = 1.55 - 1.03\log_{10}(P)')
st.latex(r'M_V(\mathrm{Cepheid}) = -2.81\log_{10}(P) - 1.43')
relation = st.selectbox('Use relation for', ['RR Lyrae','Cepheid'], index=0 if classification!='Cepheid' else 1)
Mv = (1.55-1.03*np.log10(trial)) if relation=='RR Lyrae' else (-2.81*np.log10(trial)-1.43)
st.metric('Absolute V magnitude',f'{Mv:.3f} mag')

st.header('6. Determine mean apparent V magnitude')
vi = st.number_input('Assumed V − I color (mag)', value=.683, step=.001, format='%.3f')
vmag = mean_i + vi
left,right = st.columns(2)
left.metric('Mean I magnitude',f'{mean_i:.3f} mag')
right.metric('Estimated mean V magnitude',f'{vmag:.3f} mag')
st.latex(r'V = I + (V-I)')

st.header('7. Calculate the distance')
st.latex(r'\mu = m_V - M_V = 5\log_{10}(d/\mathrm{pc})-5')
mu = vmag - Mv
dist_pc = 10**(mu/5+1)
a,b = st.columns(2)
a.metric('Distance modulus',f'{mu:.3f} mag')
b.metric('Distance',f'{dist_pc/1000:.2f} kpc')
st.caption('As in the notebook, this calculation does not include an extinction correction. The provided relations are reproduced from the assignment.')
st.info('**Reflect:** How would dust extinction affect the inferred distance? Which measurements contribute uncertainty?')

with st.expander('Instructor / student notes'):
    st.markdown('''- Periods in the periodogram are computed as **1/frequency**. The highest-power peak is a starting point, not automatically the true period.
- The original notebook explicitly compares trial periods **0.6347000** and **0.6347364 days**.
- The notebook computes the **arithmetic mean I magnitude**, not a flux-weighted mean.
- The light curve is I-band, but the supplied period–luminosity relations give **M_V**; the adopted V−I color provides the conversion.
- A phase shift changes the appearance of the folded curve but does not change the measured period.''')