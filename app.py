"""ASTR 372 — Pulsating Stars interactive laboratory."""
from pathlib import Path
import io
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from astropy.timeseries import LombScargle

st.set_page_config(page_title='PulseWorks | ASTR 372', layout='wide')
st.title('PulseWorks')
st.subheader('Pulsating Stars')
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


# Shared results remain available when students navigate between activities.
mean_i = float(np.mean(mag))
observed_range = float(np.percentile(mag,95) - np.percentile(mag,5))
base_freq, base_power = periodogram(tuple(mjd), tuple(mag), .01, 10.0, 10, False, ())
best_default = float(1 / base_freq[int(np.argmax(base_power))])
if 'period' not in st.session_state:
    st.session_state.period = round(best_default, 7)
trial = st.session_state.period
phase = (mjd / trial) % 1
classification = st.session_state.get('classification', 'Not sure yet')
Mv = (1.55 - 1.03*np.log10(trial)) if classification != 'Cepheid' else (-2.81*np.log10(trial)-1.43)
vi = st.session_state.get('vi', .683)
vmag = mean_i + vi

activities = [
    '1. Observed light curve',
    '2. Lomb–Scargle periodogram',
    '3. Refine the period',
    '4. Classify the star',
    '5. Absolute magnitude',
    '6. Apparent V magnitude',
    '7. Distance',
]
with st.sidebar:
    st.divider()
    st.header('Lab activities')
    selected = st.radio('Choose an activity', activities, label_visibility='collapsed')
    st.caption('Select a question to work on it. Your inputs are retained when switching activities.')

def lightcurve_figure(x, y, errors, *, xlabel, title, phase_mode=False, show_errors=False):
    """Interactive OGLE light curve with time, magnitude, and uncertainty on hover."""
    custom = np.column_stack((mjd, errors))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y, mode='markers', name='OGLE photometry',
        marker=dict(size=6, opacity=0.72, color='#3278b7'),
        customdata=custom,
        error_y=dict(type='data', array=errors, visible=show_errors and valid_err,
                     thickness=0.8, width=0),
        hovertemplate=(('Phase: %{x:.5f}<br>' if phase_mode else 'MJD: %{x:.5f}<br>')
                       + 'I magnitude: %{y:.3f}<br>MJD: %{customdata[0]:.5f}'
                       + '<br>I uncertainty: %{customdata[1]:.3f} mag<extra></extra>'),
    ))
    fig.update_layout(title=title, xaxis_title=xlabel, yaxis_title='I magnitude',
                      height=480, hovermode='closest', margin=dict(l=35,r=20,t=55,b=45))
    fig.update_yaxes(autorange='reversed', showgrid=True)
    fig.update_xaxes(showgrid=True, range=[0, 1] if phase_mode else None)
    return fig

if selected == activities[0]:
    st.header('1. Plot the observed light curve')
    st.plotly_chart(lightcurve_figure(mjd, mag, err, xlabel='MJD (days)',
                                    title='OGLE light curve', show_errors=show_errors),
                    use_container_width=True)
    st.caption('Hover over a point for its MJD, magnitude, and uncertainty. Drag to zoom; double-click to reset.')
    st.info('**Think:** Why is it difficult to estimate the pulsation period directly from this plot?')

if selected == activities[1]:
    st.header('2. Create a Lomb–Scargle periodogram')
    st.markdown('Choose the frequency range and sampling, then press **Calculate periodogram**.')
    a,b,c = st.columns(3)
    with a: fmin = st.number_input('Minimum frequency (cycles/day)', min_value=.001, max_value=9.0, value=.01, format='%.3f')
    with b: fmax = st.number_input('Maximum frequency (cycles/day)', min_value=.01, max_value=30.0, value=10.0, format='%.2f')
    with c: spp = st.slider('Samples per peak', 3, 20, 10)
    weighted = st.checkbox('Use measurement uncertainties in periodogram', value=False, disabled=not valid_err,
                           help='The notebook uses an unweighted Lomb–Scargle periodogram by default.')
    max_period_display = st.slider('Show periods up to (days)', .5, 20.0, 4.0, .5)
    # Keep the plotted result fixed until students explicitly recalculate.
    settings = (fmin, fmax, spp, weighted, upload.getvalue() if upload else None)
    if fmin >= fmax:
        st.error('Maximum frequency must exceed minimum frequency.')
    else:
        if st.button('Calculate periodogram', type='primary'):
            freq, power = periodogram(tuple(mjd), tuple(mag), fmin, fmax, spp,
                                      weighted, tuple(err) if weighted else ())
            st.session_state['periodogram_result'] = (freq, power)
            st.session_state['periodogram_settings'] = settings
            st.session_state['show_periodogram_peak'] = False

        if 'periodogram_result' not in st.session_state:
            st.info('Press **Calculate periodogram** to generate the plot.')
        else:
            freq, power = st.session_state['periodogram_result']
            peak = int(np.argmax(power))
            best_period = float(1 / freq[peak])
            if st.session_state.get('periodogram_settings') != settings:
                st.warning('The calculation settings have changed. Press **Calculate periodogram** to update the plot.')
            periods = 1 / freq
            mask = periods <= max_period_display
            idx = np.argsort(periods[mask])
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=periods[mask][idx], y=power[mask][idx],
                mode='lines', name='Lomb–Scargle power',
                line=dict(color='#3278b7', width=1.5),
                hovertemplate='Period: %{x:.7f} d<br>Power: %{y:.5f}<extra></extra>'))
            if st.session_state.get('show_periodogram_peak', False) and best_period <= max_period_display:
                fig.add_vline(x=best_period, line_dash='dash', line_color='#d45543',
                              annotation_text=f'Peak: {best_period:.7f} d')
            fig.update_layout(title='Periodogram', xaxis_title='Period (days)',
                              yaxis_title='Lomb–Scargle power', height=480,
                              hovermode='closest', margin=dict(l=35,r=20,t=55,b=45))
            st.plotly_chart(fig, use_container_width=True)
            st.caption('Hover to read a candidate period and its power; drag to zoom in on narrow peaks.')
            if st.button('Hide peak' if st.session_state.get('show_periodogram_peak', False) else 'Reveal highest peak'):
                st.session_state['show_periodogram_peak'] = not st.session_state.get('show_periodogram_peak', False)
                st.rerun()
            if st.session_state.get('show_periodogram_peak', False):
                st.metric('Highest-power period', f'{best_period:.7f} days')
            st.info('**Think:** Are all the peaks real pulsation periods? How might daily sampling produce aliases?')

def phase_bin_rms(times, magnitudes, period, n_bins=30):
    """RMS about the mean magnitude in each phase bin, pooling residuals."""
    phases = ((times - times.min()) / period) % 1
    bins = np.minimum((phases * n_bins).astype(int), n_bins - 1)
    counts = np.bincount(bins, minlength=n_bins)
    sums = np.bincount(bins, weights=magnitudes, minlength=n_bins)
    means = np.divide(sums, counts, out=np.zeros(n_bins), where=counts > 0)
    # Only bins containing at least two points constrain within-bin scatter.
    used = counts[bins] >= 2
    if not np.any(used):
        return np.nan, 0, 0
    residuals = magnitudes[used] - means[bins[used]]
    return float(np.sqrt(np.mean(residuals**2))), int(used.sum()), int(np.sum(counts >= 2))


if selected == activities[2]:
    st.header('3. Refine the period using the phase-folded light curve')
    st.markdown('The same observations are shown over **two consecutive cycles (phase 0–2)** '
                'so you can see how the light curve joins across phase 1.')
    # Narrow refinement window around the initial periodogram estimate.
    # Small steps let students tune the folded light curve without large jumps.
    center = best_default
    period_min = max(0.00001, round(center - 0.0002, 7))
    period_max = round(center + 0.0002, 7)
    if not period_min <= st.session_state.period <= period_max:
        st.session_state.period = round(center, 7)
    if st.button('Reset trial period to periodogram maximum'):
        st.session_state.period = round(center, 7)
    trial = st.slider('Trial period (days)', min_value=period_min,
                      max_value=period_max, step=0.0000002,
                      format='%.7f', key='period')
    n_bins = 30  # Fixed phase-bin resolution for a consistent RMS comparison.
    folded_phase = ((mjd - mjd.min()) / trial) % 1
    # Repeat every observation at phase+1 to display two cycles.
    display_phase = np.concatenate((folded_phase, folded_phase + 1))
    display_mag = np.tile(mag, 2)
    display_err = np.tile(err, 2)
    display_mjd = np.tile(mjd, 2)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=display_phase, y=display_mag, mode='markers',
        name='OGLE photometry', marker=dict(size=6, opacity=.72, color='#3278b7'),
        customdata=np.column_stack((display_mjd, display_err)),
        error_y=dict(type='data', array=display_err,
                     visible=show_errors and valid_err, thickness=.8, width=0),
        hovertemplate='Phase: %{x:.5f}<br>I magnitude: %{y:.3f}'
                      '<br>MJD: %{customdata[0]:.5f}'
                      '<br>I uncertainty: %{customdata[1]:.3f} mag<extra></extra>'))
    fig.add_vline(x=1, line_dash='dot', line_color='gray', opacity=.4)
    fig.update_layout(title=f'Folded light curve · P = {trial:.7f} days',
                      xaxis_title='Phase (two cycles)', yaxis_title='I magnitude',
                      height=500, hovermode='closest',
                      margin=dict(l=35, r=20, t=55, b=45))
    fig.update_xaxes(range=[0, 2], dtick=.25, showgrid=True)
    fig.update_yaxes(autorange='reversed', showgrid=True)
    st.plotly_chart(fig, use_container_width=True)
    rms, n_used, n_occupied = phase_bin_rms(mjd, mag, trial, n_bins)
    if np.isfinite(rms):
        st.metric('Phase-bin RMS', f'{rms:.4f} mag')
        st.caption(f'Calculated from one cycle: {n_used} observations in '
                   f'{n_occupied} bins with at least two points.')
    else:
        st.warning('Not enough observations per phase bin to calculate RMS.')
    st.caption('The RMS is calculated from the original observations folded into one '
               'cycle, not from the duplicated points displayed in the plot. '
               'It measures the scatter around each phase-bin mean magnitude; '
               'lower values generally indicate a more coherent period.')
    st.info('**Try:** Compare trial periods of 0.6347000 and 0.6347364 days. '
            'Which gives the tighter light curve and smaller RMS?')

if selected == activities[3]:
    st.header('4. Classify the light curve')
    phase = (mjd / trial + st.session_state.get('phase_shift', 0.0)) % 1
    st.markdown('Compare the observed light curve with the bundled Layden templates. The notebook focuses on **RRb** and **Cep**, but all six template types are available.')
    template_path = HERE / 'layden.txt'
    template_source = template_path if template_path.exists() else None
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
            template_name = st.selectbox('Template', ['RRb', 'Cep', 'RRa', 'RRc', 'WUMa', 'Algol'])
            x,y,z = st.columns(3)
            with x: amp = st.slider('Template amplitude (mag)', .05, 2.5, max(.05,min(2.5,round(observed_range,2))), .01)
            with y: offset = st.slider('Template phase offset (cycles)', -1.0, 1.0, 0.0, .01)
            with z: base = st.number_input('Template magnitude offset', value=round(mean_i,3), step=.01, format='%.3f')
            fig = lightcurve_figure(phase, mag, err, xlabel='Phase',
                                    title='Observed light curve and template',
                                    phase_mode=True, show_errors=show_errors)
            template_phase = (templates['phase'].to_numpy() + offset) % 1
            template_mag = amp * templates[template_name].to_numpy() + base
            idx = np.argsort(template_phase)
            fig.add_trace(go.Scatter(x=template_phase[idx], y=template_mag[idx],
                                     mode='lines', name=f'{template_name} template',
                                     line=dict(color='#d45543', width=2.5),
                                     hovertemplate='Template phase: %{x:.5f}<br>'
                                                   'I magnitude: %{y:.3f}<extra></extra>'))
            st.plotly_chart(fig, use_container_width=True)
        except Exception as exc:
            st.warning(f'Could not parse template file: {exc}')
    else:
        st.warning('Bundled Layden templates are missing from this deployment.')
    classification = st.radio('Your classification', ['RR Lyrae', 'Cepheid', 'Not sure yet'], horizontal=True, key='classification', index=None if False else 2)
    st.caption('Notebook guidance: RR Lyrae periods ~0.5–1 day, amplitudes ~0.4–1.2 mag; Cepheids periods ~3–100 days, amplitudes around 0.7 mag. Use both shape and period/amplitude.')
    st.text_area('Explain your classification using the observed period and light-curve shape:')

if selected == activities[4]:
    st.header('5. Calculate absolute magnitude from the period–luminosity relation')
    st.latex(r'M_V(\mathrm{RR\ Lyrae}) = 1.55 - 1.03\log_{10}(P)')
    st.latex(r'M_V(\mathrm{Cepheid}) = -2.81\log_{10}(P) - 1.43')
    relation = st.selectbox('Use relation for', ['RR Lyrae','Cepheid'], index=0 if classification!='Cepheid' else 1, key='relation')
    Mv = (1.55-1.03*np.log10(trial)) if relation=='RR Lyrae' else (-2.81*np.log10(trial)-1.43)
    st.metric('Absolute V magnitude',f'{Mv:.3f} mag')

if selected == activities[5]:
    st.header('6. Determine mean apparent V magnitude')
    vi = st.number_input('Assumed V − I color (mag)', value=.683, step=.001, format='%.3f', key='vi')
    vmag = mean_i + vi
    left,right = st.columns(2)
    left.metric('Mean I magnitude',f'{mean_i:.3f} mag')
    right.metric('Estimated mean V magnitude',f'{vmag:.3f} mag')
    st.latex(r'V = I + (V-I)')

if selected == activities[6]:
    st.header('7. Calculate the distance')
    st.latex(r'\mu = m_V - M_V = 5\log_{10}(d/\mathrm{pc})-5')
    relation = st.session_state.get('relation', 'RR Lyrae')
    Mv = (1.55 - 1.03*np.log10(trial)) if relation == 'RR Lyrae' else (-2.81*np.log10(trial)-1.43)
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