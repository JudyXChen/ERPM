"""Stimulate the equilibrated model in all four RyR/SOCE conditions.

Writes traces, observables, flux budgets and figures to results/stimulus/TAG and
results/Flux/TAG; --plot-only redraws them from saved CSVs without simulating.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from scipy.optimize import brentq
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model.parameters import default_parameters
from model import stimulus
from model.ode0d import build_system, equilibrate_state, _make_voltage, _solve_with_pulses, rhs
from scripts.er_flux_decomp import flux_series, write_flux_data, plot_budget_counts, cumulative
from scripts.io_utils import read_columns, write_csv
from scripts import plot_style as style

OUT = ROOT / 'results/stimulus/ryr_soce_repair'
FLUX = ROOT / 'results/Flux/ryr_soce_repair'
K_STIM1 = None  # Optional override for threshold comparisons.


def set_output(tag):
    global OUT, FLUX
    OUT = ROOT / 'results/stimulus' / tag
    FLUX = ROOT / 'results/Flux' / tag

def prepare(p, soce=False, ryr_off=False):
    p.Glu_init = 0.0
    names, y, compiled, _ = build_system(p, include_soce=soce, ryr_off=ryr_off)
    eq = equilibrate_state(y, compiled, _make_voltage(-65), names=names, max_time_s=4000)
    if not eq.converged:
        raise RuntimeError('Rest failed to converge')
    return names, compiled, eq

def time_grid(pulses, end=300.0):
    """Sample every release finely; single pulses keep the original 19851-point grid."""
    parts = []
    for start, following in zip(pulses, [*pulses[1:], None]):
        if len(pulses) == 1:
            parts.append(np.linspace(start, start+.1, 10001))
        else:  # 10 us through the fast transient, 40 us after, to limit train CSV size.
            parts += [np.linspace(start, start+.02, 2001), np.linspace(start+.02, start+.1, 2001)]
        if following is not None:
            parts.append(np.linspace(start+.1, following, round((following-start-.1)/.002)+1))
    last = pulses[-1]
    parts += [np.linspace(last+.1, last+10, 4951), np.linspace(last+10, last+60, 2501),
              np.linspace(last+60, end, round((end-last-60)/.1)+1)]
    return np.unique(np.concatenate(parts))


def pulse_table(t, er, pulses):
    """ER calcium just before each release and its minimum before the next one."""
    rows = []
    for start, following in zip(pulses, [*pulses[1:], t[-1]]):
        window = np.flatnonzero((t >= start) & (t <= following))
        low = window[np.argmin(er[window])]
        rows.append({'pulse_time_s': float(start), 'ER_before_uM': float(np.interp(start, t, er)),
                     'ER_min_uM': float(er[low]), 'ER_min_after_pulse_s': float(t[low]-start)})
    return rows


def calibrate():
    def error(leak):
        p = default_parameters(); p.k_S_leak_gq = leak
        n, _, eq = prepare(p)
        return eq.state[n.index('Ca_ER')]*1e6 - 150
    leak = brentq(error, 0.08, 0.3, xtol=1e-11)
    return leak

def plot_soce_comparison(data, conditions):
    """SOCE off/on with RyR on: Ca_c/peak (100 ms), Ca_ER/rest and gate (60 s)."""
    folder = OUT/'soce_comparison'
    panels = [('Ca_c_norm_100ms.png', 'Ca_c_nM', 'Time (ms)', 'Normalized cytosolic calcium', .1, 1e3,
               lambda v: v/v.max()),
              ('Ca_ER_norm_60s.png', 'Ca_ER_uM', 'Time (s)', 'Normalized ER calcium', 60, 1,
               lambda v: v/v[0]),
              ('SOCE_Popen_60s.png', 'SOCE_Popen', 'Time (s)', 'SOCE open probability', 60, 1,
               lambda v: v)]
    for name, field, xlabel, ylabel, end, scale, transform in panels:
        fig, ax = style.figure()
        for key, label in conditions.items():
            if field == 'SOCE_Popen' and key.endswith('soce_off'):
                continue
            _, color, ls = style.CONDITIONS[key]
            d = data[key]; mask = d['time_s'] <= end
            ax.plot(d['time_s'][mask]*scale, transform(d[field])[mask], label=label, color=color, linestyle=ls)
        ax.set(xlabel=xlabel, ylabel=ylabel, xlim=(0, end*scale))
        style.save(fig, folder/name, legend_name=None if field == 'SOCE_Popen' else 'legend.png')


def plot_er_with_baseline(data, start=-5, end=60):
    """ER calcium with the equilibrated rest drawn before the stimulus at t = 0."""
    fig, ax = style.figure()
    for key, (name, color, ls) in style.CONDITIONS.items():
        d = data[key]; mask = d['time_s'] <= end
        ax.plot(np.r_[start, d['time_s'][mask]], np.r_[d['Ca_ER_uM'][0], d['Ca_ER_uM'][mask]],
                label=name, color=color, linestyle=ls)
    ax.set(xlabel='Time (s)', ylabel='ER calcium (µM)', xlim=(start, end))
    style.save(fig, OUT/f'Ca_ER_uM_{-start:g}s_before_{end:g}s.png', legend_name='legend.png')


def plot_er_log_time(data):
    """ER calcium on log time, resolving the fast release and the slow refill."""
    fig, ax = style.figure()
    for key, (name, color, ls) in style.CONDITIONS.items():
        d = data[key]; mask = d['time_s'] > 0
        ax.plot(d['time_s'][mask], d['Ca_ER_uM'][mask], label=name, color=color, linestyle=ls)
    ax.set(xlabel='Time after stimulus (s)', ylabel='ER calcium (µM)', xscale='log',
           xlim=(1e-3, 100), ylim=(0, None))
    style.save(fig, OUT/'Ca_ER_uM_log_time.png', legend_name='legend.png')


def plot():
    style.apply_style()
    data = {key: read_columns(OUT/key/'observables.csv') for key in style.CONDITIONS}
    every = {key: value[0] for key, value in style.CONDITIONS.items()}
    ryr_on = {'ryr_on_soce_off': 'SOCE off', 'ryr_on_soce_on': 'SOCE on'}
    soce_on = {'ryr_on_soce_on': 'RyR on', 'ryr_off_soce_on': 'RyR off'}
    summary = json.loads((OUT/'summary.json').read_text()) if (OUT/'summary.json').exists() else {}
    pulses = summary.get('protocol', {}).get('pulse_times_s', [0.0])
    specs = [('Ca_c_nM', 'Cytosolic calcium (nM)', .1, every, 'legend.png'),
             ('RyR_Popen', 'RyR open probability', .1, ryr_on, 'legend_ryr_on.png'),
             ('Ca_ER_uM', 'ER calcium (µM)', 60, every, 'legend.png'),
             ('IP3R_Popen', 'IP3R open probability', 60, every, 'legend.png'),
             ('RyR_Popen', 'RyR open probability', 60, ryr_on, 'legend_ryr_on.png'),
             ('SOCE_Popen', 'SOCE open probability', 60, soce_on, 'legend_soce_on.png')]
    if len(pulses) > 1:  # The 0.1 s panels only show the first release of a train.
        specs.append(('Ca_c_nM', 'Cytosolic calcium (nM)', pulses[-1]+5, every, 'legend.png'))
    for field, label, end, conditions, legend in specs:
        fig, ax = style.figure()
        for key, name in conditions.items():
            _, color, ls = style.CONDITIONS[key]
            d = data[key]; mask = d['time_s'] <= end
            ax.plot(d['time_s'][mask], d[field][mask], label=name, color=color, linestyle=ls)
        ax.set(xlabel='Time (s)', ylabel=label, xlim=(0, end))
        style.save(fig, OUT/f'{field}_{end:g}s.png', legend_name=legend)
    plot_soce_comparison(data, ryr_on)
    plot_er_with_baseline(data)
    plot_er_log_time(data)
    for key in style.CONDITIONS:
        col=read_columns(FLUX/key/'budget.csv')
        flux={};counts={}
        for compartment,prefix in [('ER','ER'),('cytosol','cyto')]:
            flux[compartment]={k[len(prefix)+1:-7].replace('_',' '):v for k,v in col.items()
                               if k.startswith(prefix+'_') and k.endswith('_ions_s')}
            counts[compartment]=col[prefix+'_recorded_change_ions']
        plot_budget_counts(FLUX/key/'budget.png',col['time_s'],counts,flux)
        excess={comp:{name:values-values[0] for name,values in terms.items()}
                for comp,terms in flux.items()}
        plot_budget_counts(FLUX/key/'excess_budget.png',col['time_s'],counts,excess)

def run(leak, n_pulses=1, freq=1.0):
    OUT.mkdir(parents=True,exist_ok=True)
    pulses=stimulus.release_times(freq,n_pulses)
    summary={'protocol':{'glu_molecules':500,'n_pulses':n_pulses,'freq_hz':freq,'pulse_times_s':pulses},
             'parameters':{},'runs':{},'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
             for p in (ROOT/'model').rglob('*.py')}}
    for key in style.CONDITIONS:
        soce=key.endswith('soce_on'); ryr_off=key.startswith('ryr_off')
        p=default_parameters();p.k_S_leak_gq=leak
        if K_STIM1 is not None: p.K_STIM1=K_STIM1
        names,compiled,eq=prepare(p,soce,ryr_off); idx={n:i for i,n in enumerate(names)}
        rest=eq.state.copy();y=rest.copy();y[idx['Glu']]=stimulus.molecules_to_conc(500,p)
        t=time_grid(pulses)
        voltage=_make_voltage(stimulus.voltage(times=pulses,V_rest=-65))
        sol=_solve_with_pulses(rhs,(0,300),y,method='BDF',t_eval=t,args=(compiled,voltage),
                               pulse_times=pulses,pulse_amount=y[idx['Glu']],glu_idx=idx['Glu'],
                               extra_boundaries=stimulus.voltage_onsets(pulses),rtol=1e-9,atol=1e-14)
        if not sol.success or np.min(sol.y)<-1e-9: raise RuntimeError('Invalid stimulus solve')
        dest=OUT/key;dest.mkdir(parents=True,exist_ok=True)
        write_csv(dest/'traces.csv',{'t':t,**{n:sol.y[i] for i,n in enumerate(names)}})
        po=sum((sol.y[i] for i,n in enumerate(names) if n.startswith('R_O_')),np.zeros_like(t))
        obs={'time_s':t,'Ca_ER_uM':sol.y[idx['Ca_ER']]*1e6,'Ca_c_nM':sol.y[idx['Ca_c']]*1e9,
             'IP3_nM':sol.y[idx['IP3']]*1e9,'RyR_Popen':po,'IP3R_Popen':sol.y[idx['IPR_110']]**3,
             'SOCE_Popen':sol.y[idx['P_O_STIM1']] if soce else np.zeros_like(t)}
        write_csv(dest/'observables.csv',obs)
        f=flux_series(p,names,compiled,t,sol.y,voltage)
        fd=FLUX/key;fd.mkdir(parents=True,exist_ok=True)
        write_flux_data(fd/'budget.csv',p,names,t,sol.y,f)
        # Compare grouped flux with the full assembled derivatives, including RyR binding.
        errors={}
        for comp,name,volume in [('ER','Ca_ER',p.V_ER*1e-15),('cytosol','Ca_c',p.V_spine_L)]:
            exact=np.array([rhs(t[k],sol.y[:,k],compiled,voltage)[idx[name]]*p.N_A*volume for k in range(len(t))])
            grouping=float(np.max(abs(exact-f[comp]['net'])))
            recorded=(sol.y[idx[name]]-rest[idx[name]])*p.N_A*volume
            closure=float(np.max(abs(recorded-cumulative(f[comp]['net'],t))))
            errors[comp]={'max_grouping_error_ions_s':grouping,'max_integral_error_ions':closure}
            if grouping>1e-5 or closure>1.0: raise RuntimeError(errors)
        er=obs['Ca_ER_uM']; imin=int(np.argmin(er))
        # Recovery is measured from the minimum after the last release.
        after=np.flatnonzero(t>=pulses[-1]); ilast=int(after[np.argmin(er[after])])
        target=er[0]-.5*(er[0]-er[ilast])
        crossing=np.flatnonzero((t>=t[ilast]) & (er>=target))
        recovery={}
        for percent in (50,90,95,99):
            hits=np.flatnonzero((t>=t[ilast]) & (er>=er[ilast]+percent/100*(er[0]-er[ilast])))
            recovery[f'recovery_{percent}_after_min_s']=float(t[hits[0]]-t[ilast]) if len(hits) else None
        rf=flux_series(p,names,compiled,np.array([0.]),rest[:,None],_make_voltage(-65))
        row={'rest_ER_uM':float(rest[idx['Ca_ER']]*1e6),'rest_cyto_nM':float(rest[idx['Ca_c']]*1e9),
             'rest_ER_net_ions_s':float(rf['ER']['net'][0]),'rest_cyto_net_ions_s':float(rf['cytosol']['net'][0]),
             'rest_Ca_relative_rate_s':eq.ca_relative_rate_s,'minimum_ER_uM':float(er[imin]),
             'minimum_ER_time_s':float(t[imin]),'final_ER_uM':float(er[-1]),
             'half_recovery_after_min_s':float(t[crossing[0]]-t[ilast]) if len(crossing) else None,
             **recovery,'per_pulse':pulse_table(t,er,pulses),
             'peak_cyto_nM':float(max(obs['Ca_c_nM'])),'peak_RyR_Popen':float(max(po)),
             'RyR_released_ions':float(-cumulative(f['ER'].get('RyR',np.zeros_like(t)),t)[-1]),
             'IP3R_released_ions':float(-cumulative(f['ER']['IP3R'],t)[-1]),'flux_checks':errors,
             'minimum_state':float(sol.y.min()),
             'RyR_excess_release_ions':float(-cumulative(f['ER'].get('RyR',np.zeros_like(t))-rf['ER'].get('RyR',np.array([0.]))[0],t)[-1]),
             'IP3R_excess_release_ions':float(-cumulative(f['ER']['IP3R']-rf['ER']['IP3R'][0],t)[-1])}
        if abs(row['rest_ER_net_ions_s'])>1e-3 or abs(row['rest_cyto_net_ions_s'])>1e-3:
            raise RuntimeError('Resting flux imbalance')
        summary['runs'][key]=row
        summary['parameters']=vars(p)
        print(key,json.dumps(row),flush=True)
        with (dest/'channel_summary.csv').open('w') as handle:
            writer=csv.writer(handle);writer.writerow(['channel','rest_Popen','peak_Popen','time_peak_s'])
            for field in ['RyR_Popen','IP3R_Popen','SOCE_Popen']:
                vals=obs[field];writer.writerow([field,vals[0],max(vals),t[np.argmax(vals)]])
    for prefix in ('ryr_on','ryr_off'):
        a=summary['runs'][prefix+'_soce_off']['rest_ER_uM']
        b=summary['runs'][prefix+'_soce_on']['rest_ER_uM']
        if abs(b/a-1)>.01: print(f'WARNING {prefix}: SOCE resting ER shift {100*(b/a-1):.1f}%',flush=True)
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    plot()

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--calibrate',action='store_true');ap.add_argument('--plot-only',action='store_true')
    ap.add_argument('--tag',help='write to results/stimulus/TAG and results/Flux/TAG')
    ap.add_argument('--k-stim-um',type=float,help='override K_STIM1 (uM)')
    ap.add_argument('--glu-pulses',type=int,default=1,help='number of glutamate+EPSP+bAP releases')
    ap.add_argument('--glu-freq',type=float,default=1.0,help='release frequency in a train (Hz)')
    args=ap.parse_args()
    if args.tag: set_output(args.tag)
    if args.k_stim_um is not None: K_STIM1=args.k_stim_um*1e-6
    if args.plot_only: plot()
    elif args.calibrate: print('CALIBRATED_LEAK',calibrate())
    else: run(default_parameters().k_S_leak_gq,args.glu_pulses,args.glu_freq)
