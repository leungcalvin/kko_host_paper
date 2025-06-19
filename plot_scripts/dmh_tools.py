import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable


def sigma_tot(df):
    """Include IGM and foreground contributions to noise"""
    return np.sqrt(sigma_igm_walker2024(df['primary_z_spec'])**2 + (df['DM_NE2001'] * 0.2)**2)

def sigma_igm_walker2024(z):
    return np.interp(x = z,xp = [0,0.1,0.2], fp = [0, 63.30, 104.84])

def get_log10_M_star_mahajan_2016(M_r):
    log10_m_star = 0.450 - 0.464 * M_r
    unc = 0.458
    return log10_m_star, unc

def get_log10_M_star_wise_jarrett(log_L_W1):
    coefficients = [-12.62185, 5.00155, -0.43857, 0.01593]
    M_star = coefficients[0] + coefficients[1] * log_L_W1 + coefficients[2] * log_L_W2**2 + coefficients[3] * log_L_W2**3
def mr2ms_mahajan(mr):
    ms, unc = get_log10_M_star_mahajan_2016(mr)
    return ms

def ms2mr_mahajan(ms):
    M_r = (ms - 0.450) / (-0.464) # NOTE: this is for all galaxies, not just spiral... hence +- 0.46 dex
    return M_r

def mh2ms(mh):
    # Use Behroozi
    if (mh > 13).any():
        print("WARNING: Behroozi not good beyond Mh = 1e13 Msol")
    m_halo = np.atleast_1d(mh)
    behroozi_2018_smhr = pd.read_csv('/arc/home/calvin/kko_host_paper/data_products/behroozi_2018_shmr.csv',names = ['M_halo','M_star'])
    log10_m_star = np.interp(x = mh, xp = behroozi_2018_smhr['M_halo'],fp = behroozi_2018_smhr['M_star'])
    return log10_m_star

def best_stellar_mass(df):
    out = np.zeros_like(df['published_M*'])
    janky = np.isnan(df['published_M*'])
    out[janky] = mr2ms_mahajan(df['M_r'])[janky]
    out[~np.isnan(df['published_M*'])] = df['published_M*'][~np.isnan(df['published_M*'])]
    print('Still needs M* measurement:',df.index[janky])
    return out
    
def get_M_r_mahajan_2016(m_halo):
    # Use Behroozi & Mahajan to convert Mh to Mr
    ms = mh2ms(m_halo)
    M_r = (ms - 0.450) / (-0.464) # NOTE: this is for all galaxies, not just spiral... hence +- 0.46 dex
    return M_r

def get_M_h_mahajan_2016(M_r):
    M_r = np.atleast_1d(M_r)
    log10_m_star = -0.464 * M_r + 0.450
    behroozi_2018_smhr = pd.read_csv('/arc/home/calvin/kko_host_paper/data_products/behroozi_2018_shmr.csv',names = ['M_halo','M_star'])
    log10_m_halo = np.interp(x = log10_m_star, xp = behroozi_2018_smhr['M_star'],fp = behroozi_2018_smhr['M_halo'])
    return log10_m_halo

def plot_sim(ax,Mr_range = np.array([-23,-19]), which = ['Astrid','IllustrisTNG','SIMBA']):
    """Model predictions directly from two bins in Medlock+2023. M_halo -> DM_host"""
    astrid = [72, 118, 44],[174, 299, 117] # median, high, low in Medlock+2023
    tng = [46, 83, 29],[68, 105, 39]
    simba = [24,50,14],[36,69,18]
    Mh = np.array([11.5,12.5])
    implied_M_r = get_M_r_mahajan_2016(Mh)
    print('Medlock predictions valid over M_r =')
    print(implied_M_r)
    
    for sim, legend_label in zip([astrid,tng,simba],['Astrid','IllustrisTNG','SIMBA']):
        if legend_label in which:
            _y1 = [sim[0][2],sim[1][2]]
            _y2 = [sim[0][1],sim[1][1]]
            _med = [sim[0][0],sim[1][0]]
            from scipy.interpolate import interp1d
            _y1_func = interp1d(implied_M_r, _y1, kind='linear', fill_value='extrapolate')
            _y2_func = interp1d(implied_M_r, _y2, kind='linear', fill_value='extrapolate')
            _med_func = interp1d(implied_M_r, _med, kind='linear', fill_value='extrapolate')
            #EB = ax.scatter(Mr_range,_med_func(Mr_range))
            ax.fill_between(Mr_range,_y1_func(Mr_range),
                            _y2_func(Mr_range),
                            label = legend_label,alpha = 0.8)#color = EB.get_edgecolors())
    
def plot_sim_direct(ax,Mr_range = np.linspace(-23,-19), which = ['Astrid','IllustrisTNG','SIMBA']):
    """M_stellar -> DM_host in 5 bins from Isabel google doc https://docs.google.com/document/d/1CG1MHz5JBqB18qY8I9mHesMTrDYdLc3viOf4ypZvdBM/edit?tab=t.6q2ktat2xhsv"""
    m_star_bins = np.array([8.75, 9.25, 9.75, 10.25, 10.75])
    implied_M_r = ms2mr(m_star_bins)
    data = {
        'tng_mean': np.array([89.04, 115.76, 143.04, 142.42, 97.44]),
        'tng_std': np.array([89.73, 112.73, 143.36, 148.52, 108.24]),
        'simba_mean': np.array([55.54, 58.60, 62.34, 58.53, 40.69]),
        'simba_std': np.array([113.00,96.47,85.94,82.97,72.67]),
        'astrid_mean': np.array([100.29,129.71,167.72,270.16,325.13]),
        'astrid_std': np.array([117.01,127.17,116.51,124.17,173.96]),
    }
    for sim, legend_label in zip(['astrid','tng','simba'],['Astrid','IllustrisTNG','SIMBA']):
        if legend_label in which:
            _y1 = data[sim + '_mean'] - data[sim + '_std']
            _y2 = data[sim + '_mean'] + data[sim + '_std']
            from scipy.interpolate import interp1d
            _y1_func = interp1d(implied_M_r, _y1, kind='linear', fill_value='extrapolate')
            _y2_func = interp1d(implied_M_r, _y2, kind='linear', fill_value='extrapolate')
            #EB = ax.scatter(Mr_extrap,_med_func(Mr_range))
            ax.fill_between(Mr_range,_y1_func(Mr_range),
                            _y2_func(Mr_range),
                            label = legend_label,alpha = 0.8)#color = EB.get_edgecolors())
    

    
def mvsk_lognormal(m,delta_m_plus,delta_m_minus,w,delta_w_plus,delta_w_minus,label):
    """mu is in natural DM units. mu_err is also. sigma is in log units (i.e. does not matter log10 or ln). sigma_merr and sigma_perr are also in log units.
    
    Return the equivalent moments for a log-normal distribution."""

    sigma_m = (delta_m_plus + delta_m_minus) / 2
    sigma_w = (delta_w_plus + delta_w_minus) / 2

    # Moments formulas
    def mu_real(m, w):
        return np.exp(m + (w**2) / 2)

    def var_real(m, w):
        return (np.exp(w**2) - 1) * np.exp(2 * m + w**2)

    def skewness(w):
        return (np.exp(w**2) + 2) * np.sqrt(np.exp(w**2) - 1)

    def kurtosis(w):
        return np.exp(4 * w**2) + 2 * np.exp(3 * w**2) + 3 * np.exp(2 * w**2) - 6
    dmu_dm = (mu_real(m + sigma_m,w) - mu_real(m - sigma_m,w)) / sigma_m / 2
    dmu_dw = (mu_real(m,w + sigma_w) - mu_real(m,w - sigma_w)) / sigma_w / 2
    dvar_dm = (var_real(m + sigma_m,w) - var_real(m - sigma_m,w)) / sigma_m / 2
    dvar_dw = (var_real(m,w + sigma_w) - var_real(m,w - sigma_w)) / sigma_w / 2
    
    # Error propagation using Gaussian propagation formula
    mu_err = np.sqrt((dmu_dm * sigma_m)**2 + (dmu_dw * sigma_w)**2)
    var_err = np.sqrt((dvar_dm * sigma_m)**2 + (dvar_dw * sigma_w)**2)
    skew_err = np.abs(dskew_dw(w)) * sigma_w
    kurt_err = np.abs(dkurt_dw(w)) * sigma_w

    # Calculating the values of the moments in real space
    mu_value = mu_real(m, w)
    var_value = var_real(m, w)
    skew_value = skewness(w)
    kurt_value = kurtosis(w)
    std_value = np.sqrt(var_value)
    std_err = var_err / (2 * np.sqrt(var_value))
    print(label)
    print(var_value,var_err,sigma_m,sigma_w)
    # Output the results
    print(f"Mean (non-logged): {mu_value:.4f} ± {mu_err:.4f}; median = {np.exp(m):.4f}")
    print(f"std (non-logged): {std_value:.4f} ± {std_err:.4f}")
    print(f"Skewness: {skew_value:.4f} ± {skew_err:.4f}")
    print(f"Kurtosis: {kurt_value:.4f} ± {kurt_err:.4f}")

    return [[mu_value,mu_err],[std_value,std_err],[skew_value,skew_err],[kurt_value,kurt_err],label]

def mvsk_bootstrap(dm,dm_err,diagnostic = False):
    np.random.seed(0)
    if len(dm) > 100 and len(dm) < 1000:
        realizations = 1000
    if len(dm) < 100:
        realizations = 12000
    """Bootstrap realizations over zeroth axis, data over axis 1."""
    assert len(dm) == len(dm_err)
    n = len(dm)
    weights = 1 / (np.std(dm)**2 + dm_err**2) #use the un-weighted standard deviation, just to calculate the weight which we use later on
    iichoices = np.random.randint(low = 0, high = len(dm),size = 1000 * len(dm))

    _dm = dm[iichoices].reshape((1000,len(dm))) # sample with replacement
    _dm_err = dm_err[iichoices].reshape((1000,len(dm))) # sample with replacement
    _weights= weights[iichoices].reshape((1000,len(dm))) # sample with replacement
    w_sum = np.sum(_weights,axis = 1)
    m1 = np.sum(_weights * _dm,axis = 1) / np.sum(_weights,axis = 1)
    m2 = np.sum(_weights * (_dm - m1[:,None])**2,axis = 1) / np.sum(_weights,axis = 1)
    m3 = np.sum(_weights * (_dm - m1[:,None])**3,axis = 1) / np.sum(_weights,axis = 1)
    m4 = np.sum(_weights * (_dm - m1[:,None])**4,axis = 1) / np.sum(_weights,axis = 1)

    nval_variance = n / (n-1) * m2
    nval_skew = ((n - 1.0) * n)**0.5 / (n - 2.0) * m3 / m2**1.5
    nval_kurt = 1.0/(n-2)/(n-3) * ((n**2-1.0)*m4/m2**2.0 - 3*(n-1)**2.0)
    nval_kurt -= 3
    nval_kurt - 3, weights
    
    mm = np.mean(m1)
    mm_e = np.std(m1)
    
    ss = np.mean(np.sqrt(nval_variance))
    ss_e = np.std(np.sqrt(nval_variance))
    
    gg = np.mean(nval_skew)
    gg_e = np.std(nval_skew)
    
    kk = np.mean(nval_kurt)
    kk_e = np.std(nval_kurt)
    if diagnostic:
        f,axs = plt.subplots(nrows = 1, ncols = 4,figsize = (12,4))
        axs[0].hist(m1);axs[0].axvline(mm);axs[0].axvline(mm - mm_e);axs[0].axvline(mm + mm_e);axs[0].set_title('mean')
        axs[1].hist(np.sqrt(nval_variance));axs[1].axvline(ss);axs[1].axvline(ss - ss_e);axs[1].axvline(ss + ss_e);axs[1].set_title('std')
        axs[2].hist(nval_skew);axs[2].axvline(gg);axs[2].axvline(gg - gg_e);axs[2].axvline(gg + gg_e);axs[2].set_title('skew')
        axs[3].hist(nval_kurt);axs[3].axvline(kk);axs[3].axvline(kk - kk_e);axs[3].axvline(kk + kk_e);axs[3].set_title('kurt')
        axs[0].set_ylabel('PDF of 12k bootstrap samples')
        
    return [[mm, mm_e], [ss,ss_e], [gg,gg_e], [kk,kk_e]]

def analyze_sample(df,label,diagnostic = None,sigma_igm = 'walker'):
    """Get moments of DM_host_restframe contained in df.
    Returns
    -------
    mvsk_answer : List
        Of the form [[mean, mean_err], [std, std_err], [skew, skew_err], [kurt, kurt_err]]
    label : str
    a label for the big plot,
    
    sample_size : int
        how many sources were used in the sample.
    """
    dm = df['DM_host_restframe']
    if sigma_igm == "walker":
        sig_igm = sigma_igm_walker2024(df['primary_z_spec'])**2 
        sig_igm = 80
    elif sigma_igm == "none":
        sig_igm = 0
    dm_err = np.sqrt(
        10**2 + (df['DM_NE2001'] * 0.15)**2 + sig_igm**2
        )
    isfinite = np.isfinite(dm.values) * np.isfinite(dm_err.values) * (dm_err.values > 0)
    mvsk_answers =  mvsk_bootstrap(dm.values[isfinite], dm_err.values[isfinite],diagnostic = diagnostic)
    [[mu,mu_e],[sig,sig_e],[gam,gam_e],[kurt,kurt_e]] = mvsk_answers
    sample_size = len(df['DM_host_restframe'])
    label = label + f" ({sample_size:.0f})"
    
    if diagnostic:
        bins = np.linspace(np.min(df['DM_host_restframe']),np.max(df['DM_host_restframe']),num = max(10,int(sample_size / 8)))
        plt.figure()
        plt.hist(df['DM_host_restframe'],bins=bins)
        dumb_mean = np.nanmean(df['DM_host_restframe'])
        dumb_std = np.nanstd(df['DM_host_restframe'])
        plt.axvline(dumb_mean,label = f'{dumb_mean:.1f}')
        plt.axvline(dumb_mean - dumb_std,label = f'mean - {dumb_std:.1f}')
        plt.axvline(dumb_mean + dumb_std,label = f'mean + {dumb_std:.1f}')
        plt.legend()
        plt.title(label)
    return mvsk_answers,label,sample_size

def mvsk_numerical_from_lognormal(m,delta_m_plus,delta_m_minus,w,delta_w_plus,delta_w_minus,label):
    """mu is in natural DM units. mu_err is also. sigma is in log units (i.e. does not matter log10 or ln). sigma_merr and sigma_perr are also in log units."""

    # Moments formulas
    def mu_real(m, w):
        return np.exp(m + (w**2) / 2)

    def var_real(m, w):
        return (np.exp(w**2) - 1) * np.exp(2 * m + w**2)

    def skewness(w):
        return (np.exp(w**2) + 2) * np.sqrt(np.exp(w**2) - 1)

    def kurtosis(w): # excess kurtosis
        return np.exp(4 * w**2) + 2 * np.exp(3 * w**2) + 3 * np.exp(2 * w**2) - 6

    # Derivatives of the moments with respect to m and w
    def dmu_dm(m, w):
        return mu_real(m, w)

    def dvar_dm(m, w):
        return 2 * np.exp(2*m+w**2) * (np.exp(w**2) - 1)
        
    def dmu_dw(m, w):
        return mu_real(m, w) * w
    
    def dvar_dw(m, w):
        return 2 * np.exp(2*m+w**2) * (2 * np.exp(w**2) - 1) * w

    def dskew_dw(w):
        ew2 = np.exp(w**2)
        return w * np.exp(2 + w**2) * (-2 + 3 * ew2) / (np.sqrt(ew2 - 1))

    def dkurt_dw(w):
        ew2 = np.exp(w**2)
        return 4 * w * ew2**2 * (3 + 3 * ew2 + 2 * ew2**2)

    # Error propagation using Gaussian propagation formula
    sigma_m = (delta_m_plus + delta_m_minus) / 2
    sigma_w = (delta_w_plus + delta_w_minus) / 2
    mu_value = mu_real(m, w)
    mu_err = np.sqrt((dmu_dm(m, w) * sigma_m)**2 + (dmu_dw(m, w) * sigma_w)**2)

    var_value = var_real(m, w)
    var_err = np.sqrt((dvar_dm(m, w) * sigma_m)**2 + (dvar_dw(m, w) * sigma_w)**2)
    std_value = np.sqrt(var_value)
    std_err = var_err / (2 * np.sqrt(var_value))

    # For skew and kurtosis do the derivatives numerically
    skew_plus = skewness(w + delta_w_plus)
    skew_value = skewness(w)
    skew_minus = skewness(w - delta_w_minus)
    skew_err = 0.5 * (skew_plus - skew_minus)
    
    kurt_plus = kurtosis(w + delta_w_plus)
    kurt_value = kurtosis(w)
    kurt_minus = kurtosis(w - delta_w_minus)
    kurt_err = 0.5 * (kurt_plus - kurt_minus)
    
    # Calculating the values of the moments in real space
    print(f"Mean (non-logged): {mu_value:.4f} ± {mu_err:.4f}; median = {np.exp(m):.4f}")
    print(f"std (non-logged): {std_value:.4f} ± {std_err:.4f}")
    print(f"Skewness: {skew_value:.4f} + {skew_plus - skew_value:.4f} - {skew_value - skew_minus:.4f}")
    print(f"Kurtosis: {kurt_value:.4f} + {kurt_plus - kurt_value:.4f} - {kurt_value - kurt_minus:.4f}")
    return [[mu_value,mu_err],[std_value,std_err],[skew_value,skew_err],[kurt_value,kurt_err],label]

def plot_survey(ax,df,z_max,**errorbar_kwargs):
    """Apply redshift cutoff, then plot keepers and non-keepers"""
    keep = not_clusters(df) 
    keep *= not_edge_on(df)
    keep *= not_scattered(df)
    keep *= not_dwarf(df)
    keep *= not_elliptical(df)
    keep *= not_low_dm(df)
    keep *= not_low_galb(df)
    grayed_out = (df['primary_z_spec'] < z_max) * ~keep
    not_gray =  (df['primary_z_spec'] < z_max) * keep
    y_err = sigma_tot(df)
    if 0: #'extinction_mags' in df.keys():
        x_err = 0.2 * df['extinction_mags']
    else:
        x_err = 0.0
    ax.errorbar(x = df['M_r'][not_gray],
                 xerr = x_err,
                 y = df['DM_host_restframe'][not_gray],yerr=y_err[not_gray],
                 mfc = 'C0',mec = 'C0',ecolor = 'C0',
                 **errorbar_kwargs)
    ax.errorbar(x = df['M_r'][grayed_out],
                 xerr = x_err,y = df['DM_host_restframe'][grayed_out],
                 yerr=y_err[grayed_out],
                 mfc = 'gray',mec = 'gray',ecolor = 'gray',
                 **errorbar_kwargs)
    print('Blue:')
    print(df.index[not_gray].values)
    #print('Gray:')
    #print(df.index[grayed_out])


def plot_zdm(df,z_max,**errorbar_kwargs):
    keep = df['primary_z_spec'] < z_max
    y_err = df['DM_NE2001'][keep] * 0.2
    plt.errorbar(x = df['primary_z_spec'][keep],
                 xerr = 0,
                 y = (df['DM'] - df['DM_YT20'] - df['DM_NE2001'])[keep],
                 yerr=y_err,
                 **errorbar_kwargs)
def label_df(df,name,xoffset=0,yoffset=0,**text_kwargs):
    plt.text(df[df.index == name]['M_r'].values[0] + xoffset,df[df.index == name]['DM_host_restframe'].values[0] + yoffset,**text_kwargs)
    if xoffset != 0 or yoffset != 0:
        plt.arrow(x = df[df.index == name]['M_r'].values[0] + xoffset, dx = -xoffset,
              y = df[df.index == name]['DM_host_restframe'].values[0] + yoffset, dy = -yoffset,length_includes_head = True,linestyle = '--')
    print('labeling', name)
def label_above_dmh(df,amax = 200,z_max = 0.2):
    keep = (df['primary_z_spec'] < z_max) 
    for name,row in df[keep].iterrows():
        if np.abs(row['DM_host_restframe']) < amax:
            plt.text(x = row['M_r'], y = row['DM_host_restframe'], s = name)

def not_clusters(df):
    return np.array([n not in ['FRB20231206A','FRB20230203A','FRB20231204A','FRB20230703A', # CHIME clusters
                  'FRB20220914A','FRB20220509G', #DSA clusters
                  #'FRB20220920A' # why is this here? removing 05/10/2025
                  'FRB20230311A', 'FRB20190303A', 'FRB20231203A', # double galaxies!
                 ] for n in df.index])



def not_edge_on(df):
    """Inclinations >70 deg rejected"""
    return np.array([n not in [
            'FRB20240201A', # Shannon
            'FRB20240310A', # Shannon 
            'FRB20231120A', #  Sharma
            'FRB20220207C', # Law
            'FRB20230203A', # this work
            'FRB20231005A', # this work
            'FRB20210603A', # Cassanelli 2023
            'FRB20231120A', # looks edge on in https://arxiv.org/pdf/2409.16964
            'FRB20240213A', # looks edge on in https://www.legacysurvey.org/viewer?ra=166.1683&dec=74.0744&layer=ls-dr9&zoom=13
            ]
            for n in df.index
           ])

def not_low_galb(df,galb_cut = 10):
    keep = np.ones(len(df)) > 0
    if 'gal_b' in df.keys():
        keep *= np.abs(df['gal_b']) > galb_cut
    keep *= np.array([n not in [
            'FRB20210405I'
            ]
         for n in df.index
          ])
    return keep

def not_low_dm(df):
    """Mohit's and DSA's Mark""" 
    return np.array([n not in ['FRB20181030A', 'FRB20181220A', 'FRB20181223C', 'FRB20190418A',
       'FRB20190425A','FRB20200120E','FRB20180814A', 'FRB20200223B',
                               #'FRB20220319D'
                              ] for n in df.index])
        
def not_scattered(df):
    """tau > 5 ms at 600 MHz, assuming alpha = -4"""
    return np.array([name not in ['FRB20230222A', # 9 ms at 600 MHz assuming 4.0 from /arc/projects/chime_frb/knimmo/fitburst_results/event_268786306/scat/results_fitburst_input_dsamp_32_268786306.json

                                  'FRB20210410D', # Caleb 2023 29.4 ms at 1 GHz
                                  'FRB20230203A', # Vishwangi: 16 ms at 600 MHz assuming 4.0
                                  'FRB20190608B' # Chittidi + 2021 says 3.3 ms at 1.28 GHz
                                  ] for name in df.index])

def not_elliptical(df):
    # Not elliptical or quiescent
    return np.array([name not in ['FRB20240209A','FRB20221012A'] for name in df.index])

def close_in(df):
    return (0 < df['primary_z_spec']) * (df['primary_z_spec'] < 0.1)

def far_out(df):
    return (0.1 < df['primary_z_spec']) * (df['primary_z_spec'] < 0.2)

def not_dwarf(df):
    return (9 < df['published_M*']) * (-17 > df['M_r'])
