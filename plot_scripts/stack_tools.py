from scipy.special import erf
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import Normalize
from matplotlib import cm
import numpy as np
from scipy.special import erfc, erfcx, log_ndtr


import numpy as np
from scipy.special import log_ndtr
import scipy.stats as stats

def _compute_log_integral_result(a, b, w_i, w_c_array, d_array):
    """
    Computes the natural logarithm of the integral result in a vectorized way.

    Here the integral is

    int from c=0 to c=infinity of:
    ******
    exp(- 0.5 * wi**2 * (a - b - c)**2 - 0.5 * wc**2 * (c - d)**2)
    ******

    Parameters:
    -----------
    a : float
        Parameter a in the integral
    b : float
        Parameter b in the integral
    w_i : float
        Parameter w_i in the integral (fixed value)
    w_c_array : numpy.ndarray
        Array of w_c values
    d_array : numpy.ndarray
        Array of d values

    Returns:
    --------
    numpy.ndarray
        2D array of shape (len(w_c_array), len(d_array)) containing log of integral results
    """
    # Reshape arrays for broadcasting
    w_c = w_c_array.reshape(-1, 1)  # Shape: (n_w_c, 1)
    d = d_array.reshape(1, -1)      # Shape: (1, n_d)

    # Pre-calculate common terms
    ab_diff = a - b
    w_i_sq = w_i**2
    w_c_sq = w_c**2

    # Calculate terms for the exponent
    term1 = -0.5 * w_i_sq * ab_diff**2
    term2 = -0.5 * w_c_sq * d**2

    # Calculate the numerator for term3
    numerator = (w_i_sq * ab_diff + w_c_sq * d)**2

    # Calculate the denominator for term3
    denominator = 2 * (w_i_sq + w_c_sq)

    # Calculate term3
    term3 = numerator / denominator

    # Combined exponent
    exponent = term1 + term2 + term3

    # Calculate the error function argument
    erf_arg = (w_i_sq * ab_diff + w_c_sq * d) / np.sqrt(w_i_sq + w_c_sq)
    #erf_arg = (ab_diff + d) / np.sqrt(w_i_sq**-1 + w_c_sq**-1)

    # For numerical stability, we use the log of the CDF of the normal distribution
    # which is related to erf by: erf(x/sqrt(2)) = 2*CDF(x) - 1
    # So log(0.5 + 0.5*erf(x)) = log(CDF(x*sqrt(2)))
    scaled_erf_arg = erf_arg * np.sqrt(2)

    # Use log_ndtr for numerical stability (log of normal CDF)
    log_erf_term = log_ndtr(scaled_erf_arg)

    # Calculate the logarithm of the coefficient
    log_coef = 0.5 * np.log(np.pi) - 0.5 * np.log(w_i_sq + w_c_sq) + 0.5 * np.log(w_i_sq) + 0.5 * np.log(w_c_sq)

    # Combine all terms to get the final logarithm of the result
    # log(exp(exponent) * coefficient * (0.5 + 0.5*erf(...)))
    # For log(0.5 + 0.5*erf(...)), we use the approach from above
    result = exponent + log_coef + log_erf_term

    return result

def log_likelihood_mgaussian(dm_host_restframe,
        sigma_i,
        dm_model,
        sigma_cbm_arr,
        dm_cbm_arr,
):
    """Returns np.log(P(DM_i | <DM>, <sigma_cosmic>)) as a np.array of shape (n_<dm>, n_<sigmacosmic>) """
    return _compute_log_integral_result(a = dm_host_restframe,
                                        b = dm_model,
                                        w_i = sigma_i**-1,
                                        w_c_array = sigma_cbm_arr**-1,
                                        d_array = dm_cbm_arr,
                                       )

def _log_likelihood_ul(dm_data, dm_model,sigma_data, sigma_model, sigma_e = 0):
    """Returns log likelihood marginalized over DM_cbm uniform over [0,inf).
    
    Assumes Gaussian likelihood in DM_cbm.

    Parameters
    ----------
    dm_data : np.array of shape (n_frb),
    dm_model : np.array of shape (n_frb),
    sigma_data : np.array of shape (n_frb),
    sigma_model : np.array of shape (n_frb),
    sigma_e : float or np.array of shape (n_frb),

    Returns
    -------
    result : np.array of shape (n_frb,)
        the natural log likelihood, marginalized over dm_cbm.
    """
    x = (dm_data - dm_model) / (sigma_data**2 + sigma_model**2 + sigma_e**2)**0.5
    x /= np.sqrt(2)
    """Corrected stable version"""
    unc = (sigma_data**2 + sigma_model**2 + sigma_e**2)**0.5
    # For positive x, the original expression is generally stable
    if np.isscalar(x):
        if x > 0:
            return np.log(0.5 * (1 + erf(x))) - np.log(2 * np.pi * unc**2)
        else:
            # For negative x, use the relationship between erf and standard normal CDF
            # Φ(x) = 0.5 * (1 + erf(x/√2))
            # So 0.5 * (1 + erf(x)) = Φ(x√2)
            # Therefore log(0.5 * (1 + erf(x))) = log_ndtr(x√2)
            return log_ndtr(x * np.sqrt(2)) - np.log(2 * np.pi * unc**2)
    else:
        # Handle array input
        result = np.zeros_like(x, dtype=float)
        pos_mask = x > 0
        result[pos_mask] = np.log(0.5 * (1 + erf(x[pos_mask])))
        result[~pos_mask] = log_ndtr(x[~pos_mask] * np.sqrt(2)) 
        result -= np.log(2 * np.pi * unc)
        return result
        
def _log_likelihood_ul_ref(dm_data, dm_model,sigma_data, sigma_model, sigma_e = 0):
    argument = (dm_data - dm_model) / (sigma_data**2 + sigma_model**2 + sigma_e**2)**0.5
    unc = (sigma_data**2 + sigma_model**2 + sigma_e**2)**0.5
    # reference implementation, not numerically stable though
    return np.log(0.5 * (1 + erf(argument / np.sqrt(2)))) - np.log(2 * np.pi * unc**2)

def _log_likelihood_ul_nuisance(dm_data, dm_model,sigma_data, sigma_model,first = 'sigma'):
    """Integrates over the width parameter, not just location parameter"""
    sigma_e = np.linspace(0,300,num = 50)
    dsigma_e = np.diff(sigma_e)[0]
    ll_per_burst_per_sigma = np.zeros((len(dm_data),len(sigma_e)))
    prior_vs_sigma = np.log(sigma_e[None,:] * ((sigma_data**2 + sigma_model**2)[:,None] + sigma_e[None,:]**2)**-1) # shape: (n_bursts, n_sigma_e)
    for iie, se in enumerate(sigma_e):
        ll_per_burst_per_sigma[:,iie] = _log_likelihood_ul(dm_data, dm_model,sigma_data, sigma_model, sigma_e = se)
    ll_per_burst_per_sigma = ll_per_burst_per_sigma + prior_vs_sigma
    norm = np.max(ll_per_burst_per_sigma)
    if first == 'sigma': # marginalize over sigma first, on a per-burst level
        return np.log(dsigma_e * np.sum(np.exp(ll_per_burst_per_sigma),axis = -1)) 
    elif first == 'burst': # combine likelihoods over bursts first, then marginalize over sigma.
        likelihood_per_sigma = np.exp(np.sum(ll_per_burst_per_sigma,axis = 0))
        return np.log(np.sum(dsigma_e * likelihood_per_sigma))

def _log_likelihood_exact(dm_data, dm_model,sigma_data, sigma_model, sigma_e = 0):
    """Returns np.array of shape (n_frb,) of log-liklihoods"""
    argument = (dm_data - dm_model) / (sigma_data**2 + sigma_model**2 + sigma_e**2)**0.5
    return -0.5 * argument**2

def log_likelihood(ms,dm,dm_err,res,weight_index = 0,verbose = False,mode = 'ul'):
    """Computes log likelihood

    Returns an array whose shape is the same as ms."""
    msb = np.array(res['M_star_bins'])
    bin_centers = np.sqrt(msb[:,0] * msb[:,1])
    indices = np.zeros_like(ms,dtype = int)
    for ii,_ms in enumerate(ms):
        indices[ii] = np.argmin(np.abs(bin_centers - _ms))
    model = res['weighted_mean'][weight_index,[indices]].squeeze() # fix model weighting scheme, then choose correct bin to compare each data point against
    model_err = res['weighted_std'][weight_index,[indices]].squeeze() # fix model weighting scheme, then choose correct bin to compare each data point against
    dm = np.array(dm)
    dm_err = np.array(dm_err)
    model = np.array(model)
    model_err = np.array(model_err)
    if np.isnan(model_err).any():
        print(model_err)
        print(res['num_sampled_sightlines'])
        model_err[np.isnan(model_err)] = np.inf
    if mode == 'ul':
        "Using UL likelihood, not marginalizing over sigma_source"
        _log_likelihood = _log_likelihood_ul
        log_ell = _log_likelihood_ul(dm_data = dm,
                              sigma_data = dm_err,
                              dm_model = model,
                              sigma_model = model_err)
    elif mode == 'exact':
        "Using Gaussian likelihood"
        _log_likelihood = _log_likelihood_exact
        log_ell = _log_likelihood_exact(dm_data = dm,
                              sigma_data = dm_err,
                              dm_model = model,
                              sigma_model = model_err)
    elif mode == 'nuisance':
        "Using UL likelihood, marginalizing over sigma_source"
        log_ell = _log_likelihood_ul_nuisance(dm_data = dm,
                              sigma_data = dm_err,
                              dm_model = model,
                              sigma_model = model_err,
                              first=  'sigma')
    elif mode == 'nuisance2':
        "Using UL likelihood, NOT marginalizing over sigma_source"
        log_ell = _log_likelihood_ul_nuisance(dm_data = dm,
                              sigma_data = dm_err,
                              dm_model = model,
                              sigma_model = model_err,
                              first=  'burst')
    if verbose:
        indices_weird = np.where(log_ell < -20)
        print('Double check these:',indices_weird)
        print(argument)
        plt.figure()
        plt.plot(argument,marker = 'o')
        plt.setp(plt.gca().get_xticklabels(), rotation=90, ha='center', rotation_mode='anchor')
        plt.grid(True)
        plt.ylabel(r'Normalized residual ($\chi$)')
        add_secondary_yaxis(plt.gca(),_log_likelihood,lambda x: f"{x:.2f}",label = 'ln(P)')
        plt.tight_layout()
    

    return log_ell
    
def add_secondary_yaxis(ax, conversion_func, format_func, label=None, color="tab:red"):
    """
    Add secondary y-axis tick labels based on a conversion function.
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The primary axis object
    conversion_func : callable
        Function to convert primary y values to secondary units
    format_func : callable
        Function to format the secondary tick labels (takes a value, returns a string)
    label : str, optional
        Label for the secondary y-axis
    color : str, optional
        Color for the secondary axis labels and ticks
        
    Returns:
    --------
    ax2 : matplotlib.axes.Axes
        The secondary axis object
    """
    # Create secondary axis with the same position as primary
    ax2 = ax.twinx()
    
    # Get the primary axis tick positions
    primary_ticks = ax.get_yticks()
    
    # Convert the primary ticks to secondary values
    secondary_values = conversion_func(primary_ticks)
    
    # Set tick positions and format the labels
    ax2.set_yticks(primary_ticks)
    ax2.set_yticklabels([format_func(val) for val in secondary_values])
    
    # Set limits to match the primary axis
    ax2.set_ylim(ax.get_ylim())
    
    # Set optional label and color
    if label:
        ax2.set_ylabel(label, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    
    return ax2
good = _log_likelihood_ul(
    dm_data = np.linspace(-10,10), 
    dm_model = 0, 
    sigma_data=  1,
    sigma_model = 1 
    )
bad = _log_likelihood_ul_ref(
    dm_data = np.linspace(-10,10), 
    dm_model = 0, 
    sigma_data =  1,
    sigma_model = 1 
    )
_keep = np.isnan(good + bad)
assert np.isclose(good[_keep],bad[_keep]).all()

def log_likelihood_hp_gauss(ms,dm,dm_err,res,weight_index = 0,verbose = False,mode = 'g2',sigma_cbm_arr = np.linspace(1,100,num = 10),
                                dm_cbm_arr = np.linspace(0,200, num = 30)):
    """Computes log likelihood with proper treatment of the hyperparameters (hence, 'hp' in the name).

    Returns an array whose shape is the same as ms."""
    msb = np.array(res['M_star_bins'])
    bin_centers = np.sqrt(msb[:,0] * msb[:,1])
    indices = np.zeros_like(ms,dtype = int)
    for ii,_ms in enumerate(ms):
        indices[ii] = np.argmin(np.abs(bin_centers - _ms))
    model = res['weighted_mean'][weight_index,[indices]].squeeze() # fix model weighting scheme, then choose correct bin to compare each data point against
    model_err = res['weighted_std'][weight_index,[indices]].squeeze() # fix model weighting scheme, then choose correct bin to compare each data point against
    dm = np.array(dm)
    dm_err = np.array(dm_err)
    model = np.array(model)
    model_err = np.array(model_err)
    ll_per_burst_gaussian = np.zeros((len(dm),len(sigma_cbm_arr),len(dm_cbm_arr)))
    for (ii,_dm,_dme,_m,_me) in zip(np.arange(dm.size),dm,dm_err,model,model_err):
        ll_per_burst_gaussian[ii] = log_likelihood_mgaussian(
            dm_host_restframe = _dm,
            sigma_i = (_me**2 + _dme**2)**0.5,
            dm_model = _m,
            sigma_cbm_arr = sigma_cbm_arr,
            dm_cbm_arr = dm_cbm_arr,
        )
    return ll_per_burst_gaussian

def marginalize_ll_hp(ll,prior,dm_cbm_arr,sigma_cbm_arr):
    """Combines likelihoods over bursts, then integrates over hyperparameter priors (sigma_cbm_arr,dm_cbm_arr)

    Parameters
    ----------
    ll : np.array of shape (n_bursts, n_sigma_cbm_arr, n_dm_cbm_arr)
        The natural log likelihood
    prior : np.array of shape (n_sigma_cbm_arr, n_dm_cbm_arr)

    keep : np.array of bools of shape (n_bursts,)

    Returns
    -------
    marg : np.array of shape (ll.shape[0],), usually (3,)
        Marginalized log-likelihood
    """
    ddm = np.abs(np.diff(dm_cbm_arr,append = dm_cbm_arr[-1]))
    dsigma = np.abs(np.diff(sigma_cbm_arr,append = sigma_cbm_arr[-1]))
    norm = np.max(ll)
    ll_norm = ll - norm
    if ll_norm.ndim == 2:
        ll_norm.shape = (1,ll_norm.shape[0],ll_norm.shape[1])

    marg = np.sum(np.exp(np.sum(
                 ll_norm,axis = -3) # sum log-likelihoods over all (n_model, n_bursts, n_sigma, n_dm) -> (n_model, n_sigma, n_dm)
                        ) * #
                 prior[...,None,:,:] *
                 ddm[...,None,None,:] * dsigma[...,None,:,None],
                 axis = (-2,-1),
                 )
    return np.log(marg) + norm

def plot_fig(fig,axes,log_likelihoods, out_file,
    redshifts = ['z = 0.0', 'z = 0.1', 'z = 0.2'],
    impact_params = [r'$b_{max}$ = 0.12', r'$b_{max}$ = 0.16', r'$b_{max}$ = 0.20'],
    model_names = ['SIMBA', 'IllustrisTNG', 'Astrid'], cbar_label = 'Ln(posterior)',vmin = None, vmax = None):
    # Define labels
    if vmin is None:
        vmin = np.min(log_likelihoods)
    if vmax is None:
        vmax = np.max(log_likelihoods)
    
    # Create figure with 3 subplots (one for each model)
    
    # Find global min and max for consistent color scale
    norm = Normalize(vmin=vmin, vmax=vmax)
    
    # Loop through models (subplots)
    for model_idx, ax in enumerate(axes):
        # Get data for this model
        model_data = log_likelihoods[model_idx]
        
        # Create imshow plot
        im = ax.imshow(model_data, cmap='inferno', norm=norm)
        
        # Add value text to each cell
        for z_idx in range(3):  # redshift index
            for b_idx in range(3):  # impact parameter index
                value = model_data[z_idx, b_idx]
                ax.text(b_idx, z_idx, f'{value:.2f}', 
                        ha='center', va='center', 
                        color='white' if value < (vmin + vmax) / 2 else 'black',
                        fontweight='bold')
        
        # Set title and labels
        ax.set_title(model_names[model_idx])
        ax.set_xticks(np.arange(3))
        ax.set_yticks(np.arange(3))
        ax.set_xticklabels(impact_params)
        
        # Rotate x-axis labels for better readability
        plt.setp(ax.get_xticklabels(), rotation=0, ha='center', rotation_mode='anchor')
        
        # Add grid to make cells more visible
        ax.grid(False)
    axes[0].set_yticklabels(redshifts)
    axes[1].set_yticks([])
    axes[2].set_yticks([])
    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label(cbar_label, rotation=270, labelpad=20)
    
    # Adjust layout
    #plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    plt.subplots_adjust(wspace = 0,hspace = 0)
    plt.savefig(out_file,dpi = 110,bbox_inches= 'tight')

def plot_fig2(fig,ax,log_likelihoods, xnames, ynames ,out_file = None, cmap = 'inferno',cbar_label = 'Ln(posterior)',vmin = None, vmax = None):
    # Find global min and max for consistent color scale
    log_likelihoods = log_likelihoods - np.max(log_likelihoods,axis = 0)[None,:] # could also normalize to log_likelihoods[-1,0] # -1 -> Simba; 0 -> fiducial analysis
    log_likelihoods_norm = log_likelihoods - np.max(log_likelihoods,axis = 0)[None,:] # what you color (log_likelihoods_norm) 
    if vmin is None:
        vmin = np.clip(np.min(log_likelihoods_norm),a_min = -100, a_max = 0)
    if vmax is None:
        vmax = np.clip(np.max(log_likelihoods_norm),a_min = vmin, a_max = 3)
    
    norm = Normalize(vmin=vmin, vmax=vmax)
    shape = (3,int(log_likelihoods_norm.size // 3))
    model_data_norm = log_likelihoods_norm.reshape(shape)
    model_data = log_likelihoods.reshape(shape)
    ynames = ynames.flatten()
    # Create imshow plot
    im = ax.imshow(model_data_norm.T, cmap=cmap, norm=norm,aspect = 'auto')
    
    for iimodel in range(3):
        for jjrow in range(shape[1]):
            print_value = model_data[iimodel, jjrow]
            color_value = model_data_norm[iimodel, jjrow]
            ax.text(iimodel, jjrow, f'{print_value:.1f}', 
                    ha='center', va='center', color='white' if color_value < (vmin + vmax) / 2 else 'black',
                    fontweight='bold')
    
    # Set title and labels
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(xnames)
    
    # Rotate x-axis labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=0, ha='center', rotation_mode='anchor')
    
    # Add grid to make cells more visible
    ax.grid(False)
    ax.set_yticks(np.arange(shape[1]))
    ax.set_yticklabels(ynames)
    # Add colorbar
    # Create axis for colorbar with same width as image
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="5%", pad=0.1)
    cbar = fig.colorbar(im, cax=cax,orientation = 'horizontal')
    cbar.set_ticks(np.linspace(vmin, vmax,num = 5))
    cbar.set_label(cbar_label)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    if out_file is not None:
        plt.savefig(out_file,dpi = 110,bbox_inches= 'tight')

import dmh_tools
def fit_linear_with_uncertainties(df):
    """
    Fit a linear model to data with uncertainties in y.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataframe containing the data to fit
    dmh_tools : module
        Module containing the sigma_tot function for error calculation
    
    Returns:
    --------
    tuple
        (slope, slope_err, intercept, intercept_err)
    """
    # Extract x and y data
    x_data = df['published_M*'].values
    x_mean = np.mean(x_data)
    print('Subtracting mean:',x_mean)
    x_data -= x_mean
    y_data = df['DM_host_restframe'].values
    
    # Calculate y uncertainties using provided function
    y_errors = dmh_tools.sigma_tot(df)
    
    # Define linear model
    def linear_model(x, slope, intercept):
        return slope * x + intercept
    
    # Perform the weighted fit using scipy.optimize.curve_fit
    # absolute_sigma=True ensures proper error propagation
    params, covariance = curve_fit(
        linear_model, 
        x_data, 
        y_data, 
        sigma=y_errors, 
        absolute_sigma=True
    )
    
    # Extract the parameters and their uncertainties
    slope, intercept = params
    slope_err, intercept_err = np.sqrt(np.diag(covariance))
    
    # Return the results
    return {
        'slope': slope,
        'slope_err': slope_err,
        'intercept': intercept,
        'intercept_err': intercept_err,
        'x_mean' : x_mean,
    }

def get_eqn_string(mxpb):
    eqn_string = f"\\dmhm &= ({mxpb['slope']:.0f}"+ " \\pm" + f"{mxpb['slope_err']:.0f})"
    eqn_string += f"\\log(M^*/10^" + '{'
    eqn_string += f"{mxpb['x_mean']:.1f}" + '} M_\odot)'
    eqn_string += f" + ({mxpb['intercept']:.0f}" + "\\pm" + f"{mxpb['intercept_err']:.0f})"
    return eqn_string
    
