from scipy.special import erf
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import Normalize
from matplotlib import cm
import numpy as np
from scipy.special import erfc, erfcx, log_ndtr

def _log_likelihood_ul(x):
    x /= np.sqrt(2)
    """Corrected stable version"""
    # For positive x, the original expression is generally stable
    if np.isscalar(x):
        if x > 0:
            return np.log(0.5 * (1 + erf(x)))
        else:
            # For negative x, use the relationship between erf and standard normal CDF
            # Φ(x) = 0.5 * (1 + erf(x/√2))
            # So 0.5 * (1 + erf(x)) = Φ(x√2)
            # Therefore log(0.5 * (1 + erf(x))) = log_ndtr(x√2)
            return log_ndtr(x * np.sqrt(2))
    else:
        # Handle array input
        result = np.zeros_like(x, dtype=float)
        pos_mask = x > 0
        result[pos_mask] = np.log(0.5 * (1 + erf(x[pos_mask])))
        result[~pos_mask] = log_ndtr(x[~pos_mask] * np.sqrt(2))
        return result
        
def _log_likelihood_ul_ref(argument):
    # reference implementation, not numerically stable though
    return np.log(0.5 * (1 + erf(argument / np.sqrt(2))))

def _log_likelihood_exact(argument):
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
    if np.isnan(model_err).any():
        print(model_err)
        print(res['num_sampled_sightlines'])
        model_err[np.isnan(model_err)] = np.inf
    argument = (dm - model) / np.sqrt(model_err**2 + dm_err**2) # argument of the erf. Likelihood large (model allowed) when argument > 0
    
    if mode == 'ul':
        _log_likelihood = _log_likelihood_ul
    elif mode == 'exact':
        _log_likelihood = _log_likelihood_exact
    indices_weird = np.where(argument < -5)
    if verbose:
        print('Double check these:',indices_weird)
        print(argument)
        plt.figure()
        plt.plot(argument,marker = 'o')
        plt.setp(plt.gca().get_xticklabels(), rotation=90, ha='center', rotation_mode='anchor')
        plt.grid(True)
        plt.ylabel(r'Normalized residual ($\chi$)')
        add_secondary_yaxis(plt.gca(),_log_likelihood,lambda x: f"{x:.2f}",label = 'ln(P)')
        plt.tight_layout()
    
    log_ell = _log_likelihood(argument)

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

good = _log_likelihood_ul(np.linspace(-10,10))
bad = _log_likelihood_ul_ref(np.linspace(-10,10))
assert np.sum(np.abs(good - bad) > 0.01) < 12

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

def plot_fig2(fig,ax,log_likelihoods, xnames, ynames ,out_file, cbar_label = 'Ln(posterior)',vmin = None, vmax = None):
    # Find global min and max for consistent color scale
    log_likelihoods_norm = log_likelihoods - np.max(log_likelihoods,axis = 0)
    if vmin is None:
        vmin = np.clip(np.min(log_likelihoods_norm),a_min = -100, a_max = 0)
    if vmax is None:
        vmax = np.clip(np.max(log_likelihoods_norm),a_min = vmin, a_max = 0)
    
    norm = Normalize(vmin=vmin, vmax=vmax)
    shape = (3,int(log_likelihoods_norm.size // 3))
    model_data_norm = log_likelihoods_norm.reshape(shape)
    model_data = log_likelihoods.reshape(shape)
    ynames = ynames.flatten()
    # Create imshow plot
    im = ax.imshow(model_data_norm.T, cmap='inferno', norm=norm,aspect = 'auto')
    
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
    plt.savefig(out_file,dpi = 110,bbox_inches= 'tight')
