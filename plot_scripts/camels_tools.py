import os
import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
from glob import glob
from matplotlib.pyplot import cm
from matplotlib import patches

"""Ok, here are the sightlines at z = 0.0 (090), z=0.05 (088), z =0.10 (086),  z =0.15 (084), and z =0.2 (082)"""

ROOT_PATH = os.getcwd() + '/../'
CAMELS_REDSHIFTS = {'090':0.0,'088':0.05,'086':0.1,'084':0.15,'082':0.2}
def load_snapshot(filename):
    """Load snapshot and assign a redshift"""
    def get_redshift(filename):
        for key in CAMELS_REDSHIFTS:
            if key in filename:
                return CAMELS_REDSHIFTS[key]
    df = pd.read_csv(filename,names=['x_pos', 'y_pos', 'dm_total', 'b_val', 'halo_index', 'M_star', 'M_200', 'sfr'])
    df['z'] = get_redshift(filename) + np.zeros_like(len(df))
    return df
        
def analyze_halos(
    filepath: str,
    df: pd.DataFrame,
    num_halos: int,
    b_min: float,
    b_max: float,
    M_200_min: float,
    M_200_max: float,
    M_star_range: np.array,
    width_factor: 2,
    sfr_min: float,
    sfr_max: float,
    out_file: str = None,
    random_seed: int = 42,
    ):
    """
    Analyze halos from a CSV file with filtering and weighting.
    
    Parameters:
    -----------
    df : DataFrame
        To be analyzed
    num_halos : int
        Number of halos to randomly sample
    b_min, b_max : float
        Min and max impact parameter values to filter
    M_200_min, M_200_max : float
        Min and max M_200 values to filter
    sfr_min, sfr_max : float
        Min and max SFR values to filter
    random_seed : int, optional
        Random seed for reproducibility
        
    Returns:
    --------
    results : dict
        Dictionary containing analysis results
    sampled_data : pd.DataFrame
        The sampled data that was used for analysis
    """
    # Set random seed for reproducibility
    #np.random.seed(random_seed)
    
    # Filter the data based on given criteria
    n_bins = len(M_star_range)
    counts = np.zeros((4,len(M_star_range)))
    weighted_mean = np.zeros((4,len(M_star_range)))
    weighted_std = np.zeros_like(weighted_mean)
    M_star_bins = []
    assert width_factor > 1, "width_dex must be greater than 1!"
    for iibin,M_star in zip(np.arange(n_bins),M_star_range):
        M_star_min = M_star / width_factor
        M_star_max = M_star * width_factor
        filtered_df = df[
            (df['b_val'] >= b_min) & 
            (df['b_val'] <= b_max) & 
            (df['M_200'] >= M_200_min) & 
            (df['M_200'] <= M_200_max) & 
            (df['sfr'] >= sfr_min) & 
            (df['sfr'] <= sfr_max) &
            (df['M_star'] >= M_star_min) &
            (df['M_star'] <= M_star_max)
        ]
        sampled_data = filtered_df.set_index('halo_index').groupby('halo_index').apply(lambda x: x.sample(n=1,)).reset_index(drop=True)
        assert len(sampled_data.index) == len(set(sampled_data.index)),'NOT unique!'
        if len(sampled_data) < num_halos:
            print(
                f"Not enough halos after filtering (bin index={iibin})"
                )
            print(f'sightlines within halos: {len(filtered_df)}, unique halos: {len(set(filtered_df['halo_index']))}, sampled sightlines: {len(sampled_data.index)}')
        # 2. Sample: one sightline per halo, not one per unit area.
        M_star_bins.append([M_star_min,M_star_max])
        # 3. Compute weightings
        # Normalize the weights so they sum to 1
        weightings = ['uniform','M_star','sfr','both']
        for iiweighting, weighting in enumerate(weightings):
            if weighting == 'uniform':
                weights = np.ones_like( sampled_data['M_star']) / len( sampled_data['M_star'])
            if weighting == 'M_star':
                weights = sampled_data['M_star'] / sampled_data['M_star'].sum()
            if weighting == 'sfr':
                weights = sampled_data['sfr'] / sampled_data['sfr'].sum()
            if weighting == 'both':
                weights = sampled_data['sfr'] * sampled_data['M_star'] / (sampled_data['sfr'] * sampled_data['M_star']).sum()
    
            # 4. Apply weights to the DM distribution
            # For weighted mean: sum(weight * value) / sum(weights)
            # For weighted variance: sum(weight * (value - weighted_mean)^2) / sum(weights)
            _weighted_mean = np.sum(weights * sampled_data['dm_total'])
            _weighted_var = np.sum(weights * (sampled_data['dm_total'] - _weighted_mean)**2)
            weighted_mean[iiweighting,iibin] = _weighted_mean
            weighted_std[iiweighting,iibin] = np.sqrt(_weighted_var)
            counts[iiweighting,iibin] = len(sampled_data)
        
    # 5. Compile results
    results = {'filepath': filepath,
            'weighted_mean': weighted_mean,
            'weighted_std': weighted_std,
            'M_star_bins': M_star_bins,
            'num_sampled_sightlines': counts,
            'b_range': np.array([b_min, b_max]),
            'M_200_range': np.array([M_200_min, M_200_max]),
            'sfr_range': np.array([sfr_min, sfr_max])
                }
    if out_file is not None:
        import h5py
        with h5py.File(out_file,'w') as f:
            for key, val in results.items():
                if key == 'filepath':
                    f.attrs['filepath'] = filepath
                else:
                    f.create_dataset(key,data=val)
    # 6. Write back to file system
    return results,sampled_data

def from_file(filename):
    results = {}
    with h5py.File(filename,'r') as f:
        for key in f.keys(): 
            if key == 'filepath':
                results['filepath'] = f.attrs['filepath']
            else:
                results[key] = f[key][:].copy()
    return results
            
def plot_theory_predictions(path_to_file,ax,x_axis = 'M_star',weight = 'weighted',one_halo = True,label = None,**rect_kwargs):
    results = from_file(path_to_file)
    M_star_bins = np.array(results['M_star_bins'])
    x_vals = np.sqrt(M_star_bins[:,0] * M_star_bins[:,1])
    y_vals = results['weighted_mean'][0,:] / 2 # factor of 2 for 1 halo term
    y_errs = results['weighted_std'][0,:] / 2 # factor of 2 for 1 halo term
    min_max_cen_std = np.vstack((np.array(results['M_star_bins']).T,y_vals, y_errs)).T
    plot_rectangles(ax,min_max_cen_std,**rect_kwargs)
    plt.scatter(x_vals,y_vals,**rect_kwargs,label = label)
    return results
def plot_rectangles(ax,rect_data, **rect_kwargs):
    """
    Plot multiple semi-transparent rectangles from a list of tuples.
    
    Parameters:
    ----------
    rect_data : list of tuples
        Each tuple contains (x_min, x_max, y_center, y_std)
    colors : list or None
        List of colors for each rectangle. If None, uses a color cycle
    **rect_kwargs : 
        Goes into patches.Rectangle(...**rect_kwargs)
    """
    
    # Use default color cycle if colors not provided    
    for i, (x_min, x_max, y_center, y_std) in enumerate(rect_data):
        # Calculate rectangle parameters
        width = x_max - x_min
        height = 2 * y_std
        y_min = y_center - y_std
        
        # Create rectangle patch with cycling colors
        rect = patches.Rectangle(
            (x_min, y_min), width, height,
            **rect_kwargs
        )
        
        # Add rectangle to axis
        ax.add_patch(rect)
        
        # Optional: Add text label in the center of each rectangle
        # ax.text(x_min + width/2, y_center, f"{i+1}", ha='center', va='center', fontsize=10)
    return ax

def visualize_results(results: dict, sampled_data: pd.DataFrame) -> None:
    """
    Visualize the results of the halo analysis.
    
    Parameters:
    -----------
    results : dict
        Dictionary containing analysis results
    sampled_data : pd.DataFrame
        The sampled data that was used for analysis
    """
    results = {'filepath': filepath,
            'weighted_mean': weighted_mean,
            'weighted_std': weighted_std,
            'M_star_bins': M_star_bins,
            'num_sampled_halos': counts,
            'sampled_': len(sampled_data) / len(filtered_df),
            'b_range': (b_min, b_max),
            'M_200_range': (M_200_min, M_200_max),
                'sfr_range': (sfr_min, sfr_max)
                }
    # Set up a figure with subplots
    fig, axs = plt.subplots(3,2, figsize=(14, 14))
    
    # Plot 1: Histogram of dm_total with both weighted and unweighted statistics
    title = f"N = {results['num_filtered_halos']} halos sampled uniformly in sightline"
    fig.suptitle(title)
    ax = axs[0, 0]
    ax.hist(sampled_data['dm_total'], bins=20, alpha=0.7, color='skyblue', weights = np.ones_like(sampled_data['weight']),density=True)
    ax.hist(sampled_data['dm_total'], bins=20, alpha=0.7, color='red', weights = sampled_data['weight'],density=True)
    ax.axvline(results['unweighted_mean'], color='blue', linestyle='-', 
               label=f"Unweighted Mean: {results['unweighted_mean']:.2f}")
    ax.axvline(results['weighted_mean'], color='red', linestyle='-', 
               label=f"Weighted Mean: {results['weighted_mean']:.2f}")
    
    # Add shaded regions for standard deviations
    ax.axvspan(
        results['unweighted_mean'] - results['unweighted_std'],
        results['unweighted_mean'] + results['unweighted_std'],
        alpha=0.2, color='blue'
    )
    ax.axvspan(
        results['weighted_mean'] - results['weighted_std'],
        results['weighted_mean'] + results['weighted_std'],
        alpha=0.2, color='red'
    )
    ax.set_xlabel('DM Total')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of DM Total')
    ax.legend()
    
    # Plot 2: Scatter plot of M_star vs dm_total
    ax = axs[0, 1]
    ax.set_xscale('log')
    ax.set_yscale('log')
    sc = ax.scatter(sampled_data['M_star'], sampled_data['sfr'], 
                   c=sampled_data['weight'], cmap='viridis', alpha=0.7)
    ax.set_xlabel('M_star')
    ax.set_ylabel('SFR')
    ax.set_title('M_star vs DM Total (color = weight)')
    plt.colorbar(sc, ax=ax, label='Weight')
    
    # Plot 3: Scatter plot of M_200 vs dm_total
    ax = axs[1, 0]
    ax.set_xscale('log')
    ax.scatter(sampled_data['M_200'], sampled_data['dm_total'], alpha=0.7)
    ax.set_xlabel('M_200')
    ax.set_ylabel('DM Total')
    ax.set_title('M_200 vs DM Total')
    
    # Plot 4: Scatter plot of impact parameter vs dm_total
    ax = axs[1, 1]
    ax.scatter(sampled_data['b_val'], sampled_data['dm_total'], alpha=0.7)
    ax.set_xlabel('Impact Parameter (b_val)')
    ax.set_ylabel('DM Total')
    ax.set_title('Impact Parameter vs DM Total')

    
    # Plot 5: Scatter plot of SFR vs dm_total
    ax = axs[2, 1]
    ax.set_xscale('log')
    sc = ax.scatter(sampled_data['sfr'], sampled_data['dm_total'], 
                   c=sampled_data['weight'], cmap='viridis', alpha=0.7)
    ax.set_xlabel('SFR')
    ax.set_ylabel('DM Total')
    ax.set_title('SFR vs DM Total (color = weight)')
    plt.colorbar(sc, ax=ax, label='Weight')

    # Plot 5: Scatter plot of M* vs dm_total
    ax = axs[2, 0]
    ax.set_xscale('log')
    sc = ax.scatter(sampled_data['M_star'], sampled_data['dm_total'], 
                   c=sampled_data['weight'], cmap='viridis', alpha=0.7)
    ax.set_xlabel('M_star')
    ax.set_ylabel('DM Total')
    ax.set_title('M* vs DM Total (color = weight)')
    plt.colorbar(sc, ax=ax, label='Weight')
    
    
    plt.tight_layout()
    plt.show()

def plot_theory_predictions_deprecated(path_to_file,ax,x_axis = 'M_star',weight = 'weighted',one_halo = True,label = None,**rect_kwargs):
    results_list = np.load(path_to_file,allow_pickle=True)['results']
    x_vals = []
    y_vals = []
    for results in results_list:
        x_min,x_max = results['filter_criteria'][x_axis + '_range']
        x_center = np.sqrt(x_min * x_max)
        y_center = results[weight + '_mean']
        y_std = results[weight + '_std']
        if one_halo:
            y_center /= 2 # on average going through half the halo
            y_std /= 2    # on average going through half the halo
        width = x_max - x_min
        height = 2 * y_std  # Height is 2 times y_std (±y_std from center)
        y_min = y_center - y_std  # Bottom edge
        
        # Create rectangle patch
        rect = patches.Rectangle((x_min, y_min), width, height, **rect_kwargs)
        
        # Add rectangle to the axis
        ax.add_patch(rect)
        x_vals.append(x_center)
        y_vals.append(y_center)
    
    x_vals = np.array(x_vals)
    y_vals = np.array(y_vals)
    print('_'.join([str(results['num_filtered_halos']) for results in results_list]))

    ax.scatter(x_vals,y_vals,label = label,color = rect.get_facecolor(),alpha = 1,linestyle = '-')
