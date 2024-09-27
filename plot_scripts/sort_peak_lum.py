import os
import pandas as pd
import numpy as np
from astropy.cosmology import Planck18

final = pd.read_csv(os.path.join('/arc/home/calvin/kko_host_paper','data_products/kko_full_cat.csv'))
final.set_index('chime_event_id',inplace=True)
_zclean = np.clip(final['primary_z_spec'],a_min = 0, a_max = 1)
lumdist = Planck18.luminosity_distance(_zclean)
lumdist[lumdist == 0] = -1
final['peak_lum_jy_gpc2'] = final['flux'] * 4 * np.pi* lumdist**2 / 1e6

final_sorted = final.sort_values(by='peak_lum_jy_gpc2')

final_sorted[(final_sorted['primary_P_Ox'] > 0.9) * (0 < final['primary_z_spec']) * (final['primary_z_spec'] < 1)].index

final_sorted['peak_lum_jy_gpc2']

good_zspec = (0 < final['primary_z_spec']) * (final['primary_z_spec'] < 1)
for ii in final_sorted[(final_sorted['primary_P_Ox'] > 0.9) ]['name'].values:
    print("\includegraphics[width=0.22\\textwidth]{FRB " + ii[3:] + "_wfall.jpg}")
