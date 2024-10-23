from glob import glob
import numpy as np

import pandas as pd
import os
import json
from frb.surveys.panstarrs import Pan_STARRS_Survey
from frb.surveys.decals import DECaL_Survey
from astropy.coordinates import SkyCoord
from astropy import units as u

import astropy.units as u
import pyne2001
import astropy.coordinates as coord
from frb.surveys.decals import DECaL_Survey
from frb.surveys.panstarrs import Pan_STARRS_Survey


def get_ne2001(c_icrs,get_scatt = True):
    dm_ne2001 = np.zeros(len(c_icrs))
    tau600_ne2001 = np.zeros(len(c_icrs))

    for ii,(gl,gb) in enumerate(zip(c_icrs.galactic.l,c_icrs.galactic.b)):
        ne2001_pred = pyne2001.get_dm_full(gl.value,gb.value,dist_kpc = 100.0)
        #print(f"dist: {ne2001_pred['DIST']}")
        dm_ne2001[ii] = ne2001_pred['DM']
        tau600_ne2001[ii] = ne2001_pred['TAU'] * (1/0.6)**4
    if get_scatt:
        return dm_ne2001,tau600_ne2001
    else:
        return dm_ne2001
def calculate_halo_dm_yt20(l, b):
    coeffs = np.array(
        [
            [250.12, -871.06, 1877.5, -2553.0, 2181.3, -1127.5, 321.72, -38.905],
            [-154.82, 783.43, -1593.9, 1727.6, -1046.5, 332.09, -42.815, 0],
            [-116.72, -76.815, 428.49, -419.00, 174.60, -27.610, 0, 0],
            [216.67, -193.30, 12.234, 32.145, -8.3602, 0, 0, 0],
            [-129.95, 103.80, -22.800, 0.44171, 0, 0, 0, 0],
            [39.652, -21.398, 2.7694, 0, 0, 0, 0, 0],
            [-6.1926, 1.6162, 0, 0, 0, 0, 0, 0],
            [0.39346, 0, 0, 0, 0, 0, 0, 0],
        ]
    )
    l_absolute = np.abs(np.deg2rad(l))
    b_absolute = np.abs(np.deg2rad(b))
    DM_halo = 0
    for ii in range(8):
        for jj in range(8):
            DM_halo += coeffs[ii, jj] * l_absolute ** ii * b_absolute ** jj
    return DM_halo # u.pc / u.cm ** 3

# Load in final ellipse catalog

final = pd.read_csv('/arc/home/calvin/kko_host_paper/data_products/kko_ellipse_cat_20240919.csv',
        names = ['name','chime_event_id', 'ra_frb','dec_frb','b_err','a_err','theta','tags','frb_survey','DM','include','locnote', 'hgnote', 'desi_cutout'],skiprows=1)
final.set_index('chime_event_id',inplace=True)
final.sort_index(inplace=True)
final = final[final['include'] == 'yes']

final.keys()

# Add galactic DM & tau

from astropy.coordinates import SkyCoord

c_icrs = SkyCoord(ra=final['ra_frb'].values,
                  dec=final['dec_frb'].values,unit = 'deg',frame = 'icrs') # baseband position

dm_ne2001, tau600_ne2001 = get_ne2001(c_icrs,get_scatt = True)
dm_yt20 = calculate_halo_dm_yt20(l = c_icrs.galactic.l.deg, b= c_icrs.galactic.b.deg)

final.insert(loc = 5, column = 'DM_NE2001',value = dm_ne2001,allow_duplicates = False)
final['tau_NE2001_600'] = tau600_ne2001
final.insert(loc = 5, column = 'DM_YT20',value = dm_yt20,allow_duplicates = False)
final.insert(loc = 5, column = 'gal_l', value = c_icrs.galactic.l.deg)
final.insert(loc = 5, column = 'gal_b', value = c_icrs.galactic.b.deg)


# Add PATH; HG Spectroscopic Redshifts from FFFFPZ; HG photo z from DR9 and PS1; and DM IGM estimates

PATH_TO_PATH = '/arc/home/calvin/kko_host_paper/data_products/b_err_1.3_PATH/'
path_runs = glob('/arc/home/calvin/kko_host_paper/data_products/b_err_1.3_PATH/*/*candidates.xlsx')
path_runs.sort()

for pr in path_runs:
    tns = pr.split('/')[-2][3:]
    print(tns)
    path_input = glob(os.path.join(PATH_TO_PATH,'FRB'+tns,'*input.json'))[0]
    xlsx_input = glob(os.path.join(PATH_TO_PATH,'FRB'+tns,'*candidates.xlsx'))[0]
    cands = pd.read_excel(xlsx_input)
    where = np.where('FRB' + tns == final['name'])
    if len(where[0]) > 0:
        print(f'adding FRB{tns} successfully')
        index = final.index[where[0][0]]  
        final.loc[index,'primary_id'] = f"{cands.iloc[0]['survey']} {cands.iloc[0]['ID']}"
        final.loc[index,'secondary_id'] = f"{cands.iloc[1]['survey']} {cands.iloc[1]['ID']}"
        for key in ['ra','dec','P_Ox','z_phot_median','z_phot_l95','z_phot_u95','z_spec','mag']:
            try:
                final.loc[index,f'primary_{key}'] = cands.iloc[0][key]
            except KeyError:
                final.loc[index,f'primary_{key}'] = -1
        for key in ['ra','dec','P_Ox','z_phot_median','z_phot_l95','z_phot_u95','z_spec','mag']:
            try:
                final.loc[index,f'secondary_{key}'] = cands.iloc[1][key]
            except KeyError:
                final.loc[index,f'secondary_{key}'] = -1
                print(index,tns,'had no',key)

# Add DR9 photo-z
import requests
for evid,row in final.iterrows():
    for cand in ['primary','secondary']:
        if type(row[f'{cand}_id']) is str:
            if row[f'{cand}_id'][0:3] == 'DEC':
                ra = row[f'{cand}_ra']
                dec=row[f'{cand}_dec']
                ralo = row[f'{cand}_ra'] - 1/3600
                rahi = row[f'{cand}_ra'] + 1/3600
                declo= row[f'{cand}_dec'] - 1/3600
                dechi= row[f'{cand}_dec'] + 1/3600
                req_str = f"https://www.legacysurvey.org/viewer/photoz-dr9/1/cat.json?ralo={ralo:.4f}&rahi={rahi:.4f}&declo={declo:.4f}&dechi={dechi:.4f}"
                response = requests.get(req_str)
                data = response.json()
                if len(data['rd']) > 0:
                    print(f'{evid} offset:',(data['rd'][0][0] - ra) * 3600,'offset:',(data['rd'][0][1] - dec) * 3600,'z=',data['phot_z_mean'][0])
                if len(data['rd']) > 1:
                    print(f'{evid} multiple matches for {evid}; photoz are {data["phot_z_mean"]}')
                if len(data['rd']) == 1:
                    final.at[evid,f"{cand}_z_phot_median"] = data["phot_z_mean"][0]
                    final.at[evid,f"{cand}_z_phot_l95"] = max(data["phot_z_mean"][0] - 2 * data["phot_z_std"][0],0)
                    final.at[evid,f"{cand}_z_phot_u95"] = max(data["phot_z_mean"][0] + 2 * data["phot_z_std"][0],0)
        else:
            print(f"No path for {evid}")

# Add PS1-STRM photo z
ps1_strm_by_objid = { # objid: (zphot0, zphoterr),
    113360826227259720: (0.07495457, 0.0132193777203354),
    140170545876545920: (0.09083903, 0.0303555731615571),
    140420596664782839: (0.46922562, 0.0727915566138116),
    147790546649113381: (0.3192104, 0.162667679515143),
    147880719743595797: (-3, -3),
    148733303360097347: (0.41217527, 0.600852325026966),
    150120264676039997: (0.031180382, 0.00601822409074501),
    151973467535638727: (0.25075325, 0.0725420760314956),
    153483123862476859: (-3, -3),
    154360650307570235: (-3, -3),
    158090182417819186: (0.08914095, 0.0175031167179699),
    166460786900208765: (1.1279283, 0.191816992946666),
    170170111139029291: (0.4101674, 0.151843578631204),
    171220360141353770: (0.25026855, 0.758206622052578),
    173383437614223218: (0.71603227, 0.642966150018758),
    178690368364970411: (-3, -3),
    195463075298975481: (0.08306709, 0.166993417244353),
    201713557741821895: (0.44740435, 0.101919920038202),
    209731095953009613: (0.5818887, 0.0830314155449226),
    209893045145819992: (0.19366728, 0.0137913787265872),
    215062842517036820: (0.5523359, 0.182941045291407),
    215110220483728025: (-3, -3)
}

ps1_strm_wise_by_objid = {
    113360826227259720: (0.06860938, 0.00888159718073157),
    140170545876545920: (0.17583069, 0.0186939412537729),
    158090182417819186: (0.07240619, 0.0118954230702462),
    166460786900208765: (0.34135154, 0.149252641767956),
    178690368364970411: (-3, -3),
    195463075298975481: (0.12988344, 0.0633948829036078),
    209731095953009613: (0.66161686, 0.0667724097346421),
    209893045145819992: (0.16091643, 0.0327217209121002),
    215062842517036820: (0.47843924, 0.135657825703001)
}
for evid,row in final.iterrows():
    for cand in ['primary','secondary']:
        if type(row[f'{cand}_id']) is str:
            if row[f'{cand}_id'][0:3] == 'Pan':
                _,ps1_objid = row[f'{cand}_id'].split(' ')
                ps1_objid = int(ps1_objid)
                if ps1_objid in ps1_strm_by_objid.keys():
                    print(f'adding PS1-STRM zphot0 to {evid}')
                    final.at[evid,f"{cand}_z_phot_median"] = ps1_strm_by_objid[ps1_objid][0]
                    final.at[evid,f"{cand}_z_phot_l95"] = max(ps1_strm_by_objid[ps1_objid][0] - 2 * ps1_strm_by_objid[ps1_objid][1],0) # no redshift less than zero
                    final.at[evid,f"{cand}_z_phot_u95"] = max(ps1_strm_by_objid[ps1_objid][0] + 2 * ps1_strm_by_objid[ps1_objid][1],0) # no redshift less than zero
                if ps1_objid in ps1_strm_wise_by_objid.keys():
                    print(f'adding PS1-STRM-WISE zphot0 to {evid}')
                    final.at[evid,f"{cand}_z_phot_median"] = ps1_strm_wise_by_objid[ps1_objid][0]
                    final.at[evid,f"{cand}_z_phot_l95"] = max(ps1_strm_wise_by_objid[ps1_objid][0] - 2 * ps1_strm_wise_by_objid[ps1_objid][1],0) # no redshift less than zero
                    final.at[evid,f"{cand}_z_phot_u95"] = max(ps1_strm_wise_by_objid[ps1_objid][0] + 2 * ps1_strm_wise_by_objid[ps1_objid][1],0) # no redshift less than zero

# Add redshifts from old spreadsheet
csv = pd.read_csv('/arc/home/calvin/kko_host_paper/data_products/KKO_bursts_full.csv')
csv.set_index('chime_event_id',inplace=True)

for key in final.index:
    if final.loc[key]['primary_P_Ox'] > 0.9:
        print('what we have in new file:')
        print(key,final.loc[key]['primary_id'],final.loc[key]['primary_z_spec'])
        if key in csv.index:
            print('present in old file')
            print(csv.loc[key]['hg_id'],csv.loc[key]['redshift'])
            final.at[key,'primary_z_spec'] = np.nanmax([final.loc[key]['primary_z_spec'], csv.loc[key]['redshift']])
            print('Replaced with z=',final.loc[key]['primary_z_spec'])
        else:
            print('no redshift')

# Add (and supersede) specz from FFFFPZ
ffffpz = {
"FRB20230203A": (0.1464,"Keck-2023B-3"),
"FRB20230222A": (0.1223,"Lick-2023B-1"),
"FRB20230222B": (0.1100,"Lick-2023B-1"),
"FRB20230311A": (0.1918,"Keck-2023B-3"),
"FRB20230410A": (0.1684,"V/147/sdss12"),
"FRB20230616A": (0.4485,"V/147/sdss12"),
"FRB20230702A": (0.3658,"V/147/sdss12"),
"FRB20230703A": (0.1184,"Keck-2023B-3"),
"FRB20230730A": (0.2115,"Keck-2023B-3"),
"FRB20230828A": (0.6268,"V/147/sdss12"),
"FRB20230923A": (0.4445,"V/147/sdss12"),
"FRB20230918A": (0.0600,"V/147/sdss12"),
"FRB20230924A": (0.3942,"V/147/sdss12"),
"FRB20230926A": (0.0553,"Lick-2024A-1"),
"FRB20231005A": (0.0713,"Lick-2024A-1"),
"FRB20231006B": (0.4780,"V/147/sdss12"),
"FRB20231007B": (-2,"Dusty"),
"FRB20231008A": (0.0771,"2023ApJ...949L...3R"),
"FRB20231011A": (0.0783,"Keck-2023B-3"),
"FRB20231017A": (0.245,"Lick-2024B-1"),
"FRB20231025B": (0.3238,"V/147/sdss12"),
"FRB20231025C": (-2,"Dusty"),
"FRB20231102A": (0.6878,"V/147/sdss12"),
"FRB20231123A": (0.0729,"Lick-2024B-1"),
"FRB20231128A": (0.1079,"Lick-2024A-1"),
"FRB20231201A": (0.1190,"GN-2024A-3"),
"FRB20231202A": (-2,"Dusty"),
"FRB20231204A": (0.0644,"Lick-2024A-1"),
"FRB20231206A": (0.0659,"2000AJ....120.2338R"),
"FRB20231223C": (0.1059,"Lick-2024A-1"),
"FRB20231223D": (0.1606,"V/147/sdss12"),
"FRB20231224A": (0.0800,"2015ApJS..218...10V"),
"FRB20240210C": (0.3658,"V/147/sdss12"),
}
for ind,row in final.iterrows():
    if row['name'] in ffffpz.keys():
        final.at[ind,'primary_z_spec'] = ffffpz[row['name']][0]
        final.at[ind,'primary_z_spec_source'] = ffffpz[row['name']][1]
    else:
        final.at[ind,'primary_z_spec'] = -1
        final.at[ind,'primary_z_spec_source'] = "None"
        
import matplotlib.pyplot as plt

# Add DM IGM estimates
from frb.dm import igm
z_interp = np.geomspace(0.001,1,num = 30)
dm_igm_interp = []
for ii,z_interp_i in enumerate(z_interp):
    print(f"DM IGM: {ii}/30")
    dm_igm_interp.append(igm.average_DM(z_interp_i).value)

best_redshifts = []
for evid,row in final.iterrows():
    if row['primary_z_spec'] > 0:
        best_redshift = row['primary_z_spec']
    elif row['primary_z_phot_median'] > 0:
        best_redshift = row['primary_z_phot_median']
    else:
        best_redshift = -1
    best_redshifts.append(best_redshift)
dm_igm = np.interp(xp = z_interp, fp = dm_igm_interp, x = best_redshifts)
best_redshifts = np.squeeze(best_redshifts)
dm_igm[best_redshifts < 0] = np.nan
dm_igm[~np.isfinite(best_redshifts)] = np.nan

final['dm_igm'] = dm_igm

for evid,row in final.iterrows():
    if row['primary_z_spec'] > 0:
        best_redshift = row['primary_z_spec']
    elif row['primary_z_phot_median'] > 0:
        best_redshift = row['primary_z_phot_median']
    else:
        best_redshift = -1
    if best_redshift > 0:
        dm_igm = igm.average_DM(best_redshift).value
        final.at[evid,'dm_igm'] = dm_igm
    else:
        final.at[evid,'dm_igm'] = np.nan

# Add Galactic extinction from F & M 2007
import dustmaps
from dustmaps.sfd import SFDQuery
import dustmaps.sfd as sfd
#dustmaps.sfd.fetch()
sfd = SFDQuery()
sources = coord.SkyCoord(final['primary_ra'],final['primary_dec'],unit = 'deg',frame = 'icrs')
e_b_minus_v = sfd(sources)
r_v = 3.1
a_v = r_v * e_b_minus_v
import extinction
evs = []
for av in a_v:
    ext_value_mags = extinction.fm07(wave = np.array([6000.0]), a_v = av, unit = 'aa')
    evs.append(ext_value_mags)
evs = np.squeeze(evs)        
final['extinction_mags'] = evs

fluxfluence_all = {'335002503': (6.02, 0.69, 152.83, 15.46),
 '348570237': (78.34, 7.91, 46.06, 4.86),
 '325140573': (221.19, 22.13, 60.02, 6.05),
 '331641308': (74.93, 7.5, 64.11, 6.44),
 '326033351': (9.08, 1.23, 46.15, 5.14),
 '347265274': (271.87, 27.76, 374.13, 38.71),
 '308358106': (7.17, 0.79, 15.63, 1.73),
 '342256558': (4.66, 0.55, 25.53, 2.71),
 '341979347': (15.04, 1.53, 10.6, 1.12),
 '344693725': (83.82, 8.4, 19.22, 2.02),
 '324962446': (16.3, 1.69, 33.48, 3.48),
 '348285021': (211.23, 21.13, 95.47, 9.58),
 '339589360': (2.77, 0.32, 67.68, 6.87),
 '348687097': (204.39, 20.45, 280.96, 28.17),
 '325515646': (13.68, 1.45, 36.74, 3.88),
 '347082860': (68.07, 6.81, 287.17, 28.75),
 '343537552': (11.82, 1.21, 36.27, 3.73),
 '330976597': (5.97, 0.64, 5.23, 0.61),
 '330838705': (16.58, 1.68, 3.93, 0.44),
 '339069867': (23.29, 2.4, 15.46, 1.76),
 '322840327': (224.84, 22.49, 9942.72, 994.28),
 '328261629': (33.6, 3.37, 8.19, 0.86),
 '333031657': (15.19, 1.54, 12.01, 1.26),
 '347027446': (10.67, 1.16, 15.02, 1.72),
 '321977347': (26.78, 2.78, 28.41, 3.08),
 '347261948': (20.09, 2.05, 13.18, 1.41),
 '332593291': (33.15, 3.4, 215.85, 21.8),
 '333356311': (3.7, 0.46, 18.78, 2.05),
 '319829796': (88.57, 8.86, 58.22, 5.85),
 '338403209': (52.66, 5.31, 56.84, 5.85),
 '343770778': (8.89, 0.94, 17.29, 1.86),
 '347052803': (16.88, 1.76, 38.69, 4.05),
 '328879027': (11.5, 1.21, 53.57, 5.54),
 '343712196': (24.34, 2.47, 10.97, 1.23),
 '347465543': (14.73, 1.52, 25.34, 2.65),
 '347433921': (10.37, 1.45, 65.59, 7.32),
 '329033047': (176.1, 17.61, 162.87, 16.3),
 '327479591': (12.23, 1.27, 40.5, 4.19),
 '321375070': (23.33, 2.35, 34.0, 3.45),
 '348537586': (6.59, 1.05, 37.1, 4.37),
 '337105685': (7.29, 0.77, 109.12, 11.03),
 '334942484': (7.22, 0.76, 81.13, 8.25),
 '341758878': (31.1, 3.14, 33.82, 3.47),
 '329248428': (7.0, 0.83, 18.34, 2.14),
 '328495927': (23.44, 2.57, 805.04, 81.1),
 '341477821': (35.13, 3.53, 10.83, 1.15),
 '331618070': (32.41, 3.37, 66.25, 6.93),
 '347419602': (22.54, 2.66, 81.68, 9.0),
 '317888392': (368.82, 36.89, 773.55, 77.42),
 '326456063': (129.53, 12.96, 58.32, 5.86),
 '283032006': (23.18, 2.33, 7.6, 0.8),
 '275629567': (9.47, 0.97, 12.18, 1.29),
 '318361859': (336.81, 33.69, 92.04, 9.22),
 '281936786': (28.35, 2.96, 72.72, 7.56),
 '282988614': (73.29, 7.34, 11.8, 1.21),
 '313479875': (50.74, 5.11, 338.89, 34.03),
 '276387136': (298.55, 29.86, 139.39, 13.97),
 '282352003': (45.49, 4.56, 16.73, 1.7),
 '307424887': (22.81, 2.31, 69.3, 7.02),
 '300938875': (9.36, 0.97, 8.45, 0.93),
 '306528286': (43.88, 4.41, 37.07, 3.82),
 '268786306': (6.15, 0.65, 173.03, 17.4),
 '272350056': (4.47, 0.47, 129.17, 12.97),
 '300895079': (29.75, 3.04, 52.6, 5.4),
 '254837189': (388.21, 39.09, 489.61, 50.01),
 '296843842': (31.91, 3.26, 83.81, 8.65),
 '304166930': (28.96, 2.93, 433.2, 43.46),
 '282159121': (7.87, 0.83, 12.77, 1.39),
 '296798120': (4.35, 0.54, 19.18, 2.11),
 '277059004': (10.84, 1.11, 97.55, 9.85),
 '272367619': (91.35, 9.14, 37.55, 3.8),
 '358105468': (46.45, 4.74, 241.12, 24.38),
 '346002524': (13.33, 1.45, 12.33, 1.51),
 '347603667': (8.76, 0.97, 75.06, 7.72),
 '342760050': (9.88, 1.13, 179.43, 18.23),
 '338344597': (59.62, 5.98, 8.92, 0.99),
 '339188139': (17.53, 1.78, 185.69, 18.66),
 '358233910': (44.22, 5.68, 369.63, 39.39),
 '337199498': (180.14, 18.3, 98.95, 10.88),
 '325442526': (52.87, 5.3, 138.87, 13.93),
 '325287323': (68.22, 6.85, 20.55, 2.14),
 '321637098': (8.0, 0.83, 9.68, 1.04),
 '347043008': (68.02, 8.0, 131.58, 15.44)}

# Add Flux Fluence
final['flux'] = np.nan
final['flux_err'] = np.nan
final['fluence'] = np.nan
final['fluence_err'] = np.nan
for evid,row in final.iterrows():
    if str(evid) in fluxfluence_all.keys():
        final.at[evid,'flux'] = fluxfluence_all[str(evid)][0]
        final.at[evid,'flux_err'] = fluxfluence_all[str(evid)][1]
        final.at[evid,'fluence'] = fluxfluence_all[str(evid)][2]
        final.at[evid,'fluence_err'] = fluxfluence_all[str(evid)][3]
    else:
        print(evid)

def dataframe_to_latex_formatted(df, filename,**to_latex_kwargs):
    """
    Outputs a formatted pandas DataFrame to a LaTeX file, filtering out rows based on the 'include' column,
    omitting specified columns, formatting specific columns with the desired precision, and including/excluding
    host-related columns based on the 'hosts' argument.

    Parameters:
    df (pd.DataFrame): The DataFrame to be converted to LaTeX.
    filename (str): The name of the output LaTeX file.
    hosts (bool): If True, include 'primary_z_spec', 'primary_mag', 'primary_id', 'primary_P_Ox', 'flux', and 'fluence' columns.
                  If False, include only 'primary_P_Ox'.
    """
    # Filter rows where 'include' == 'yes'
    df = df[df['include'] == 'yes']
    # Format specific columns
    # Combine flux and flux_err into a formatted string
    if 'flux' in df.columns:
        #df['flux'] = df.apply(lambda row: f"${row['flux']:.1f} \pm {row['flux_err']:.1f}$ Jy", axis=1)
        df['flux'] = df.apply(lambda row: f"${row['flux']:.1f}$", axis=1)
    # Combine fluence and fluence_err into a formatted string
    if 'fluence' in df.columns:
        #df['fluence'] = df.apply(lambda row: f"${row['fluence']:.1f} \pm {row['fluence_err']:.1f}$ Jy ms", axis=1)
        df['fluence'] = df.apply(lambda row: f"${row['fluence']:.1f}$", axis=1)
    df['primary_mag'] = df['primary_mag'].map(lambda x: f"{x:.2f}")

    # Drop unwanted columns
    df = df.drop(columns=[
        'tags', 'frb_survey', 'locnote', 'hgnote', 'desi_cutout', 'tau_NE2001_600',
        'secondary_ra', 'secondary_dec', 'secondary_P_Ox', 'secondary_z_phot_median',
        'secondary_z_phot_l95', 'secondary_z_phot_u95', 'secondary_z_spec'
    ])

    # Ensure that only the required columns are kept
    df = df[['name', 'ra_frb', 'dec_frb', 'b_err', 'a_err','theta', 'DM','flux', 'fluence', 'primary_P_Ox']]


    # Format specific columns with the desired precision
    def zspecformat(x):
        if x == -2:
            return "Dusty"
        else:
            return f"{x:.4f}"
    df['ra_frb'] = df['ra_frb'].map(lambda x: f"{x:.5f}")
    df['dec_frb'] = df['dec_frb'].map(lambda x: f"{x:.5f}")
    df['b_err'] = df['b_err'].map(lambda x: f"{x * 60:.3f}")
    df['a_err'] = df['a_err'].map(lambda x: f"{x * 60:.3f}")
    df['theta'] = df['theta'].map(lambda x: f"{x:.2f}")
    df['primary_P_Ox'] = df['primary_P_Ox'].map(lambda x: f"{x:.3f}")
    #df['primary_z_spec_source'] = df['primary_z_spec_source']
    #df['primary_z_spec'] = df['primary_z_spec'].map(zspecformat)
    df['DM'] = df['DM'].map(lambda x: f"${x:.1f}$")
    #df['DM_YT20'] = df['DM_YT20'].map(lambda x: f"{x:.1f}")
    #df['DM_NE2001'] = df['DM_NE2001'].map(lambda x: f"{x:.1f}")
    # Rename columns to be more LaTeX-friendly
    formatted_columns = {
        'name': r'$\text{Name}$',
        'ra_frb': r'$\text{RA}_{\text{FRB}}$', 
        'dec_frb': r'$\text{DEC}_{\text{FRB}}$', 
        'b_err': r'$b_{\text{err}}$', 
        'a_err': r'$a_{\text{err}}$', 
        #'gal_b': r'$\text{gal\_b}$',
        #'gal_l': r'$\text{gal\_l}$', 
        'DM_YT20': r'$\text{DM}_{\text{YT20}}$', 
        'DM_NE2001': r'$\text{DM}_{\text{NE2001}}$', 
        'theta': r'Angle',
        'DM': r'$\text{DM}$', 
        'primary_id': r'$\text{ID}_\text{HG}$',
        'primary_ra': r'$\text{RA}_\text{HG}$', 
        'primary_dec': r'$\text{DEC}_\text{HG}$', 
        'primary_P_Ox': r'$\text{P(O} \vert \text{x)}$', 
        'primary_z_spec_source': r'$\text{Source of } z_\mathrm{spec}$',
        #'primary_mag': r'$m_r$',
        #'primary_z_phot_median': r'$\text{Primary } z_{\text{phot, median}}$', 
        #'primary_z_phot_l95': r'$\text{Primary } z_{\text{phot, l95}}$', 
        #'primary_z_phot_u95': r'$\text{Primary } z_{\text{phot, u95}}$', 
        #'primary_z_spec': r'$z_{\text{spec}}$', 
        #'m_r': r'$m_r$',
        'flux': r'$\text{Flux (Jy)}$',
        'fluence': r'$\text{Fluence (Jy ms)}$'
        }
    

    # Rename the columns for LaTeX output
    df = df.rename(columns=formatted_columns)

    # Convert DataFrame to LaTeX format and save to file
    latex_code = df.to_latex(index=False, escape=False,longtable=True,**to_latex_kwargs)

    # Write the LaTeX code to the specified file
    with open(filename, 'w') as f:
        f.write(latex_code)

    print(f"LaTeX table successfully saved to {filename}")

### WRITE IT OUT
gold_sample = (final['primary_P_Ox'] > 0.9) + (final['name'] == 'FRB20230311A')
dataframe_to_latex_formatted(final[gold_sample], '/arc/home/calvin/kko_host_paper/sample_gold.tex',label = 'tab:gold_sample',caption = 'FRBs presented in this paper with secure host galaxies.')
dataframe_to_latex_formatted(final[~gold_sample], '/arc/home/calvin/kko_host_paper/sample_full.tex',label = 'fig:full_sample', caption = 'The remaining FRB localizations, in the same format as Tab.~\ref{tab:gold_sample}')
final.to_csv('/arc/home/calvin/kko_host_paper/data_products/kko_full_cat.csv')
