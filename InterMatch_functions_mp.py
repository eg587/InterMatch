import json
import os
from tqdm import tqdm
from scipy import linalg
import itertools
import math
from itertools import combinations
from pymatgen.ext.matproj import MPRester
from pymatgen.analysis.phase_diagram import PhaseDiagram
import numpy as np
from math import *
from scipy.optimize import newton
from scipy.integrate import trapz
from scipy.interpolate import interp1d
from pymatgen.core.periodic_table import Element

mpr = MPRester("your API key goes here")

#constants
kB = 8.61733e-5
A_to_cm = 1e-8
GPA_to_eV_per_angstrom3 = 160.21766208
thickness = 3.45

#constraints
elastic_energy_max = 0.001
tol_occ = 0.001

#calculate energy above hull to screen systems for stability
def calculate_energy_above_hull(mpid):
    """
    returns energy above hull of system specified by param mpid
    """
    idstructure = mpr.get_structure_by_material_id(mpid)
    idparams = idstructure.species
    specieslist = []
    for spec in idparams:
        elt = Element(spec).symbol
        if elt not in specieslist:
            specieslist.append(elt)
    entries = mpr.get_entries_in_chemsys(specieslist)
    entry_of_interest = mpr.get_entry_by_material_id(mpid)
    pd = PhaseDiagram(entries)
    e_above_hull = pd.get_e_above_hull(entry_of_interest)

    return e_above_hull

#calculate CBM (conduction band minimum) and VBM (valence band maximum)
def get_cbm_vbm(dos):
    tdos = dos["densities"]["1"]
    tol = tol_occ * tdos.sum() / tdos.shape[0]

    i_fermi = 0
    while dos["energies"][i_fermi] <= dos["efermi"]:
        i_fermi += 1


    i_gap_start = i_fermi
    while i_gap_start - 1 >= 0 and tdos[i_gap_start - 1] <= tol:
        i_gap_start -= 1


    i_gap_end = i_gap_start
    while i_gap_end < len(tdos) and tdos[i_gap_end] <= tol:
        i_gap_end += 1
    i_gap_end -= 1
    return dos["energies"][i_gap_end], dos["energies"][i_gap_start]

#calculate bandgap
def get_gap(dos):
    (cbm, vbm) = get_cbm_vbm(dos)
    return max(cbm - vbm, 0.0)
#assign donor and acceptor
def assign_donor(mpid_1, mpid_2):
    mpids = [mpid_1, mpid_2]
    fermi_energies=[mpr.get_dos_by_material_id(mpid).as_dict()["efermi"] for mpid in mpids]
    if fermi_energies[0] == max(fermi_energies):
        donor_mpid = mpids[0]
        acceptor_mpid = mpids[1]
    else:
        donor_mpid = mpids[1]
        acceptor_mpid = mpids[0]
    return [acceptor_mpid, donor_mpid]

#calculate charge transfer at an interface
def calculate_charge_transfer(mpid1, mpid2):
    """
    Returns estimate of charge transfer at the interface
    of systems mpid1 and mpid2 based on a planar capacitor model.
    """

    mpids = [mpid1, mpid2]

    # if hasattr(mpr.get_dos_by_material_id(mpid1), "efermi") == True and hasattr(mpr.get_dos_by_material_id(mpid2), "efermi") == True:
    fermi_energies = [
        mpr.get_dos_by_material_id(mpid1).as_dict()["efermi"],
        mpr.get_dos_by_material_id(mpid2).as_dict()["efermi"],
    ]
    for i in range(len(mpids)):
        if fermi_energies[i] == max(fermi_energies):
            donor_mpid = mpids[i]
        elif fermi_energies[i] == min(fermi_energies):
            acceptor_mpid = mpids[i]
        else:
            donor_mpid = mpids[0]
            acceptor_mpids = mpids[1]
    mpids_list = [donor_mpid, acceptor_mpid]
    structures = [mpr.get_structure_by_material_id(mpid) for mpid in mpids_list]
    doses = [mpr.get_dos_by_material_id(mpid) for mpid in mpids_list]

    donor_dos = doses[0].as_dict()
    donor_x_shifted = np.array(donor_dos["energies"]) - donor_dos["efermi"]
    donor_f = interp1d(
        donor_x_shifted, donor_dos["densities"]["1"], fill_value="extrapolate"
    )

    acceptor_dos = doses[1].as_dict()
    acceptor_x_shifted = np.array(acceptor_dos["energies"]) - acceptor_dos["efermi"]
    acceptor_f = interp1d(
        acceptor_x_shifted, acceptor_dos["densities"]["1"], fill_value="extrapolate"
    )

    surface_areas = [
        structure.lattice.a
        * structure.lattice.b
        * sin(np.deg2rad(structure.lattice.gamma))
        for structure in structures
    ]

    params = [structure.species for structure in structures]

    max_rad1 = 1
    max_rad2 = 1
    for spec1 in params[0]:
        elt1 = Element(spec1)
        if elt1.van_der_waals_radius == None:
            print("no vdW radius data")
        else:
            max_rad1 = max(max_rad1, elt1.van_der_waals_radius)
    for spec2 in params[1]:
        elt2 = Element(spec2)
        if elt2.van_der_waals_radius == None:
            print("no vdW radius data")
        else:
            max_rad2 = max(max_rad2, elt2.van_der_waals_radius)

    d = max_rad1 + max_rad2

    if d == 0:
        d = 3.0
    else:
        d = max_rad1 + max_rad2

    alpha = 1e4 / max(surface_areas)
    Wa = -acceptor_dos["efermi"]
    Wd = -donor_dos["efermi"]

    def charge(N):
        delta_Ef_d = Wd - Wa - alpha * N * d
        # delta_Ef_d = N
        # delta_Ef_a = Wa - Wd - alpha * N * d
        domain_d = np.linspace(
            donor_dos["efermi"], donor_dos["efermi"] + delta_Ef_d, 100
        )
        domain_a = np.linspace(
            acceptor_dos["efermi"], acceptor_dos["efermi"] - delta_Ef_d, 100
        )
        return trapz(acceptor_f(domain_a), domain_a) - trapz(
            donor_f(domain_d), domain_d
        )

    def main():
        optimal_N = newton(charge, 1, maxiter=10000)
        return optimal_N*1e3

    return main()

#retrieve elasticity data for two systems
# def get_elasticity_data(mpid1,mpid2):
#
#     mpid1_elast_data = mpr.query(mpid1, ["elasticity"])[0]
#     #mpid2_elast_data = mpr.query(mpid2, ["elasticity"])[0]
#
#     mpid1_elastic_tensor = mpid1_elast_data['elasticity']['elastic_tensor']
#     #mpid2_elastic_tensor = mpid2_elast_data['elasticity']['elastic_tensor']
#
#     C11_mpid1 = mpid1_elastic_tensor[0][0]/GPA_to_eV_per_angstrom3
#     C12_mpid1 = mpid1_elastic_tensor[0][1]/GPA_to_eV_per_angstrom3
#     C21_mpid1 = mpid1_elastic_tensor[1][0] / GPA_to_eV_per_angstrom3
#     C44_mpid1 = mpid1_elastic_tensor[3][3]/GPA_to_eV_per_angstrom3
#
#     # C11_mpid2 = mpid2_elastic_tensor[0][0] / GPA_to_eV_per_angstrom3
#     # C12_mpid2 = mpid2_elastic_tensor[0][1] / GPA_to_eV_per_angstrom3
#     # C21_mpid2 = mpid2_elastic_tensor[1][0] / GPA_to_eV_per_angstrom3
#     # C44_mpid2 = mpid2_elastic_tensor[3][3] / GPA_to_eV_per_angstrom3
#
#     return C11_mpid1, C12_mpid1, C21_mpid1, C44_mpid1  #, C11_mpid2, C12_mpid2, C21_mpid2, C44_mpid2

#calculate interface superlattice stress tensor and elastic energy per unit area
def compute_stress_tensor_supercell(v1, v2, u1, u2, mpid1, mpid2):
    '''
    Returns the strain tensor from the (v1, v2) cell to the
    corresponding (u1, u2) cell and vice versa, as well as the
    elastic energy of the interface per unit area.
    '''

    mpid1_elast_data = mpr.query(mpid1, ["elasticity"])[0]
    # mpid2_elast_data = mpr.query(mpid2, ["elasticity"])[0]

    mpid1_elastic_tensor = mpid1_elast_data['elasticity']['elastic_tensor']
    # mpid2_elastic_tensor = mpid2_elast_data['elasticity']['elastic_tensor']

    C11_mpid1 = mpid1_elastic_tensor[0][0] / GPA_to_eV_per_angstrom3
    C12_mpid1 = mpid1_elastic_tensor[0][1] / GPA_to_eV_per_angstrom3
    C21_mpid1 = mpid1_elastic_tensor[1][0] / GPA_to_eV_per_angstrom3
    C44_mpid1 = mpid1_elastic_tensor[3][3] / GPA_to_eV_per_angstrom3

    # C11_mpid2 = mpid2_elastic_tensor[0][0] / GPA_to_eV_per_angstrom3
    # C12_mpid2 = mpid2_elastic_tensor[0][1] / GPA_to_eV_per_angstrom3
    # C21_mpid2 = mpid2_elastic_tensor[1][0] / GPA_to_eV_per_angstrom3
    # C44_mpid2 = mpid2_elastic_tensor[3][3] / GPA_to_eV_per_angstrom3

    [v1x, v1y] = v1
    [v2x, v2y] = v2
    [u1x, u1y] = u1
    [u2x, u2y] = u2

    εv11 = abs(v1x/u1x)-1
    εv22 = abs(v2y/u2y)-1
    εv12 = 0.5*(v2x-(v1x/u1x)*u2x)/v2y
    εvav = (εv11+εv22+εv12)/3

    # εu11 = abs(u1x / v1x) - 1
    # εu22 = abs(u2y / v2y) - 1
    # εu12 = 0.5 * (u2x - (u1x / v1x) * v2x) / u2y
    # εuav = (εu11 + εu22 + εu12) / 3


    elastic_energy_per_unit_area_v_1 = thickness * ((εv11 ** 2) * C11_mpid1 + εv11 * εv22 * C12_mpid1 + 0.5 * (εv12 ** 2) * C44_mpid1)
    #elastic_energy_per_unit_area_v_2 = thickness * ((εv11 ** 2) * C11_mpid2 + εv11 * εv22 * C12_mpid2 + 0.5 * (εv12 ** 2) * C44_mpid2)

    vstrain = εv11, εv22, εv12, εvav, abs(elastic_energy_per_unit_area_v_1)  #, abs(elastic_energy_per_unit_area_v_2)

    #ustrain = εu11, εu22, εu12, εuav

    return vstrain

def calculate_strain(a1, b1, a2, b2):
    [a1x, a1y] = a1
    [b1x, b1y] = b1
    [a2x, a2y] = a2
    [b2x, b2y] = b2

    metric_tensor1 = np.matrix(
        [
            [
                a1x ** 2 + a1y ** 2,
                a1x * b1x + a1y * b1y,
            ],
            [
                a1x * b1x + a1y * b1y,
                b1x ** 2 + b1y ** 2,
            ]
        ]
    )
    metric_tensor2 = np.matrix(
        [
            [
                a2x ** 2 + a2y ** 2,
                a2x * b2x + a2y * b2y,
            ],
            [
                a2x * b2x + a2y * b2y,
                b2x ** 2 + b2y ** 2,
            ]
        ]
    )
    try:
        rt1 = np.linalg.cholesky(metric_tensor1).transpose()
        rt2 = np.linalg.cholesky(metric_tensor2).transpose()

        unit_matrix = np.matrix([[1, 0], [0, 1]])

        evec = rt2 * rt1.getI() - unit_matrix

        strain2 = 0.5 * (evec + evec.transpose() + evec * evec.transpose())
        #print("finite lagrangian tensor")
        str1, str2= np.linalg.eig(strain2)[0]
        deformation = sqrt(str1 ** 2 + str2 ** 2) / 2.0
        strain_tensor = [str1,str2,deformation]
        #print("eigenvalues ", str1, str2)
        #print("deformation", deformation)
    except:
        str1, str2 =1,1
        deformation = sqrt(str1 ** 2 + str2 ** 2) / 2.0
        strain_tensor = [str1, str2, deformation]


    return strain_tensor


def compute_supercell(mpid_1,mpid_2,charge_transfer,nmax,mmax,theta,elastic_energy_max,strain_max,Natoms_max):

    mpids = [mpid_1, mpid_2]

    structures = [mpr.get_structure_by_material_id(mpid) for mpid in mpids]

    atoms1 = len(structures[0])
    atoms2 = len(structures[1])

    lattice_params = [
        [structure.lattice.a, structure.lattice.b, structure.lattice.gamma]
        for structure in structures
    ]
    a1x = lattice_params[0][0]
    a1y = 0.0
    a1z = 0.0
    a2x = lattice_params[0][1] * cos(np.deg2rad(lattice_params[0][2]))
    a2y = lattice_params[0][1] * sin(np.deg2rad(lattice_params[0][2]))

    U = np.array([[a1x, a2x], [a1y, a2y]])

    b1x = lattice_params[1][0]
    b1y = 0.0
    b2x = lattice_params[1][1] * cos(np.deg2rad(lattice_params[1][2]))
    b2y = lattice_params[1][1] * sin(np.deg2rad(lattice_params[1][2]))

    b1xy = np.array([b1x, b1y])
    b2xy = np.array([b2x, b2y])

    a1 = np.array([a1x, a1y])
    a2 = np.array([a2x, a2y])

    b1 = np.array([b1x, b1y])
    b2 = np.array([b2x, b2y])

    t = {}
    supercells = []
    strains = []
    atoms = []
    datalist = []

    bvecs = []
    bcells = []
    rotmat = np.array(
        [[cos(np.deg2rad(theta)), -sin(np.deg2rad(theta))], [sin(np.deg2rad(theta)), cos(np.deg2rad(theta))]])
    b1_rotated = rotmat.dot(b1xy)
    b2_rotated = rotmat.dot(b2xy)
    br1x = b1_rotated[0]
    br1y = b1_rotated[1]
    br2x = b2_rotated[0]
    br2y = b2_rotated[1]

    for i in range(-nmax, nmax):
        for j in range(0, mmax):
            v = i * b1_rotated + j * b2_rotated
            s = linalg.inv(U).dot(v)
            s_int = np.around(s)
            u = U.dot(s_int)
            bvecs.append([v, u, [i, j], s_int, theta])
    comb = combinations(bvecs, 2)
    for V1, V2 in list(comb):
        bcells.append([V1, V2])
    for x in range(len(bcells)):
        V1_v = bcells[x][0][0]
        V2_v = bcells[x][1][0]
        V1_u = bcells[x][0][1]
        V2_u = bcells[x][1][1]
        elastic_energy_tensor = compute_stress_tensor_supercell(V1_v, V2_v, V1_u, V2_u,mpid_1,mpid_2)
        strain = calculate_strain(V1_v, V2_v, V1_u, V2_u)
        area_ratio_v = round(abs((V1_v[0] * V2_v[1] - V1_v[1] * V2_v[0]) / (br1x * br2y - br1y * br2x)))
        area_ratio_u = round(abs((V1_u[0] * V2_u[1] - V1_u[1] * V2_u[0]) / (a1x * a2y - a1y * a2x)))
        mpid2_area = round(abs((V1_v[0] * V2_v[1] - V1_v[1] * V2_v[0])))
        mpid1_area = round(abs((V1_u[0] * V2_u[1] - V1_u[1] * V2_u[0])))
        min_area = min(mpid1_area,mpid2_area)
        atoms_u = round(atoms1 * area_ratio_u)
        atoms_v = round(atoms2 * area_ratio_v)
        total_atoms = round(atoms1 * area_ratio_u + atoms2 * area_ratio_v)

        #selection criteria for desired superlattices
        if abs(elastic_energy_tensor[4]) < elastic_energy_max and abs(strain[2]) < strain_max:
        # if abs(elastic_energy_tensor[4]) < elastic_energy_max and round(np.linalg.norm(np.array(V1_v))) == round(
        #         np.linalg.norm(np.array(V2_v))) and abs(strain[2]) < strain_max:
            unit_v1 = np.array(V1_v) / np.linalg.norm(np.array(V1_v))
            unit_v2 = np.array(V2_v) / np.linalg.norm(np.array(V2_v))
            dot_product = np.dot(unit_v1, unit_v2)
            angle = round(np.arccos(dot_product) * 180 / math.pi, 4)
            new_t = t.copy()
            new_t['project'] = 'intermatch'
            # new_t['is_public']= True,
            new_t['identifier'] = mpid_1
            new_t['data'] = {'interface': mpid_2,
                             '\u0394n': charge_transfer,
                             '\u03B5': {'\u03B5\u2081\u2081': strain[0], '\u03B5\u2082\u2082': strain[1],
                                        'deformation': strain[2]},
                             'elasticenergy': {'εv11': elastic_energy_tensor[0], 'εv22': elastic_energy_tensor[1],
                                               'εv12': elastic_energy_tensor[2], 'εvav': abs(elastic_energy_tensor[3]),
                                               'elastic_energy': abs(elastic_energy_tensor[4] * min_area)},
                             'max_strain_component': max(abs(elastic_energy_tensor[0]), abs(elastic_energy_tensor[1]),
                                                         abs(elastic_energy_tensor[2])),
                             'atoms': total_atoms,
                             '\u03B8': theta,
                             'surface': {'N\u2081': atoms_u, 'N\u2082': atoms_v},
                             'area': min_area,
                             'v1': np.array(V1_v),
                             'v2': np.array(V2_v),
                             'u1': np.array(V1_u),
                             'u2': np.array(V2_u),
                             '\u03B8ₘ': angle,
                             'L':round(np.linalg.norm(np.array(V1_v))),
                             'v\u2081': {'i\u2081\u2081': bcells[x][0][2][0], 'j\u2081\u2081': bcells[x][0][2][1],
                                         'i\u2082\u2081': bcells[x][1][2][0], 'j\u2082\u2081': bcells[x][1][2][1]},
                             'v\u2082': {'i\u2081\u2082': bcells[x][0][3][0], 'j\u2081\u2082': bcells[x][0][3][1],
                                         'i\u2082\u2082': bcells[x][1][3][0], 'j\u2082\u2082': bcells[x][1][3][1]}
                             }
            supercells.append(new_t)
            strains.append(strain[2])
            atoms.append(total_atoms)

    #further refine superlattice selection based on strain, elastic energy, lagrangian strain tensor eigenvalue, number of atoms
    for i in range(len(supercells)):
        if supercells[i]["data"]["\u03B5"]['deformation'] == min(strains) or supercells[i]["data"]["atoms"] == min(atoms):
        # if supercells[i]["data"]["\u03B5"]['deformation'] < 0.005:
        # if abs(supercells[i]["data"]["elasticenergy"]['εvav']) < 0.002:
        #if abs(supercells[i]["data"]['max_strain_component']) < 0.005:
            #datalist.append(supercells[i])   #make list of cells if you want more to choose from
            best_cell = supercells[i]
            return best_cell

entries = []
mpids_list_1 = ["mp-22850"] #RuCl3
mpids_list_2 = ["mp-48","mp-2815", "mp-9813", "mp-1634", "mp-1821", "mp-602", "mp-22693"] #graphene + some TMDs
for i in tqdm(itertools.islice(list(itertools.product(mpids_list_1, mpids_list_2)))):
    try:
        print(i)
        mpid1, mpid2 = i
        dN = calculate_charge_transfer(mpid1, mpid2)
        ids = assign_donor(mpid1, mpid2)
        acceptor_id = ids[0]
        donor_id = ids[1]
        cell = compute_supercell(acceptor_id, donor_id, dN, 10, 10, 0, 0.001, 0.10,300)
        if cell is not None:
            entries.append(cell)
            with open(os.path.join('your path to desired save folder goes here','interface_' + str(cell['identifier'] + cell['data']['interface']) + '.json'), 'w') as fp:
                json.dump(cell, fp)

    except:
        entrieslist = open(os.path.join('your path to desired save folder goes here', 'entries_err_list' + str(i) + '.txt'), 'w')
        for element in entries:
            entrieslist.write(str(element) + ",\n")
        entrieslist.close()
    else:
        pass



# #example of querying some TMDs for screening using the Materials Project API:
#
# tmd_data_S = mpr.query(criteria = {'elements':{'$in':['S','Mo','W'],'$all':['S']},'nelements':2,'efermi':{"$exists":True,"$ne":None}},
#                  properties=['material_id','pretty_formula'])
# tmd_data_Se = mpr.query(criteria = {'elements':{'$in':['Se','Mo','W'],'$all':['Se']},'nelements':2,'efermi':{"$exists":True,"$ne":None}},
#                  properties=['material_id','pretty_formula'])
# tmd_data_Te = mpr.query(criteria = {'elements':{'$in':['Te','Mo','W'],'$all':['Te']},'nelements':2,'efermi':{"$exists":True,"$ne":None}},
#                  properties=['material_id','pretty_formula'])
# big_list = tmd_data_S + tmd_data_Se + tmd_data_Te
# all_tmds = []
# target_strings = ['S2','Se2','Te2']
# for compound in tqdm(big_list):
#     name = compound['pretty_formula']
#     if any(x in name for x in target_strings) and any(y in name for y in ['Mo','W']) and hasattr(mpr.get_dos_by_material_id(compound['material_id']), "efermi") == True:
#         all_tmds.append(compound)
# tmd_mp_ids = [x['material_id'] for x in all_tmds if x['material_id'] is not None]
# print("found",len(tmd_mp_ids),"TMD mpids")
#
# good_tmd_ids = []
# for tmd in tqdm(tmd_mp_ids):
#     try:
#         if hasattr(mpr.get_dos_by_material_id(tmd), "efermi") == True:
#                 e_above_hull = calculate_energy_above_hull(tmd)
#                 if e_above_hull < 0.002:
#                     good_tmd_ids.append(tmd)
#     except:
#         print(good_tmd_ids)
#         pass
# print("found",len(good_tmd_ids),"stable TMD mpids")
#
# for tmd in good_tmd_ids:
#     print(tmd)
# #should give the list ["mp-2815", "mp-9813", "mp-1634", "mp-1821", "mp-602", "mp-22693"]
