import json
import os
from tqdm import tqdm
import pymatgen as pmg
import scipy as scipy
from scipy import linalg
import itertools
from itertools import combinations
import cProfile, io,pstats
from pstats import SortKey
import sys, time
from pymatgen.ext.matproj import MPRester
from pymatgen.analysis.phase_diagram import PhaseDiagram
import numpy as np
from math import *
from scipy.optimize import newton
from scipy.integrate import trapz
from scipy.interpolate import interp1d
from pymatgen.core.periodic_table import Element

mpr = MPRester('v2anpbHsAOr4tmXQ')

def calculate_energy_above_hull(mpid):
    '''
    returns energy above hull of system specified by param mpid
    '''
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

def calculate_charge_transfer(mpid1,mpid2):
    '''
    Returns estimate of charge transfer at the interface
    of systems mpid1 and mpid2 based on a planar capacitor model.
    '''

    mpids = [mpid1, mpid2]

    #if hasattr(mpr.get_dos_by_material_id(mpid1), "efermi") == True and hasattr(mpr.get_dos_by_material_id(mpid2), "efermi") == True:
    fermi_energies = [mpr.get_dos_by_material_id(mpid1).as_dict()["efermi"],
                      mpr.get_dos_by_material_id(mpid2).as_dict()["efermi"]]
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
    donor_f = interp1d(donor_x_shifted, donor_dos["densities"]["1"], fill_value="extrapolate")

    acceptor_dos = doses[1].as_dict()
    acceptor_x_shifted = np.array(acceptor_dos["energies"]) - acceptor_dos["efermi"]
    acceptor_f = interp1d(acceptor_x_shifted, acceptor_dos["densities"]["1"], fill_value="extrapolate")

    surface_areas = [structure.lattice.a * structure.lattice.b * sin(np.deg2rad(structure.lattice.gamma))
                     for structure in structures]

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
        domain_d = np.linspace(donor_dos["efermi"], donor_dos["efermi"] + delta_Ef_d, 100)
        domain_a = np.linspace(acceptor_dos["efermi"], acceptor_dos["efermi"] - delta_Ef_d, 100)
        return trapz(acceptor_f(domain_a), domain_a) - trapz(donor_f(domain_d), domain_d)

    def main():
        optimal_N = newton(charge, 1, maxiter=10000)
        return optimal_N

    return main()

def compute_stress_tensor_supercell(v1, v2, u1, u2):
    '''
    Returns the strain tensor from the (v1, v2) cell to the
    corresponding (u1, u2) cell and vice versa.
    '''
    [v1x, v1y] = v1
    [v2x, v2y] = v2
    [u1x, u1y] = u1
    [u2x, u2y] = u2

    εv11 = abs(v1x/u1x)-1
    εv22 = abs(v2y/u2y)-1
    εv12 = 0.5*(v2x-(v1x/u1x)*u2x)/v2y
    εvav = (εv11+εv22+εv12)/3

    εu11 = abs(u1x / v1x) - 1
    εu22 = abs(u2y / v2y) - 1
    εu12 = 0.5 * (u2x - (u1x / v1x) * v2x) / u2y
    εuav = (εu11 + εu22 + εu12) / 3

    vstrain = εv11, εv22, εv12, εvav

    ustrain = εu11, εu22, εu12, εuav

    return vstrain

def calculate_strain(a1, b1, a2, b2):
    '''
    Returns linear Lagrangian strain tensor (deformation) and
    corresponding eigenvalues given two initial unit cells with lattice vectors
    (a1, b1) and (a2, b2).
    '''
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

def calculate_elastic_strain_supercell(mpid1, mpid2, V1_v, V2_v, V1_u, V2_u):
    '''
     Returns elastic strain at the interface of a supercell of systems mpid1 and mpid2,'
     using elasticity data from MP.
     (V1_u, V2_u) are the supercell lattice vectors in the basis of mpid1,
     (V1_v, V2_v) are the supercell lattice vectors in the basis of mpid2.
     Returns 0 array if elasticity data does not exist.
     '''

    data_mpid_1 = mpr.query(mpid1, ['elasticity'])[0]
    data_mpid_2 = mpr.query(mpid2, ['elasticity'])[0]

    if data_mpid_1['elasticity'] is not None and data_mpid_2['elasticity'] is not None :

        poisson_mpid1 = data_mpid_1['elasticity']['poisson_ratio']
        poisson_mpid2 = data_mpid_2['elasticity']['poisson_ratio']

        bulkmod_mpid1 = data_mpid_1['elasticity']['K_Voigt_Reuss_Hill']
        bulkmod_mpid2 = data_mpid_2['elasticity']['K_Voigt_Reuss_Hill']

        youngmod_mpid1 = 3*bulkmod_mpid1*(1-2*poisson_mpid1)
        youngmod_mpid2 = 3*bulkmod_mpid2*(1-2*poisson_mpid2)

        axial_comp_1 = (1 - poisson_mpid1) / youngmod_mpid1
        axial_comp_2 = (1 - poisson_mpid2) / youngmod_mpid2

        axial_comp_max = max(axial_comp_1, axial_comp_2)
        axial_comp_min = min(axial_comp_1, axial_comp_2)

        a1 = V1_u
        b1 = V2_u

        a2 = V1_v
        b2 = V2_v

        a_max = max(a1, a2)
        b_max = max(b1, b2)
        a_min = min(a1, a2)
        b_min = min(b1, b2)

        a_eff = (axial_comp_max + axial_comp_min) * a_max * a_min / (axial_comp_max * a_max + axial_comp_min * a_min)
        b_eff = (axial_comp_max + axial_comp_min) * b_max * b_min / (axial_comp_max * b_max + axial_comp_min * b_min)

        strain_int_comp_a = (a_max - a_eff) / a_max
        strain_int_tens_a = (a_eff - a_min) / a_min

        strain_int_comp_b = (b_max - b_eff) / b_max
        strain_int_tens_b = (b_eff - b_min) / b_min

        return [strain_int_comp_a, strain_int_tens_a, strain_int_comp_b, strain_int_tens_b, a_eff, b_eff]
    else:
        return [0, 0, 0, 0, 0, 0]

#params for compute_best_supercell():
dtheta=1
thetamin=0
thetamax=60
nmax = 10
mmax = 10
strain_min = -0.1
strain_max = 0.1
tolerance = 1e-6
atoms_min = 1
atoms_max = 400

def compute_best_supercell(mpid_1,mpid_2,nmax,mmax,theta,strain_min,strain_max,tolerance,atoms_min,atoms_max):
    '''

    :param mpid_1:
    :param mpid_2:
    :param nmax: maximum integer multiple of first lattice vector of mpid_2 used in supercell search
    :param mmax: maximum integer multiple of second lattice vector of mpid_2 used in supercell search
    :param theta: twist angle between systems specified by mpid_1 and mpid_2
    :param strain_min: minimum strain allowed in supercell
    :param strain_max: maximum strain allowed in supercell
    :param tolerance: margin of error for cells to be considered commensurate
    :param atoms_min: minimum number of total atoms allowed in supercell
    :param atoms_max: maximum number of total atoms allowed in supercell
    :return: 'interface' dict object ready to upload to MPContribs
    '''
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

    #print('a1x:', a1x, 'a1y:', a1y, 'gamma1:', lattice_params[0][2])


    U = np.array([[a1x, a2x], [a1y, a2y]])

    b1x = lattice_params[1][0]
    b1y = 0.0
    b2x = lattice_params[1][1] * cos(np.deg2rad(lattice_params[1][2]))
    b2y = lattice_params[1][1] * sin(np.deg2rad(lattice_params[1][2]))

    #print('b1x:', b1x, 'b1y:', b1y, 'gamma2:', lattice_params[1][2])

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

    for i in range(-nmax, nmax + 1):
        for j in range(0, mmax + 1):
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
        strain = calculate_strain(V1_v, V2_v, V1_u, V2_u)
        # strain = compute_stress_tensor(bcells[x][0][0], bcells[x][1][0], bcells[x][0][1], bcells[x][1][1])
        area_ratio_v = round(abs((V1_v[0] * V2_v[1] - V1_v[1] * V2_v[0]) / (br1x * br2y - br1y * br2x)))
        area_ratio_u = round(abs((V1_u[0] * V2_u[1] - V1_u[1] * V2_u[0]) / (a1x * a2y - a1y * a2x)))
        atoms_u = round(atoms1 * area_ratio_u)
        atoms_v = round(atoms2 * area_ratio_v)
        # total_atoms = round(atoms1 * area_ratio_u + atoms2 * area_ratio_v)
        total_atoms = round(atoms_u + atoms_v)
        if abs(strain[0]) < strain_max and abs(strain[1]) < strain_max and abs(
                strain[2]) < strain_max and atoms_max > total_atoms > atoms_min and round(bcells[x][0][0][0]) == round(
                bcells[x][0][1][0]) and round(bcells[x][0][0][1]) == round(bcells[x][0][1][1]) and round(
                bcells[x][1][0][0]) == round(bcells[x][1][1][0]) and round(bcells[x][1][0][1]) == round(
                bcells[x][1][1][1]):
            # bcells_str.append(bcells[x])
            #print(strain, total_atoms, bcells[x])
            new_t = t.copy()
            new_t['project'] = 'intermatch'
            # new_t['is_public']= True,
            new_t['identifier'] = mpid_1
            new_t['data'] = {'interface': mpid_2,
                             '\u03B5': {'\u03B5\u2081\u2081': strain[0], '\u03B5\u2082\u2082': strain[1],
                                        'deformation': strain[2]},
                             'atoms': total_atoms,
                             '\u03B8': theta,
                             'surface': {'N\u2081': atoms_u, 'N\u2082': atoms_v},
                             'v1':V1_v,
                             'v2': V2_v,
                             'u1': V1_u,
                             'u2': V2_u,
                             'v\u2081': {'i\u2081\u2081': bcells[x][0][2][0], 'j\u2081\u2081': bcells[x][0][2][1],
                                         'i\u2082\u2081': bcells[x][1][2][0], 'j\u2082\u2081': bcells[x][1][2][1]},
                             'v\u2082': {'i\u2081\u2082': bcells[x][0][3][0], 'j\u2081\u2082': bcells[x][0][3][1],
                                         'i\u2082\u2082': bcells[x][1][3][0], 'j\u2082\u2082': bcells[x][1][3][1]}
                             }
            supercells.append(new_t)
            strains.append(strain[2])
            atoms.append(total_atoms)
    for i in range(len(supercells)):
        #if supercells[i]["data"]["\u03B5"]['deformation'] == min(strains):
        if supercells[i]["data"]["\u03B5"]['deformation']==min(strains) or supercells[i]["data"]["atoms"]==min(atoms):
            best_cell = supercells[i]
            print(best_cell)

                #print("The supercell with minimal strain is:")
                #print(supercells[i])
                #with open('interface_'+str(supercells[i]["identifier"]+supercells[i]["data"]["interface"])+'.json', 'w') as fp:
                #    json.dump(supercells[i], fp)
            return best_cell
        else:
            pass

for theta in tqdm(range(thetamin,thetamax+1,dtheta)):
    print(theta)
    compute_best_supercell("mp-1984", "mp-1027692", nmax, mmax, theta, strain_min, strain_max, tolerance, atoms_min, atoms_max)
