import json
from tqdm import tqdm
from scipy import linalg
from itertools import combinations
from pymatgen.ext.matproj import MPRester
import numpy as np
from math import *
import math

#script for finding superlattices that are commensurate or near-commensurate which minimize elastic energy

mpr = MPRester('your API key goes here')

#constants
kB = 8.61733e-5
A_to_cm = 1e-8
GPA_to_eV_per_angstrom3 = 160.21766208
thickness = 3.45

#constraints
elastic_energy_max = 0.001
strain_max = 0.1
tol_occ = 0.001

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

def compute_supercell(mpid_1,mpid_2,nmax,mmax,theta,elastic_energy_max,strain_max):

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

    for i in range(0, nmax):
        for j in range(-mmax, mmax):
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
        if abs(elastic_energy_tensor[4]) < elastic_energy_max and round(np.linalg.norm(np.array(V1_v))) == round(
                np.linalg.norm(np.array(V2_v))) and abs(strain[2]) < strain_max:
            unit_v1 = np.array(V1_v) / np.linalg.norm(np.array(V1_v))
            unit_v2 = np.array(V2_v) / np.linalg.norm(np.array(V2_v))
            dot_product = np.dot(unit_v1, unit_v2)
            angle = round(np.arccos(dot_product) * 180 / math.pi, 4)
            new_t = t.copy()
            new_t['project'] = 'intermatch'
            # new_t['is_public']= True,
            new_t['identifier'] = mpid_1
            new_t['data'] = {'interface': mpid_2,
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
            #strains.append(strain[2])
            #atoms.append(total_atoms)

    #further refine superlattice selection based on strain, elastic energy, lagrangian strain tensor eigenvalue
    for i in range(len(supercells)):
        # if supercells[i]["data"]["\u03B5"]['deformation'] < 0.005:
        # if abs(supercells[i]["data"]["elasticenergy"]['εvav']) < 0.002:
        if abs(supercells[i]["data"]['max_strain_component']) < 0.005:
            # if (round(supercells[i]["data"]['\u03B8ₘ']) == 60.0 or round(supercells[i]["data"]['\u03B8ₘ']) == 120.0) and supercells[i]["data"]["elasticenergy"]['elastic_energy'] < elastic_energy_max:
            datalist.append(supercells[i])

            with open('superlattices_'+str(supercells[i]["identifier"]+supercells[i]["data"]["interface"])+'.json', 'w') as fp:
               json.dump(supercells[i], fp)
        else:
            pass

    # for cell in sorted(datalist, key=lambda cell: cell["data"]['elasticenergy']['elastic_energy']):
    #     print("{" + str(cell['data']['\u03B8']) + "," + str(round(np.linalg.norm(cell["data"]['v1']))) + "," + str(
    #         cell["data"]['max_strain_component']) + "," + str(
    #         cell["data"]['elasticenergy']['elastic_energy']) + "}" + ",")

dtheta = 1
thetamin = 0
thetamax = 30
for theta in tqdm(range(thetamin, thetamax + 1, dtheta)):
    print(theta)
    # thetalist.append(theta)
    compute_supercell("mp-48","mp-22850", 120, 120, theta, elastic_energy_max,strain_max)
    print("#################################################################")
