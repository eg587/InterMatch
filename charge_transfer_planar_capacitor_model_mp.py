from pymatgen.ext.matproj import MPRester
import numpy as np
from math import *

mpr = MPRester("your API key goes here")
A_to_cm = 1e-8
tol = 0.001

def get_cbm_vbm(dos):

    tdos = dos["densities"]["1"]

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
            acceptor_mpid = mpids[1]

    print("donor:", donor_mpid, "acceptor:", acceptor_mpid)


    mpids_list = [donor_mpid, acceptor_mpid]
    structures = [mpr.get_structure_by_material_id(mpid) for mpid in mpids_list]
    doses = [mpr.get_dos_by_material_id(mpid) for mpid in mpids_list]

    donor_dos = doses[0].as_dict()
    donor_energies = np.array(donor_dos["energies"])
    donor_efermi = donor_dos["efermi"]

    acceptor_dos = doses[1].as_dict()
    acceptor_energies = np.array(acceptor_dos["energies"])
    acceptor_efermi = acceptor_dos["efermi"]

    idx_efermi_donor = int(np.argmin(abs(donor_energies - donor_efermi)))
    idx_efermi_acceptor = int(np.argmin(abs(acceptor_energies - acceptor_efermi)))

    #print(np.argwhere(donor_dos["densities"]["1"] is None))
    if donor_dos["densities"]["1"][idx_efermi_donor] == 0.0:
        donor_ecbm, donor_evbm = get_cbm_vbm(donor_dos)
        idx_vbm_donor = int(np.argmin(abs(donor_energies - donor_evbm)))
        idx_cbm_donor = int(np.argmin(abs(donor_energies - donor_ecbm)))
        if abs((donor_dos["densities"]["1"][idx_efermi_donor]-donor_dos["densities"]["1"][idx_vbm_donor]) < abs(donor_dos["densities"]["1"][idx_efermi_donor]-donor_dos["densities"]["1"][idx_cbm_donor])):
            idx_efermi_donor = idx_vbm_donor-1
        else:
            idx_efermi_donor = idx_cbm_donor+1
    if acceptor_dos["densities"]["1"][idx_efermi_acceptor] == 0.0:
        acceptor_ecbm, acceptor_evbm = get_cbm_vbm(acceptor_dos)
        idx_vbm_acceptor = int(np.argmin(abs(acceptor_energies - acceptor_evbm)))
        idx_cbm_acceptor = int(np.argmin(abs(acceptor_energies - acceptor_ecbm)))
        if abs((acceptor_dos["densities"]["1"][idx_efermi_acceptor]-acceptor_dos["densities"]["1"][idx_vbm_acceptor]) < abs(acceptor_dos["densities"]["1"][idx_efermi_acceptor]-acceptor_dos["densities"]["1"][idx_cbm_acceptor])):
            idx_efermi_acceptor = idx_vbm_acceptor-1
        else:
            idx_efermi_acceptor = idx_cbm_acceptor+1


#     if donor_dos["densities"]["1"][idx_efermi_donor] == 0.0:
#         idx_min_nonzero_donor = min(np.argwhere(donor_dos["densities"]["1"] == 0.0))-1
#         idx_max_nonzero_donor = max(np.argwhere(donor_dos["densities"]["1"] == 0.0))+1
#         min(np.argwhere(donor_dos["densities"]["1"] == 0.0), key=lambda x:abs(x-donor_dos["densities"]["1"][idx_efermi_donor]))
#
    donor_dos_at_efermi = donor_dos["densities"]["1"][idx_efermi_donor]
    acceptor_dos_at_efermi = acceptor_dos["densities"]["1"][idx_efermi_acceptor]

    # print(donor_dos["densities"]["1"][idx_vbm_donor-1])
    # print(donor_dos["densities"]["1"][idx_cbm_donor+1])
    # print(donor_dos_at_efermi)
    # print(acceptor_dos_at_efermi)
    # print(donor_efermi)
    # print(acceptor_efermi)

    dN = donor_dos_at_efermi * acceptor_dos_at_efermi * (donor_efermi-acceptor_efermi) / (donor_dos_at_efermi + acceptor_dos_at_efermi)


    surface_areas = [
        structure.lattice.a
        * structure.lattice.b
        * sin(np.deg2rad(structure.lattice.gamma))
        for structure in structures
    ]
        #computing bohr radii in case we want to add the geometrical capacitance:
    # params = [structure.species for structure in structures]
    #
    # max_rad1 = 1
    # max_rad2 = 1
    # for spec1 in params[0]:
    #     elt1 = Element(spec1)
    #     if elt1.van_der_waals_radius == None:
    #         print("no vdW radius data")
    #     else:
    #         max_rad1 = max(max_rad1, elt1.van_der_waals_radius)
    # for spec2 in params[1]:
    #     elt2 = Element(spec2)
    #     if elt2.van_der_waals_radius == None:
    #         print("no vdW radius data")
    #     else:
    #         max_rad2 = max(max_rad2, elt2.van_der_waals_radius)
    #
    # d = max_rad1 + max_rad2
    #
    # if d == 0:
    #     d = 3.0
    # else:
    #     d = max_rad1 + max_rad2

    return dN * 1e-13 / (min(surface_areas)* A_to_cm ** 2)

#example list of materials for screening with RuCl3: graphene + TMDS
mpid_list = ["mp-48", "mp-2815", "mp-9813", "mp-1027692", "mp-1821", "mp-1030319", "mp-22693"]
for mat in mpid_list:
    print(calculate_charge_transfer(mat,"mp-22850"))
