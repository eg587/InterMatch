from pymatgen.ext.matproj import MPRester
import numpy as np
from math import *
from scipy.optimize import newton
from scipy.integrate import trapz
from scipy.interpolate import interp1d
from pymatgen.core.periodic_table import Element

mpr = MPRester("your API key goes here")
A_to_cm = 1e-8
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

#list of materials for screening with RuCl3: graphene + TMDS
mpid_list = ["mp-48", "mp-2815", "mp-9813", "mp-1634", "mp-1821", "mp-602", "mp-22693"]
for mat in mpid_list:
    print(abs(calculate_charge_transfer(mat,"mp-22850")))
