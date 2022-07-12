import json
import os
import matplotlib.pyplot as plt
from scipy.integrate import trapz
from scipy.interpolate import interp1d
import numpy as np
from math import *
from collections import Counter


path_to_json = "your path to 2dm- JSON files goes here"
json_files = [
    pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith(".json") and pos_json.startswith("2dm-3993")
]


tol = 0.001
kB = 8.61733e-5
A_to_cm = 1e-8
t = {}

def get_formula(id):
    labels = []
    # labels_unique = []
    for site in id['structure']['sites']:
        labels.append(site['label'])
        # if site['label'] not in labels_unique:
        # labels_unique.append(site['label'])
    # formula = str(str(labels_unique[0])) + str(Counter(labels)[str(labels_unique[0])]) + str(str(labels_unique[1]))+ str(Counter(labels)[str(labels_unique[1])])
    return Counter(labels)

def get_dos_at_efermi(dos):

    #total_density = sum(dos["densities"].values()) only for summing over multiple spins
    max_density = max(dos["densities"]["1"])
    min_index = np.argmin(abs(np.array(dos["energies"]) - dos["efermi"]))

    return dos["densities"]["1"][min_index], 100 * dos["densities"]["1"][min_index] / max_density

def get_cbm_vbm(dos):

    tdos = dos["densities"]["1"]

    # find index of fermi energy
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

def get_gap(dos):
    (cbm, vbm) = get_cbm_vbm(dos)
    return max(cbm - vbm, 0.0)


with open(os.path.join(path_to_json, '2dm-4487.json')) as json_file1:
    mpid1 = json.load(json_file1)
with open(os.path.join(path_to_json, '2dm-3594.json')) as json_file2:
    mpid2 = json.load(json_file2)
    try:
        mpids = [mpid1, mpid2]
        if mpid1["dos"]["efermi"] is not None and mpid2["dos"]["efermi"] is not None:
            fermi_energies = [mpid1["dos"]["efermi"], mpid2["dos"]["efermi"]]
            for i in range(len(mpids)):
                if fermi_energies[i] == max(fermi_energies):
                    donor_mpid = mpids[i]
                elif fermi_energies[i] == min(fermi_energies):
                    acceptor_mpid = mpids[i]
                else:
                    print("fermi energies are equal")

        print("donor:", donor_mpid["material_id"], "acceptor:", acceptor_mpid["material_id"])

        donor_dos = donor_mpid["dos"]
        donor_bandgap = get_gap(donor_dos)
        donor_energies = np.array(donor_dos["energies"])
        donor_efermi = donor_dos["efermi"]
        donor_tdos = np.array(donor_dos["densities"]["1"])
        donor_a = donor_mpid["structure"]["lattice"]["a"]
        donor_b = donor_mpid["structure"]["lattice"]["b"]
        donor_c = donor_mpid["structure"]["lattice"]["c"]
        donor_alpha = donor_mpid["structure"]["lattice"]["alpha"]
        donor_beta = donor_mpid["structure"]["lattice"]["beta"]
        donor_gamma = donor_mpid["structure"]["lattice"]["gamma"]
        donor_area = donor_a * donor_b * sin(np.deg2rad(donor_gamma))
        donor_volume = donor_a * donor_b * donor_c * np.sqrt(1-(cos(np.deg2rad(donor_alpha)))**2 - (cos(np.deg2rad(donor_beta)))**2 - (cos(np.deg2rad(donor_gamma)))**2 + 2 * cos(np.deg2rad(donor_alpha)) * cos(np.deg2rad(donor_beta)) * cos(np.deg2rad(donor_gamma)))

        acceptor_dos = acceptor_mpid["dos"]
        acceptor_energies = np.array(acceptor_dos["energies"])
        acceptor_bandgap = get_gap(acceptor_dos)
        acceptor_efermi = acceptor_dos["efermi"]
        acceptor_tdos = np.array(acceptor_dos["densities"]["1"])
        acceptor_a = acceptor_mpid["structure"]["lattice"]["a"]
        acceptor_b = acceptor_mpid["structure"]["lattice"]["b"]
        acceptor_c = acceptor_mpid["structure"]["lattice"]["c"]
        acceptor_alpha = acceptor_mpid["structure"]["lattice"]["alpha"]
        acceptor_beta = acceptor_mpid["structure"]["lattice"]["beta"]
        acceptor_gamma = acceptor_mpid["structure"]["lattice"]["gamma"]
        acceptor_area = acceptor_a * acceptor_b * sin(np.deg2rad(acceptor_gamma))
        acceptor_volume = acceptor_a * acceptor_b * acceptor_c * np.sqrt(
            1 - (cos(np.deg2rad(acceptor_alpha))) ** 2 - (cos(np.deg2rad(acceptor_beta))) ** 2 - (
                cos(np.deg2rad(acceptor_gamma))) ** 2 + 2 * cos(np.deg2rad(acceptor_alpha)) * cos(
                np.deg2rad(acceptor_beta)) * cos(np.deg2rad(acceptor_gamma)))

        acceptor_ecbm, acceptor_evbm = get_cbm_vbm(acceptor_dos)
        donor_ecbm, donor_evbm = get_cbm_vbm(donor_dos)
        idx_vbm_donor = int(np.argmin(abs(donor_energies - donor_evbm)))
        idx_cbm_donor = int(np.argmin(abs(donor_energies - donor_ecbm)))

        idx_rel_donorcbm_acceptorbasis = int(np.argmin(abs(acceptor_energies - donor_ecbm)))
        idx_rel_acceptorcbm_donorbasis = int(np.argmin(abs(donor_energies - acceptor_ecbm)))

        idx_rel_donorvbm_acceptorbasis = int(np.argmin(abs(acceptor_energies - donor_evbm)))
        idx_rel_acceptorvbm_donorbasis = int(np.argmin(abs(donor_energies - acceptor_evbm)))

        donor_energies = np.array(donor_dos["energies"])
        donor_de = np.hstack((donor_energies[1:], donor_energies[-1])) - donor_energies
        acceptor_energies = np.array(acceptor_dos["energies"])
        acceptor_de = np.hstack((acceptor_energies[1:], acceptor_energies[-1])) - acceptor_energies

        # if donor_bandgap:
        #     if donor_evbm < donor_efermi < donor_ecbm:
        #         donor_eref = donor_efermi
        #     else:
        #         donor_eref = (donor_evbm + donor_ecbm) / 2.0
        #
        #     idx_fermi_donor = int(np.argmin(abs(donor_energies - donor_eref)))
        #
        #     if idx_fermi_donor == idx_vbm_donor:
        #         # Fermi level and vbm should be different indices
        #         idx_fermi_donor += 1
        #
        # if acceptor_bandgap:
        #     if acceptor_evbm < acceptor_efermi < acceptor_ecbm:
        #         acceptor_eref = acceptor_efermi
        #     else:
        #         acceptor_eref = (acceptor_evbm + acceptor_ecbm) / 2.0

        idx_vbm_acceptor = int(np.argmin(abs(acceptor_energies - acceptor_evbm)))
        idx_cbm_acceptor = int(np.argmin(abs(acceptor_energies - acceptor_ecbm)))

        cb_integral = np.sum(
            acceptor_tdos[idx_cbm_acceptor: idx_rel_donorcbm_acceptorbasis + 1]
            * acceptor_de[idx_cbm_acceptor: idx_rel_donorcbm_acceptorbasis + 1],
            axis=0,
        )
        vb_integral = np.sum(
            donor_tdos[idx_rel_acceptorvbm_donorbasis: idx_vbm_donor + 1]
            * donor_de[idx_rel_acceptorvbm_donorbasis: idx_vbm_donor + 1],
            axis=0,
        )
        print("the charge transfer is:")
        print(0.5*(vb_integral - cb_integral) * 1e-13 / (acceptor_area * A_to_cm ** 2))


        #acceptor_x_shifted = np.array(acceptor_dos["energies"]) - acceptor_dos["efermi"]
        acceptor_x = np.array(acceptor_dos["energies"])
        acceptor_f = interp1d(acceptor_x, acceptor_dos["densities"]["1"], fill_value="extrapolate")

        #donor_x_shifted = np.array(donor_dos["energies"]) - donor_dos["efermi"]
        donor_x = np.array(donor_dos["energies"])
        donor_f = interp1d(donor_x, donor_dos["densities"]["1"], fill_value="extrapolate")

        #domain_d = np.linspace(donor_dos["efermi"], donor_dos["efermi"]+donor_evbm, 100)
        domain_d = np.linspace(acceptor_evbm, donor_evbm,
                               100)
        domain_a = np.linspace(acceptor_evbm, donor_evbm,
                               100)
        domain_test = np.linspace(acceptor_evbm,donor_evbm,100)
        domain_test2 = np.linspace(acceptor_ecbm, donor_ecbm, 100)
        domain_cbm = np.linspace(acceptor_energies[idx_cbm_acceptor],acceptor_energies[idx_rel_donorcbm_acceptorbasis+1],100)
        domain_vbm = np.linspace(donor_energies[idx_rel_acceptorvbm_donorbasis],donor_energies[idx_vbm_donor+1])

        print(trapz(donor_f(domain_vbm), domain_vbm,donor_de[idx_rel_acceptorvbm_donorbasis: idx_vbm_donor + 1]) - trapz(
            acceptor_f(domain_cbm), domain_cbm,acceptor_de[idx_cbm_acceptor: idx_rel_donorcbm_acceptorbasis + 1]
        ))
        # print(trapz(donor_f(domain_test4), domain_test4,
        #             donor_de[idx_rel_acceptorvbm_donorbasis: idx_cbm_donor + 1]))
        # print(trapz(acceptor_f(domain_test3), domain_test3,acceptor_de[idx_cbm_acceptor: idx_rel_donorcbm_acceptorbasis + 1]))

        plt.plot(donor_x, donor_dos["densities"]["1"], linewidth = 3)
        plt.plot(acceptor_x, acceptor_dos["densities"]["1"], linewidth=3,color="orange")
        plt.fill_between(domain_test, donor_f(domain_test), color="green")
        plt.fill_between(domain_cbm, acceptor_f(domain_cbm), color="magenta")
        plt.fill_between(domain_vbm, donor_f(domain_vbm), color="red")

        plt.axvline(acceptor_efermi, label="acceptor Ef", color="black", ls="solid")
        plt.axvline(donor_efermi, label="donor Ef", color="gray", ls="solid")
        plt.axvline(acceptor_evbm, label="acceptor vbm", color="blue", ls="dotted")
        plt.axvline(acceptor_ecbm, label="acceptor cbm", color="red", ls="dotted")
        plt.axvline(donor_evbm, label="donor vbm", color="blue", ls="--")
        plt.axvline(donor_ecbm, label="donor cbm", color="red", ls="--")
        plt.legend()


        plt.show()
    except:
        pass
