InterMatch
==============

***High-Throughput Design of Complex Atomic Interfaces***

**Authors:** *Eli Gerber and Steven B. Torrisi*

# Overview

Scripts comprising the InterMatch framework for computing charge transfer, band alignments, elastic energy, strain tensor, moiré superlattices, 
and supercells using data from computational materials databases Materials Project and 2Dmatpedia as inputs.

* The file `InterMatch_functions_mp.py` contains the main functions of the algorithm, as well as examples of looping over lists of candidate materials 
and querying such lists from the Materials Project.

* The file `superlattice_finder_mp.py` searches for stable moiré superlattices which minimize elastic energy.

* The file `create_interface_cif_mp.py` uses output of `InterMatch_functions_mp.py` or `superlattice_finder_mp.py` to generate CIF files of the 
interface supercells.

* The files `charge_transfer_...` contain different models of interfacial charge transfer based on band alignment, DOS integration, and a 
planar quantum capacitor model. The file `charge_transfer_band_alignment_model_2dm.py` uses data from the 2Dmatpedia database as inputs, and we include 
two example 2Dmatpedia data files `2dm-3594.json` and `2dm-4487.json` for use with this script.

---


