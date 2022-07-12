from pymatgen.ext.matproj import MPRester
from pymatgen.core.operations import SymmOp
from pymatgen.vis.structure_vtk import StructureVis
from pymatgen.vis.structure_vtk import MultiStructuresVis
from pymatgen.io.cif import CifWriter

mpr = MPRester('your API key goes here')
#mpid_1 should be the first input in "calculate_supercell()" function

mpid_1 = "mp-1634"  # MoSe2
mpid_2 = "mp-2089"  # ZrTe3

mpids = [mpid_1, mpid_2]

structures = [mpr.get_structure_by_material_id(mpid) for mpid in mpids]
print(structures[0])
print(structures[1])
lattice_params = [
    [structure.lattice.a, structure.lattice.b, structure.lattice.c, structure.lattice.gamma,structure.lattice.alpha,structure.lattice.beta]
    for structure in structures
]

theta = 83  #twist angle in degrees between systems mpid_1 and mpid_2
structures[1].apply_operation(SymmOp.from_axis_angle_and_translation([0,0,1],theta))

#integers defining superlattice vectors, output from supercell calculation
i11 = -4
j11 = 9
i21 = -3
j21 = 4
i12 = -18
j12 = -3
i22 = -9
j22 = -3

structures[1].make_supercell([[i11,j11,0],[i21,j21,0],[0,0,1]])
structures[0].make_supercell([[i12,j12,0],[i22,j22,0],[0,0,1]])

print(structures[0])
print(structures[1])

#visualize each layer of the supercell
multivis = MultiStructuresVis()
multivis.set_structures([structures[0],structures[1]])
multivis.show()

# vis = StructureVis()
# vis.set_structure(structures[0])
# vis.set_structure(structures[1])
# vis.show()

#write and save cif files of each layer in the supercell
writecif0 = CifWriter(structures[0])
writecif1 = CifWriter(structures[1])

writecif0.write_file("MoSe2_supercell.cif")
writecif1.write_file("ZrTe3_supercell.cif")


