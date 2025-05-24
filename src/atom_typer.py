from rdkit import Chem

# Atom type definitions based on SMARTS patterns.
# Order matters: the first rule that matches an atom assigns its type.
ATOM_TYPE_DEFINITIONS = [
    # Hydrogens
    # H: Hydrophobic/Non-polar Hydrogen
    ("H", "[hD0&!$(*~[#7,#8])]"),  # Hydrogen not bonded to N or O

    # Carbons
    # C_H: Hydrophobic Carbon
    ("C_H", "[cD1,cD2,cD3,cD4](-[H])"),  # Aliphatic C with at least one H
    ("C_H", "[aC&H0]"),  # Aromatic C with no H, likely substituted
    ("C_H", "[aC&H1]"),  # Aromatic C with one H
    # C_P: Polar Carbon (Carbon bonded to heteroatom, not part of a C=O group)
    ("C_P", "[C!$(C=O)]~[#7,#8,#15,#16]"),  # Carbon bonded to N, O, P, or S, not carbonyl C

    # Nitrogens
    # N_D: Nitrogen Donor
    ("N_D", "[#7&!$([#7]~[O,S,P]=*)][H]"),  # Nitrogen with H, not part of nitro/sulfonamide/phosphonamide
    # N_A: Nitrogen Acceptor
    ("N_A", "[#7&!$([#7]~[O,S,P]=*)][!H]"),  # Nitrogen without H, not part of nitro/sulfonamide/phosphonamide and not pyrrole-like
    # N_P: Polar Nitrogen (e.g., in amides, nitro groups)
    ("N_P", "[N+]"),  # Positively charged N
    ("N_P", "[#7]~[O,S,P]=*"),  # Nitrogen in nitro, sulfonamide, phosphonamide
    ("N_P", "n"),  # aromatic nitrogen, e.g. in Pyridine
    
    # Oxygens
    # O_A: Oxygen Acceptor
    ("O_A", "[o,O;H1]"),  # Oxygen with one H - hydroxyl, acid (as per prompt, this is O_A and comes first)
    ("O_A", "[O;H0;X2]"),  # Oxygen with no H and two connections - ether, ester
    ("O_A", "[O;H0;X1]=*"),  # Carbonyl oxygen
    # O_D: Oxygen Donor (part of hydroxyl)
    ("O_D", "[o,O;H1]"),  # This specific SMARTS for O_D will be shadowed by the ("O_A", "[o,O;H1]") rule above.
                         # If O_D is desired for these, its rule should be placed before the O_A rule for "[o,O;H1]".

    # Polar Sulfur
    ("S_P", "[#16&X2H0]"),  # Thioether, disulfide
    ("S_P", "[#16&X3H0]"),  # Sulfoxide
    ("S_P", "[#16&X4H0]"),  # Sulfone
    
    # Polar Phosphorus
    ("P_P", "[#15]"),

    # Halogens
    ("F_H", "[F]"),      # Fluorine
    ("Cl_H", "[Cl]"),    # Chlorine
    ("Br_H", "[Br]"),    # Bromine
    ("I_H", "[I]"),      # Iodine
    
    # Metals
    ("Metal", "[Ca,Mg,Zn,Mn,Fe,Cu,Co,Ni,Na,K,Li]")  # Common metals
]

def assign_atom_types(molecule):
    """
    Assigns an 'atom_type' string property to each atom in an RDKit molecule.

    The function iterates through predefined rules (SMARTS patterns) stored in
    ATOM_TYPE_DEFINITIONS. For each atom, the first rule that matches assigns 
    the type. If an atom does not match any rule, it's assigned 'UNK'.

    Args:
        molecule (rdkit.Chem.rdchem.Mol): The input RDKit molecule object.
                                           It's recommended to add hydrogens to the
                                           molecule (e.g., using Chem.AddHs(mol))
                                           before calling this function, as some
                                           SMARTS patterns depend on explicit hydrogens.

    Returns:
        rdkit.Chem.rdchem.Mol: The modified molecule object with 'atom_type' 
                               properties set on its atoms.
    """
    if not molecule:
        raise ValueError("Input molecule cannot be None.")

    # Initialize all atoms with a default type 'UNK' and a flag to track typing.
    for atom in molecule.GetAtoms():
        atom.SetProp("atom_type", "UNK")
        atom.SetBoolProp("_is_typed", False)  # Helper property, underscore indicates internal use

    for atom_type_name, smarts_pattern in ATOM_TYPE_DEFINITIONS:
        pattern = Chem.MolFromSmarts(smarts_pattern)
        if not pattern:
            # This can happen if a SMARTS pattern is invalid.
            # Depending on desired behavior, one might log this or raise an error.
            # For now, we'll skip invalid patterns.
            # print(f"Warning: Could not parse SMARTS pattern: {smarts_pattern} for type {atom_type_name}")
            continue
        
        matches = molecule.GetSubstructMatches(pattern)
        for match_indices in matches:
            for atom_idx in match_indices: # match_indices can be a tuple of atom indices
                atom = molecule.GetAtomWithIdx(atom_idx)
                # Only set type if not already typed by a previous, likely more specific rule.
                if not atom.GetBoolProp("_is_typed"):
                    atom.SetProp("atom_type", atom_type_name)
                    atom.SetBoolProp("_is_typed", True)
    
    # Clean up the helper property from atoms
    for atom in molecule.GetAtoms():
        atom.ClearProp("_is_typed")

    return molecule
