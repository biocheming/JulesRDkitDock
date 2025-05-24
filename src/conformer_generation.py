from rdkit import Chem
from rdkit.Chem import AllChem

def generate_conformers(ligand_mol, num_conformers=50, max_attempts=1000, random_seed=42):
    """
    Generates and optimizes conformers for a given RDKit molecule.

    Args:
        ligand_mol (rdkit.Chem.rdchem.Mol): The input RDKit molecule.
        num_conformers (int): The desired number of conformers to generate.
        max_attempts (int): Maximum embedding attempts for each conformer.
        random_seed (int): Seed for the random number generator.

    Returns:
        rdkit.Chem.rdchem.Mol: The molecule object with generated and optimized 
                               conformers. If embedding fails, the molecule will
                               have 0 conformers. The molecule passed as ligand_mol
                               is not modified if AddHs returns a new object; the
                               returned molecule is the one with conformers.
    Raises:
        ValueError: If ligand_mol is None.
    """
    if ligand_mol is None:
        raise ValueError("Input ligand_mol cannot be None.")

    # 1. Add hydrogens. addCoords=True is important for EmbedMultipleConfs.
    # Chem.AddHs usually returns a new molecule object if addCoords=True or if the
    # input molecule is not perceived as editable.
    # We operate on this new molecule 'mol_with_hs'.
    mol_with_hs = Chem.AddHs(ligand_mol, addCoords=True)

    # 2. Set up embedding parameters using ETKDGv3
    params = AllChem.ETKDGv3()
    params.randomSeed = random_seed
    params.maxAttempts = max_attempts
    # Other parameters like params.numThreads = 0 (use all cores) could be set here.

    # 3. Generate conformers
    # EmbedMultipleConfs returns a list of conformer IDs that were successfully generated.
    # The conformers are stored directly in the mol_with_hs object.
    conformer_ids = list(AllChem.EmbedMultipleConfs(mol_with_hs, numConfs=num_conformers, params=params))

    if not conformer_ids:
        # Embedding failed to generate any conformers.
        # mol_with_hs will have 0 conformers.
        # (Note: if AddHs created an initial conformer, EmbedMultipleConfs failing might clear it,
        # or it might remain if numConfs=0 was somehow requested and AddHs created one.
        # Standard behavior for EmbedMultipleConfs failure is 0 conformers on the molecule.)
        # print(f"Warning: Conformer embedding failed for molecule. Returning molecule with 0 conformers.")
        pass # Fall through to return mol_with_hs, which will have 0 conformers.
             # The molecule state (e.g. number of atoms due to AddHs) is preserved.

    # 4. Optimize generated conformers (if any)
    if len(conformer_ids) > 0:
        try:
            # UFFOptimizeMoleculeConfs optimizes all conformers present on the molecule.
            AllChem.UFFOptimizeMoleculeConfs(mol_with_hs)
        except Exception as e:
            # Force field optimization can sometimes fail (e.g., for very strained structures).
            # print(f"Warning: UFFOptimizeMoleculeConfs failed: {e}. Conformers may not be optimized.")
            # Depending on desired behavior, one might clear conformers or return as is.
            # For now, return with unoptimized (or partially optimized) conformers if optimization fails.
            pass 
            
    return mol_with_hs
