from fairchem.core import pretrained_mlip
from tools.rease_calculator import FAIRChemCalculator
from ase.atoms import Atoms
from rdkit import Chem

class Atom(Atoms):
    def __init__(self, symbols=None,
                 positions=None, numbers=None,
                 tags=None, momenta=None, masses=None,
                 magmoms=None, charges=None,
                 scaled_positions=None,
                 cell=None, pbc=None, celldisp=None,
                 constraint=None,
                 calculator=None,
                 info=None,
                 velocities=None):
        super(Atom, self).__init__(symbols=symbols, positions=positions, numbers=numbers,
                 tags=tags, momenta=momenta, masses=masses,
                 magmoms=magmoms, charges=charges,
                 scaled_positions=scaled_positions,
                 cell=cell, pbc=pbc, celldisp=celldisp,
                 constraint=constraint,
                 calculator=calculator,
                 info=info,
                 velocities=velocities)

    def get_embedding(self):
        potential_energy = self._calc.get_potential_energy(self)
        embedding_data = self._calc.results["embedding"].detach().cpu()
        embedding_data = embedding_data.reshape(len(embedding_data), -1)
        return embedding_data, potential_energy

class UMAEmbedding:
    def __init__(self):
        self.predictor = pretrained_mlip.get_predict_unit("uma-s-1p1")
        self.get_potential_energy = None
        self.embedding = None


    def fit(self, seq, position, mol_charge=0):
        molecule = Atom(symbols=seq, positions=position)
        molecule.calc = FAIRChemCalculator(self.predictor, task_name="omol")
        molecule.info.update({"spin": 1, "charge": mol_charge})
        embedding, potential_energy = molecule.get_embedding()
        self.get_potential_energy = potential_energy
        self.embedding = embedding
