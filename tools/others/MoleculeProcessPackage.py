from rdkit.Chem.MolStandardize.rdMolStandardize import Uncharger
from collections import Counter
from rdkit.Chem import Descriptors
from rdkit.Chem import SaltRemover
from molvs import normalize, metal
from functools import partial
from rdkit import RDLogger
from rdkit import Chem
import numpy as np
import pickle
import tqdm
import random
import sys
# import dgl


# Pipline with some necessary params
class ArgsForPreprocess:
    def __init__(self):
        # salt_remover
        self.salt_type = "[Na,K,Mg,I,Cl,Br]"
        # molecular_weight_filter
        self.weight_filter = [20, 700]  # [min, max]
        # atomic_number_limit
        self.atom_number = [2, 150]  # [min, max]
        # atomic_type_filter
        self.atomic_type = ["C", "H", "O", "N", "P", "F", "Cl", "Br", "S"]


# External params' setting
# ======================================================================↑↑↑↑↑↑↑↑


# Pipline cell
MOL_BASED_FUNCTIONS = [
    "isomeric_remover",
    "add_hydrogen",
    "remove_hydrogen",
    "salt_remover",
    "metal_disconnect",
    "atomic_number_limit",
    "inorganic_remover",
    "molecular_weight_filter",
    "atomic_type_filter",
    "neutralize_charge",
    "functional_group_optimization",
    "other_mol_method"
]

SMILEs_BASED_FUNCTIONS = [
    "mixture_remover",
    "other_smiles_method"
]


def _fixed_operation():
    RDLogger.DisableLog("rdApp.*")
    random.seed(1407)


def _map(*a):
    return list(map(*a))


args = ArgsForPreprocess()
_fixed_operation()


class BasicTools:
    def __init__(self):
        self.smiles = None
        self.mol = None
        self._final_state = None
        self.info_risk = None

        # optional init tools
        self.metal = None
        self.salt = None
        self.charge = None
        self.functional_group = None

    def function(self, functions):
        """
        This operation is used to execute the specific function of different Data representation. (automatic)
        """
        for function_name in functions:
            function = partial(self._skip_none, function=eval("self." + function_name))
            if "%s" % function_name in MOL_BASED_FUNCTIONS:
                if not self._state_update(function, "Mol"):
                    return None
            elif "%s" % function_name in SMILEs_BASED_FUNCTIONS:
                self.info_risk = True
                if not self._state_update(function, "Smiles"):
                    return None
            else:
                try:
                    raise PiplineModuleError("Error: pipline with an unidentifiable module <%s>" % function_name)
                except PiplineModuleError as e:
                    print(f"\033[91m {e} \033[0m")
                    exit(1)

    def _state_update(self, function, state):
        """
        Operate the function with the correspondingly typing, then update the smiles and mol.
        """
        if state == "Mol":
            data = self.mol
        else:
            data = self.smiles
        new_data = function(data)
        if self.info_risk:
            self.info_risk = None
            try:
                assert Chem.MolFromSmiles(new_data) is not None, ""
            except:
                new_data = None
        if new_data is None:  # None
            self.smiles = None
            self.mol = None
            return new_data
        if isinstance(new_data, str):
            self._final_state = "Smiles"
            self.smiles = new_data
            self.mol = Chem.MolFromSmiles(new_data)
        else:
            self._final_state = "Mol"
            self.mol = new_data
            Chem.SanitizeMol(self.mol)
            self.smiles = Chem.MolToSmiles(new_data)
        # assert Chem.MolToSmiles(self.mol) == self.smiles, "Smiles is inconformity with Mol"
        return True

    @staticmethod
    def _skip_none(data, function):
        if data is None:
            return None
        else:
            return function(data)

    @staticmethod
    def mixture_remover(smiles: str):
        """
        Input: molecules (SMILEs)
        >> Process: the max length of SMILEs' section is selected as the valid section.
        Output: molecules (SMILEs)
        """
        smiles_splitting = smiles.split('.')
        section_length = list()
        for s in smiles_splitting:
            section_length.append(len(s))
        if (len(section_length) >= 2) and (len(smiles_splitting[np.argmax(section_length)])) > 5:
            max_index = np.argmax(np.array(section_length))
            return smiles_splitting[max_index]
        else:
            return smiles

    @staticmethod
    def add_hydrogen(mol: Chem.rdchem.Mol):
        return Chem.AddHs(mol)

    @staticmethod
    def remove_hydrogen(mol: Chem.rdchem.Mol):
        return Chem.RemoveHs(mol)

    @staticmethod
    def isomeric_remover(mol: Chem.rdchem.Mol):
        smiles = Chem.MolToSmiles(mol, isomericSmiles=False)
        if "[H]" or "[2H]" in smiles:
            mol = Chem.MolFromSmiles(smiles)
            smiles = Chem.MolToSmiles(mol)
        return smiles

    @staticmethod
    def molecular_weight_filter(mol: Chem.rdchem.Mol):
        mol = Chem.AddHs(mol)
        mw = Descriptors.MolWt(mol)
        if args.weight_filter[0] <= mw <= args.weight_filter[1]:
            return Chem.RemoveHs(mol)
        else:
            return None

    @staticmethod
    def inorganic_remover(mol: Chem.rdchem.Mol):
        all_atom = mol.GetAtoms()
        if len(all_atom) < 2:
            return None
        all_symbol = [a.GetSymbol() for a in all_atom]
        atom_dict = Counter(all_symbol)
        if atom_dict["C"] == 0:
            return None
        return mol

    @staticmethod
    def atomic_type_filter(mol: Chem.rdchem.Mol):
        atom_info = mol.GetAtoms()
        for i in atom_info:
            if i.GetSymbol() not in args.atomic_type:
                return None
        return mol

    @staticmethod
    def atomic_number_limit(mol: Chem.rdchem.Mol):
        all_atoms = mol.GetAtoms()
        atom_number = len(all_atoms)
        try:
            if args.atom_number[0] < atom_number < args.atom_number[1]:
                return mol
            else:
                return None
        except:
            raise ValueError("Atomic_number_error")

    @staticmethod
    def random_smiles(mol: Chem.rdchem.Mol):
        return Chem.MolToSmiles(mol, canonical=False)

    def salt_remover(self, mol: Chem.rdchem.Mol):
        return self.salt(mol)

    def metal_disconnect(self, mol: Chem.rdchem.Mol):
        return self.metal(mol)

    def neutralize_charge(self, mol: Chem.rdchem.Mol):
        return self.charge.uncharge(mol)

    def functional_group_optimization(self, mol: Chem.rdchem.Mol):
        """
            for example: S+ O- O-  --->  O=S=O
        """
        return self.functional_group(mol)

    @staticmethod
    def other_mol_method(mol: Chem.rdchem.Mol):
        raise NotImplementedError

    @staticmethod
    def other_smiles_method(smiles: str):
        raise NotImplementedError


class PiplineModuleError(Exception):
    pass


class MoleculesStore:
    def __init__(self, idx: int = None, mol: Chem.rdchem.Mol = None, smiles: str = None):
        self.idx = idx
        self.mol = mol
        self.smiles = smiles
        self.valid = False
        if self.mol and self.smiles is not None:
            self.valid = True

    def __repr__(self):
        if self.valid:
            return "Molecule: %s (valid)" % self.idx
        else:
            return "Molecule: %s (invalid)" % self.idx


class MoleculePreprocess(BasicTools):
    def __init__(self, pipline_function: list = None, output_type="Smiles", save_name: str = None):
        super().__init__()
        self._functions = pipline_function
        self.moles_set = None
        self.output_type = output_type
        self.save_name = save_name
        self.mol = None
        self.smiles = None
        self.processed_queue = None

        self.salt = None
        self.metal = None
        self.charge = None
        self.functional_group = None

    def pipline(self, _input):
        self.moles_set = self.molecules_load(_input)
        assert isinstance(self._functions, list), "Function must is list ['Function1', 'Function2'...]"
        processed_queue = list()
        self._init_some_method(self._functions)
        for _ in tqdm.tqdm(range(len(self.moles_set)), file=sys.stdout, disable=True):
            m = self.moles_set.pop(0)
            if not m.valid:
                processed_queue.append(m)
                continue
            self.smiles = m.smiles
            self.mol = m.mol
            self.function(self._functions)
            m.smiles = self.smiles
            m.mol = self.mol
            processed_queue.append(m)
        self.processed_queue = processed_queue
        return self.save_and_output()

    def save_and_output(self):
        assert self.output_type in ["Smiles", "Mol"], "Output type: Smiles / Mol"
        output_smiles = [i.smiles for i in self.processed_queue]
        output_mols = [i.mol for i in self.processed_queue]
        if self.output_type == "Smiles":
            output = output_smiles
        else:
            output = output_mols
        if self.save_name:
            with open("%s.pickle" % self.save_name, "wb") as F:
                pickle.dump([output_smiles, output_mols], F)
        return output

    def _init_some_method(self, functions):
        if "salt_remover" in functions:
            self.salt = SaltRemover.SaltRemover(defnData=args.salt_type)
        if "metal_disconnect" in functions:
            self.metal = metal.MetalDisconnector()
        if "neutralize_charge" in functions:
            self.charge = Uncharger(None)
        if "functional_group_optimization" in functions:
            self.functional_group = normalize.Normalizer()

    @staticmethod
    def molecules_load(queue):
        if not isinstance(queue, list):
            queue = [queue]
        if isinstance(queue[0], str):
            molecule_set = list()
            for idx, q in enumerate(queue):
                if len(q) == 0:
                    new_molecule = MoleculesStore(idx=idx)
                else:
                    try:
                        smiles = Chem.MolToSmiles(Chem.MolFromSmiles(q))
                        mol = Chem.MolFromSmiles(q)
                    except:
                        smiles = None
                        mol = None
                    new_molecule = MoleculesStore(idx=idx, smiles=smiles,
                                                  mol=mol)
                molecule_set.append(new_molecule)
            return molecule_set
        elif isinstance(queue[0], Chem.rdchem.Mol):
            molecule_set = list()
            for idx, q in enumerate(queue):
                try:
                    smiles = Chem.MolToSmiles(q)
                    mol = q
                except:
                    smiles = None
                    mol = None
                new_molecule = MoleculesStore(idx=idx, smiles=smiles, mol=mol)
                molecule_set.append(new_molecule)
            return molecule_set
        else:
            raise TypeError("Input Data must is SMILEs or Mol.")


if __name__ == "__main__":
    smiles_queue = ["CCCCCCCCC"]
    class Custom(MoleculePreprocess):
        def __init__(self, _input, output_type="Smiles", save_name: str = None):
            super().__init__(_input, output_type, save_name)

        def other_smiles_method(self, smiles: str):
            """
            Custom modules (input smiles)
            """
            return smiles

        def other_mol_method(self, mol: Chem.rdchem.Mol):
            """
            Custom modules (input mol)
            """
            return mol


    p = Custom(["mixture_remover",
                     "isomeric_remover",
                     "add_hydrogen",
                     "remove_hydrogen",
                     "salt_remover",
                     "metal_disconnect",
                     "atomic_number_limit",
                     "inorganic_remover",
                     "molecular_weight_filter",
                     "atomic_type_filter",
                     "neutralize_charge",
                     "functional_group_optimization"], output_type="Smiles", save_name="pk")
    out = p.pipline(smiles_queue)
