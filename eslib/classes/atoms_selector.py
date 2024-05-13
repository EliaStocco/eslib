import re
import numpy as np
from eslib.classes.aseio import integer_to_slice_string

class AtomSelector:
    def __init__(self):
        pass
    
    @staticmethod
    def select(selection_str, atoms_obj):
        # Helper function to extract the range from the selection string
        def extract_range(selection):
            if ':' in selection:
                start, end = map(int, selection.split(':'))
                return range(start, end + 1)
            else:
                index = int(selection)
                return range(index, index + 1)
        
        # Split the selection string by ';'
        selections = selection_str.split(';')
        selected_atoms = []
        
        for selection in selections:
            # Match atom type and index using regular expressions
            match = re.match(r'([A-Za-z]+)?(\d+)?(:\d+)?', selection)
            atom_type, index_str, range_str = match.groups()
            
            # Select atoms by type
            if atom_type:
                selected_atoms.extend([n for n,atom in enumerate(atoms_obj) if atom.symbol == atom_type])
            # Select atoms by index or range
            elif index_str:
                ii = integer_to_slice_string(selection)
                selected_atoms.extend([n for _,n in enumerate(np.arange(len(atoms_obj))[ii])])
            else:
                raise ValueError("coding error")
                
        
        return np.unique(selected_atoms)

def main():
    import numpy as np
    from ase.build import molecule
    num_atoms = 3
    box_size = 4
    positions = np.random.rand(num_atoms, 3) * box_size
    atoms = molecule('H2O')
    atoms.set_positions(positions)
    selector = AtomSelector()

    # Example selection strings
    selection_str = 'H;1:3;O;5'
    indices = selector.select(selection_str, atoms)
    print(indices)

if __name__ == "__main__":
    main()

