"""The characterizer module for the composite uFJC scission model 
implemented in the Unified Form Language (UFL) for FEniCS.
"""

# import necessary libraries
from __future__ import division
from .default_parameters import default_parameters
from .utility import generate_savedir


class CompositeuFJCScissionUFLFEniCSCharacterizer(object):
    """Characterizer class for composite uFJCs with scission implemented
    in the Unified Form Language (UFL) for FEniCS.
    """

    def __init__(self):
        """Initializes the
        ``CompositeuFJCScissionUFLFEniCSCharacterizer`` class, producing
        a characterization instance for the composite uFJC scission
        model implemented in the Unified Form Language (UFL) for FEniCS.
        """

        # Set default parameters and user-defined parameters
        self.parameters = default_parameters()
        self.set_user_parameters()

        # Setup filesystem
        self.savedir = generate_savedir(self.prefix())
    
    def set_user_parameters(self):
        """Set user-defined parameters.

        This function sets user-defined parameters via class
        inheritance.
        """
        pass

    def prefix(self):
        """Set characterization prefix.

        This function sets the prefix for the characterization, which is
        used as the name for the directory where finalized results are
        saved. Can be modified via class inheritance.
        """
        return "characterizer"

    def characterization(self):
        """Define characterization routine.
        
        This function defines the characterization routine via class
        inheritance.
        """
        pass

    def finalization(self):
        """Define finalization analysis.
        
        This function defines the finalization analysis via class
        inheritance.
        """
        pass