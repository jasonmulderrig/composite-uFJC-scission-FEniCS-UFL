"""The module for the composite uFJC scission model implemented in the
Unified Form Language (UFL) for FEniCS specifying the rate-dependent or
rate-independent nature of scission.
"""

# Import external modules
from __future__ import division
from dolfin import *


class RateIndependentScissionUFLFEniCS(object):
    """The composite uFJC scission model class implemented in the
    Unified Form Language (UFL) for FEniCS specifying rate-independent
    scission.

    This class contains methods specifying rate-independent scission for
    the composite uFJC chain model implemented in the Unified Form
    Language (UFL) for FEniCS, which involve defining both energetic and
    probabilistic quantities. Via class inheritance in the
    ``RateIndependentScissionCompositeuFJCUFLFEniCS`` class, the
    ``RateIndependentSmoothstepScissionCompositeuFJCUFLFEniCS`` class,
    and the ``RateIndependentSigmoidScissionCompositeuFJCUFLFEniCS``
    class, this class inherits all attributes and methods from the
    ``AnalyticalScissionCompositeuFJCUFLFEniCS`` class, the
    ``SmoothstepScissionCompositeuFJCUFLFEniCS`` class, and the
    ``SigmoidScissionCompositeuFJCUFLFEniCS`` class, and each of these
    classes inherit all attributes and methods from the
    ``CompositeuFJCUFLFEniCS`` class.
    """
    def __init__(self):
        pass

    def epsilon_nu_diss_hat_ufl_fenics_func(
            self, lmbda_nu_hat_max_val, lmbda_nu_hat_max_val_prior,
            lmbda_nu_hat_val, lmbda_nu_hat_val_prior,
            epsilon_nu_diss_hat_val_prior):
        """Nondimensional rate-independent dissipated segment scission
        energy.
        
        This function computes the nondimensional rate-independent
        dissipated segment scission energy as a function of its prior
        value, the current and prior values of the applied segment
        stretch, and the current and prior values of the maximum applied
        segment stretch. This function is implemented in the Unified
        Form Language (UFL) for FEniCS.
        """
        return (
            self.epsilon_nu_diss_hat_rate_independent_scission_ufl_fenics_func(
                lmbda_nu_hat_max_val, lmbda_nu_hat_max_val_prior,
                lmbda_nu_hat_val, lmbda_nu_hat_val_prior,
                epsilon_nu_diss_hat_val_prior)
        )
    
    def epsilon_cnu_diss_hat_ufl_fenics_func(
            self, lmbda_nu_hat_max_val, lmbda_nu_hat_max_val_prior,
            lmbda_nu_hat_val, lmbda_nu_hat_val_prior,
            epsilon_cnu_diss_hat_val_prior):
        """Nondimensional rate-independent dissipated chain scission
        energy per segment.
        
        This function computes the nondimensional rate-independent
        dissipated chain scission energy per segment as a function of
        its prior value, the current and prior values of the applied
        segment stretch, and the current and prior values of the maximum
        applied segment stretch. This function is implemented in the
        Unified Form Language (UFL) for FEniCS.
        """
        return (
            self.epsilon_cnu_diss_hat_rate_independent_scission_ufl_fenics_func(
                lmbda_nu_hat_max_val, lmbda_nu_hat_max_val_prior,
                lmbda_nu_hat_val, lmbda_nu_hat_val_prior,
                epsilon_cnu_diss_hat_val_prior)
        )
    

class RateDependentScissionUFLFEniCS(object):
    """The composite uFJC scission model class implemented in the
    Unified Form Language (UFL) for FEniCS specifying rate-dependent
    scission.

    This class contains methods specifying rate-dependent scission for
    the composite uFJC chain model implemented in the Unified Form
    Language (UFL) for FEniCS, which involve defining both energetic and
    probabilistic quantities. Via class inheritance in the
    ``RateDependentScissionCompositeuFJCUFLFEniCS`` class, the
    ``RateDependentSmoothstepScissionCompositeuFJCUFLFEniCS`` class,
    and the ``RateDependentSigmoidScissionCompositeuFJCUFLFEniCS``
    class, this class inherits all attributes and methods from the
    ``AnalyticalScissionCompositeuFJCUFLFEniCS`` class, the
    ``SmoothstepScissionCompositeuFJCUFLFEniCS`` class, and the
    ``SigmoidScissionCompositeuFJCUFLFEniCS`` class, and each of these
    classes inherit all attributes and methods from the
    ``CompositeuFJCUFLFEniCS`` class.
    """
    def __init__(self):
        pass
    
    def p_nu_sci_hat_cmltv_intgrl_ufl_fenics_func(
            self, p_nu_sci_hat_val, t_val, p_nu_sci_hat_val_prior, t_prior,
            p_nu_sci_hat_cmltv_intgrl_val_prior):
        """History-dependent time integral of the rate-independent
        probability of segment scission.
        
        This function computes the history-dependent time integral of
        the rate-independent probability of segment scission as a
        function of its prior value and the current and prior values of
        both the rate-independent probability of segment scission and
        time. This function is implemented in the Unified Form Language
        (UFL) for FEniCS.
        """
        # (Single segment) Trapezoidal rule of integration
        delta_p_nu_sci_hat_cmltv_intgrl_val = (
            (t_val-t_prior) * 0.5 * (p_nu_sci_hat_val_prior+p_nu_sci_hat_val)
        )
        return (
            p_nu_sci_hat_cmltv_intgrl_val_prior
            + delta_p_nu_sci_hat_cmltv_intgrl_val
        )
    
    def rho_nu_ufl_fenics_func(self, p_nu_sci_hat_cmltv_intgrl_val):
        """Rate-dependent probability of segment survival.
        
        This function computes the rate-dependent probability of segment
        survival as a function of the history-dependent time integral of
        the rate-independent probability of segment scission. This
        function is implemented in the Unified Form Language (UFL) for
        FEniCS.
        """
        return exp(-self.omega_0*p_nu_sci_hat_cmltv_intgrl_val)
    
    def gamma_nu_ufl_fenics_func(self, p_nu_sci_hat_cmltv_intgrl_val):
        """Rate-dependent probability of segment scission.
        
        This function computes the rate-dependent probability of segment
        scission as a function of the history-dependent time integral of
        the rate-independent probability of segment scission. This
        function is implemented in the Unified Form Language (UFL) for
        FEniCS.
        """
        return 1. - self.rho_nu_ufl_fenics_func(p_nu_sci_hat_cmltv_intgrl_val)
    
    def rho_nu_dot_ufl_fenics_func(
            self, p_nu_sci_hat_val, p_nu_sci_hat_cmltv_intgrl_val):
        """Time rate-of-change of the rate-dependent probability of
        segment survival.
        
        This function computes the time rate-of-change of the
        rate-dependent probability of segment survival as a function of
        the rate-independent probability of segment scission and the
        history-dependent time integral of the rate-independent
        probability of segment scission. This function is implemented in
        the Unified Form Language (UFL) for FEniCS.
        """
        return (
            -self.omega_0 * p_nu_sci_hat_val
            * self.rho_nu_ufl_fenics_func(p_nu_sci_hat_cmltv_intgrl_val)
        )
    
    def gamma_nu_dot_ufl_fenics_func(
            self, p_nu_sci_hat_val, p_nu_sci_hat_cmltv_intgrl_val):
        """Time rate-of-change of the rate-dependent probability of
        segment scission.
        
        This function computes the time rate-of-change of the
        rate-dependent probability of segment scission as a function of
        the rate-independent probability of segment scission and the
        history-dependent time integral of the rate-independent
        probability of segment scission. This function is implemented in
        the Unified Form Language (UFL) for FEniCS.
        """
        return (
            -self.rho_nu_dot_ufl_fenics_func(
                p_nu_sci_hat_val, p_nu_sci_hat_cmltv_intgrl_val)
        )
    
    def epsilon_nu_diss_hat_ufl_fenics_func(
            self, p_nu_sci_hat_val, p_nu_sci_hat_cmltv_intgrl_val,
            epsilon_nu_sci_hat_val, t_val, t_prior,
            epsilon_nu_diss_hat_val_prior):
        """Nondimensional rate-dependent dissipated segment scission
        energy.
        
        This function computes the nondimensional rate-dependent
        dissipated segment scission energy as a function of its prior
        value, the current nondimensional segment scission energy, the
        current and prior values of time, the current rate-independent
        probability of segment scission, and the history-dependent time
        integral of the rate-independent probability of segment
        scission. This function is implemented in the Unified Form
        Language (UFL) for FEniCS.
        """
        gamma_nu_dot_val = (
            self.gamma_nu_dot_ufl_fenics_func(
                p_nu_sci_hat_val, p_nu_sci_hat_cmltv_intgrl_val)
        )
        epsilon_nu_diss_hat_dot_val = gamma_nu_dot_val * epsilon_nu_sci_hat_val
        
        return (
            epsilon_nu_diss_hat_val_prior + epsilon_nu_diss_hat_dot_val
            * (t_val-t_prior)
        )
    
    def rho_c_ufl_fenics_func(self, p_nu_sci_hat_cmltv_intgrl_val):
        """Rate-dependent probability of chain survival.
        
        This function computes the rate-dependent probability of chain
        survival as a function of the history-dependent time integral of
        the rate-independent probability of segment scission. This
        function is implemented in the Unified Form Language (UFL) for
        FEniCS.
        """
        return exp(-self.nu*self.omega_0*p_nu_sci_hat_cmltv_intgrl_val)
    
    def gamma_c_ufl_fenics_func(self, p_nu_sci_hat_cmltv_intgrl_val):
        """Rate-dependent probability of chain scission.
        
        This function computes the rate-dependent probability of chain
        scission as a function of the history-dependent time integral of
        the rate-independent probability of segment scission. This
        function is implemented in the Unified Form Language (UFL) for
        FEniCS.
        """
        return 1. - self.rho_c_ufl_fenics_func(p_nu_sci_hat_cmltv_intgrl_val)
    
    def rho_c_dot_ufl_fenics_func(
            self, p_nu_sci_hat_val, p_nu_sci_hat_cmltv_intgrl_val):
        """Time rate-of-change of the rate-dependent probability of
        chain survival.
        
        This function computes the time rate-of-change of the
        rate-dependent probability of chain survival as a function of
        the rate-independent probability of segment scission and the
        history-dependent time integral of the rate-independent
        probability of segment scission. This function is implemented in
        the Unified Form Language (UFL) for FEniCS.
        """
        return (
            -self.nu * self.omega_0 * p_nu_sci_hat_val
            * self.rho_c_ufl_fenics_func(p_nu_sci_hat_cmltv_intgrl_val)
        )
    
    def gamma_c_dot_ufl_fenics_func(
            self, p_nu_sci_hat_val, p_nu_sci_hat_cmltv_intgrl_val):
        """Time rate-of-change of the rate-dependent probability of
        chain scission.
        
        This function computes the time rate-of-change of the
        rate-dependent probability of chain scission as a function of
        the rate-independent probability of segment scission and the
        history-dependent time integral of the rate-independent
        probability of segment scission. This function is implemented in
        the Unified Form Language (UFL) for FEniCS.
        """
        return (
            -self.rho_c_dot_ufl_fenics_func(
                p_nu_sci_hat_val, p_nu_sci_hat_cmltv_intgrl_val)
        )
    
    def epsilon_cnu_diss_hat_ufl_fenics_func(
            self, p_nu_sci_hat_val, p_nu_sci_hat_cmltv_intgrl_val,
            epsilon_cnu_sci_hat_val, t_val, t_prior,
            epsilon_cnu_diss_hat_val_prior):
        """Nondimensional rate-dependent dissipated chain scission
        energy per segment.
        
        This function computes the nondimensional rate-dependent
        dissipated chain scission energy per segment as a function of
        its prior value, the current nondimensional chain scission
        energy per segment, the current and prior values of time, the
        current rate-independent probability of segment scission, and
        the history-dependent time integral of the rate-independent
        probability of segment scission. This function is implemented in
        the Unified Form Language (UFL) for FEniCS.
        """
        gamma_c_dot_val = (
            self.gamma_c_dot_ufl_fenics_func(
                p_nu_sci_hat_val, p_nu_sci_hat_cmltv_intgrl_val)
        )
        epsilon_cnu_diss_hat_dot_val = gamma_c_dot_val * epsilon_cnu_sci_hat_val
        
        return (
            epsilon_cnu_diss_hat_val_prior + epsilon_cnu_diss_hat_dot_val
            * (t_val-t_prior)
        )