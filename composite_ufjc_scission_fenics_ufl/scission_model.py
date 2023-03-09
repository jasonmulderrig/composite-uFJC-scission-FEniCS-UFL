"""The module for the composite uFJC scission model implemented in the
Unified Form Language (UFL) for FEniCS specifying the fundamental
analytical scission model.
"""

# Import external modules
from __future__ import division
from dolfin import *

# Import internal modules
from .core import CompositeuFJCUFLFEniCS


class AnalyticalScissionCompositeuFJCUFLFEniCS(CompositeuFJCUFLFEniCS):
    """The composite uFJC scission model class implemented in the
    Unified Form Language (UFL) for FEniCS specifying the fundamental
    analytical scission model.

    This class contains methods specifying the fundamental scission 
    model implemented in the Unified Form Language (UFL) for FEniCS,
    which involve defining both energetic and probabilistic quantities.
    It inherits all attributes and methods from the
    ``CompositeuFJCUFLFEniCS`` class.
    """
    def __init__(self):
        """
        Initializes the ``AnalyticalScissionCompositeuFJCUFLFEniCS``
        class.
        
        Initialize and inherit all attributes and methods from the
        ``CompositeuFJCUFLFEniCS`` class instance.
        """
        CompositeuFJCUFLFEniCS.__init__(self)

    def u_nu_tot_hat_func(self, lmbda_nu_hat, lmbda_nu):
        """Nondimensional total segment potential under an applied chain
        force.
        
        This function computes the nondimensional total segment
        potential under an applied chain force as a function of the
        applied segment stretch and the segment stretch specifying a
        particular state in the energy landscape.
        """
        lmbda_c_eq_hat = self.lmbda_c_eq_func(lmbda_nu_hat)
        
        return (
            self.u_nu_func(lmbda_nu)
            - lmbda_nu * self.xi_c_func(lmbda_nu_hat, lmbda_c_eq_hat)
        )
    
    def u_nu_hat_func(self, lmbda_nu_hat, lmbda_nu):
        """Nondimensional total distorted segment potential under an
        applied chain force.
        
        This function computes the nondimensional total distorted
        segment potential under an applied chain force as a function
        of the applied segment stretch and the segment stretch
        specifying a particular state in the energy landscape.
        """
        lmbda_c_eq_hat = self.lmbda_c_eq_func(lmbda_nu_hat)
        
        return (
            self.u_nu_func(lmbda_nu)
            - (lmbda_nu-lmbda_nu_hat)
            * self.xi_c_func(lmbda_nu_hat, lmbda_c_eq_hat)
        )
    
    def lmbda_nu_locmin_hat_func(self, lmbda_nu_hat):
        """Segment stretch corresponding to the local minimum of the
        nondimensional total (distorted) segment potential under an 
        applied chain force.
        
        This function computes the segment stretch corresponding to the
        local minimum of the nondimensional total (distorted) segment
        potential under an applied chain force as a function of the
        applied segment stretch.
        """
        lmbda_c_eq_hat = self.lmbda_c_eq_func(lmbda_nu_hat)
        
        return (
            1. + self.xi_c_func(lmbda_nu_hat, lmbda_c_eq_hat) / self.kappa_nu
        )
    
    def lmbda_nu_locmax_hat_func(self, lmbda_nu_hat):
        """Segment stretch corresponding to the local maximum of the
        nondimensional total (distorted) segment potential under an 
        applied chain force.
        
        This function computes the segment stretch corresponding to the
        local maximum of the nondimensional total (distorted) segment
        potential under an applied chain force as a function of the
        applied segment stretch.
        """
        lmbda_c_eq_hat = self.lmbda_c_eq_func(lmbda_nu_hat)
        
        if lmbda_nu_hat <= 1.:
            return np.inf
        else:
            cbrt_arg_nmrtr = self.zeta_nu_char**2
            cbrt_arg_dnmntr = (
                self.kappa_nu * self.xi_c_func(lmbda_nu_hat, lmbda_c_eq_hat)
            )
            cbrt_arg = cbrt_arg_nmrtr / cbrt_arg_dnmntr
            return 1. + np.cbrt(cbrt_arg)
    
    def epsilon_nu_sci_hat_func(self, lmbda_nu_hat):
        """Nondimensional segment scission energy.
        
        This function computes the nondimensional segment scission
        energy as a function of the applied segment stretch.
        """
        lmbda_c_eq_hat = self.lmbda_c_eq_func(lmbda_nu_hat)
        
        return (
            self.psi_cnu_func(lmbda_nu_hat, lmbda_c_eq_hat) + self.zeta_nu_char
        )
    
    def epsilon_cnu_sci_hat_func(self, lmbda_nu_hat):
        """Nondimensional chain scission energy per segment.
        
        This function computes the nondimensional chain scission energy
        per segment as a function of the applied segment stretch.
        """
        return self.epsilon_nu_sci_hat_func(lmbda_nu_hat)
    
    def e_nu_sci_hat_func(self, lmbda_nu_hat):
        """Nondimensional segment scission activation energy barrier.
        
        This function computes the nondimensional segment scission
        activation energy barrier as a function of the applied segment
        stretch.
        """
        lmbda_nu_locmin_hat = self.lmbda_nu_locmin_hat_func(lmbda_nu_hat)
        lmbda_nu_locmax_hat = self.lmbda_nu_locmax_hat_func(lmbda_nu_hat)
        
        if lmbda_nu_hat <= 1:
            return self.zeta_nu_char
        elif lmbda_nu_hat <= self.lmbda_nu_crit:
            return (
                self.u_nu_hat_func(lmbda_nu_hat, lmbda_nu_locmax_hat)
                - self.u_nu_hat_func(lmbda_nu_hat, lmbda_nu_locmin_hat)
            )
        else:
            return 0.
    
    def p_nu_sci_hat_func(self, lmbda_nu_hat):
        """Rate-independent probability of segment scission.
        
        This function computes the rate-independent probability of
        segment scission as a function of the applied segment stretch.
        """
        return np.exp(-self.e_nu_sci_hat_func(lmbda_nu_hat))

    def p_nu_sur_hat_func(self, lmbda_nu_hat):
        """Rate-independent probability of segment survival.
        
        This function computes the rate-independent probability of
        segment survival as a function of the applied segment stretch.
        """
        return 1. - self.p_nu_sci_hat_func(lmbda_nu_hat)
    
    def p_c_sur_hat_func(self, lmbda_nu_hat):
        """Rate-independent probability of chain survival.
        
        This function computes the rate-independent probability of chain
        survival as a function of the applied segment stretch.
        """
        return self.p_nu_sur_hat_func(lmbda_nu_hat)**self.nu
    
    def p_c_sci_hat_func(self, lmbda_nu_hat):
        """Rate-independent probability of chain scission.
        
        This function computes the rate-independent probability of chain
        scission as a function of the applied segment stretch.
        """
        return 1. - self.p_c_sur_hat_func(lmbda_nu_hat)
    
    def epsilon_nu_sci_hat_analytical_func(self, lmbda_nu_hat):
        """Analytical form of the nondimensional segment scission
        energy function.
        
        This function computes the nondimensional segment scission
        energy as a function of the applied segment stretch.
        """
        return self.psi_cnu_analytical_func(lmbda_nu_hat) + self.zeta_nu_char
    
    def epsilon_cnu_sci_hat_analytical_func(self, lmbda_nu_hat):
        """Analytical form of the nondimensional chain scission
        energy function per segment.
        
        This function computes the nondimensional chain scission
        energy per segment as a function of the applied segment stretch.
        """
        return self.epsilon_nu_sci_hat_analytical_func(lmbda_nu_hat)
    
    def e_nu_sci_hat_analytical_func(self, lmbda_nu_hat):
        """Analytical form of the nondimensional segment scission
        activation energy barrier.
        
        This function computes the nondimensional segment scission
        activation energy barrier as a function of the applied segment
        stretch.
        """
        if lmbda_nu_hat <= 1. + self.cond_val:
            return self.zeta_nu_char
        elif lmbda_nu_hat <= self.lmbda_nu_crit:
            cbrt_arg = (
                self.zeta_nu_char**2 * self.kappa_nu * (lmbda_nu_hat-1.)**2
            )
            return (
                0.5 * self.kappa_nu * (lmbda_nu_hat-1.)**2
                - 1.5 * np.cbrt(cbrt_arg) + self.zeta_nu_char
            )
        else:
            return 0.
    
    def e_nu_sci_hat_prime_analytical_func(self, lmbda_nu_hat):
        """Analytical form of the derivative of the nondimensional
        segment scission activation energy barrier taken with respect to
        the applied segment stretch.
        
        This function computes the derivative of the nondimensional
        segment scission activation energy barrier taken with respect to
        the applied segment stretch as a function of applied segment
        stretch.
        """
        if lmbda_nu_hat <= 1. + self.cond_val:
            return -np.inf
        elif lmbda_nu_hat <= self.lmbda_nu_crit:
            cbrt_arg = self.zeta_nu_char**2 * self.kappa_nu / (lmbda_nu_hat-1.)
            return self.kappa_nu * (lmbda_nu_hat-1.) - np.cbrt(cbrt_arg)
        else:
            return 0.
    
    def p_nu_sci_hat_analytical_func(self, lmbda_nu_hat):
        """Analytical form of the rate-independent probability of
        segment scission.
        
        This function computes the rate-independent probability of
        segment scission as a function of the applied segment stretch.
        """
        return np.exp(-self.e_nu_sci_hat_analytical_func(lmbda_nu_hat))

    def p_nu_sur_hat_analytical_func(self, lmbda_nu_hat):
        """Analytical form of the rate-independent probability of
        segment survival.
        
        This function computes the rate-independent probability of
        segment survival as a function of the applied segment stretch.
        """
        return 1. - self.p_nu_sci_hat_analytical_func(lmbda_nu_hat)
    
    def p_c_sur_hat_analytical_func(self, lmbda_nu_hat):
        """Analytical form of the rate-independent probability of
        chain survival.
        
        This function computes the rate-independent probability of
        chain survival as a function of the applied segment stretch.
        """
        return self.p_nu_sur_hat_analytical_func(lmbda_nu_hat)**self.nu
    
    def p_c_sci_hat_analytical_func(self, lmbda_nu_hat):
        """Analytical form of the rate-independent probability of
        chain scission.
        
        This function computes the rate-independent probability of
        chain scission as a function of the applied segment stretch.
        """
        return 1. - self.p_c_sur_hat_analytical_func(lmbda_nu_hat)
    
    def lmbda_nu_p_nu_sci_hat_analytical_func(self, p_nu_sci_hat_val):
        """Segment stretch as a function of the rate-independent
        probability of segment scission.
        
        This function calculates the segment stretch as a function of
        the rate-independent probability of segment scission. This
        calculation is based upon the analytical form of the
        nondimensional segment scission activation energy barrier.
        """
        pi_tilde = -3. * np.cbrt((self.zeta_nu_char/self.kappa_nu)**2)

        rho_tilde = (
            2 / self.kappa_nu * (self.zeta_nu_char+np.log(p_nu_sci_hat_val))
        )
        arccos_arg = 3. * rho_tilde / (2.*pi_tilde) * np.sqrt(-3./pi_tilde)
        cos_arg = 1. / 3. * np.arccos(arccos_arg) - 2. * np.pi / 3.

        phi_tilde =  2. * np.sqrt(-pi_tilde/3.) * np.cos(cos_arg)

        return 1. + np.sqrt(phi_tilde**3)
    
    def lmbda_nu_p_c_sci_hat_analytical_func(self, p_c_sci_hat_val):
        """Segment stretch as a function of the rate-independent
        probability of chain scission.
        
        This function calculates the segment stretch as a function of
        the rate-independent probability of chain scission. This
        calculation is based upon the analytical form of the
        nondimensional segment scission activation energy barrier.
        """
        p_nu_sci_hat_val = 1. - (1.-p_c_sci_hat_val)**(1./self.nu)

        return self.lmbda_nu_p_nu_sci_hat_analytical_func(p_nu_sci_hat_val)

    def p_nu_sci_hat_prime_analytical_func(self, lmbda_nu_hat):
        """Analytical form of the derivative of the rate-independent
        probability of segment scission taken with respect to the
        applied segment stretch.
        
        This function computes the derivative of the rate-independent
        probability of segment scission taken with respect to the
        applied segment stretch as a function of applied segment
        stretch.
        """
        if lmbda_nu_hat <= 1. + self.cond_val:
            return 0.
        else:
            return (
                -self.p_nu_sci_hat_analytical_func(lmbda_nu_hat)
                * self.e_nu_sci_hat_prime_analytical_func(lmbda_nu_hat)
            )
    
    def p_nu_sur_hat_prime_analytical_func(self, lmbda_nu_hat):
        """Analytical form of the derivative of the rate-independent
        probability of segment survival taken with respect to the
        applied segment stretch.
        
        This function computes the derivative of the rate-independent
        probability of segment survival taken with respect to the
        applied segment stretch as a function of applied segment
        stretch.
        """
        if lmbda_nu_hat <= 1. + self.cond_val:
            return 0.
        else:
            return -self.p_nu_sci_hat_prime_analytical_func(lmbda_nu_hat)
    
    def p_c_sur_hat_prime_analytical_func(self, lmbda_nu_hat):
        """Analytical form of the derivative of the rate-independent
        probability of chain survival taken with respect to the applied
        segment stretch.
        
        This function computes the derivative of the rate-independent
        probability of chain survival taken with respect to the applied
        segment stretch as a function of applied segment stretch.
        """
        if lmbda_nu_hat <= 1. + self.cond_val:
            return 0.
        else:
            return (
                -self.nu
                * (1.-self.p_nu_sci_hat_analytical_func(lmbda_nu_hat))**(self.nu-1)
                * self.p_nu_sci_hat_prime_analytical_func(lmbda_nu_hat)
            )

    def p_c_sci_hat_prime_analytical_func(self, lmbda_nu_hat):
        """Analytical form of the derivative of the rate-independent
        probability of chain scission taken with respect to the applied
        segment stretch.
        
        This function computes the derivative of the rate-independent
        probability of chain scission taken with respect to the applied
        segment stretch as a function of applied segment stretch.
        """
        if lmbda_nu_hat <= 1. + self.cond_val:
            return 0.
        else:
            return -self.p_c_sur_hat_prime_analytical_func(lmbda_nu_hat)

class SmoothstepScissionCompositeuFJCUFLFEniCS(CompositeuFJCUFLFEniCS):
    """The composite uFJC scission model class implemented in the
    Unified Form Language (UFL) for FEniCS specifying the smoothstep
    scission model.

    This class contains methods specifying the smoothstep scission
    model implemented in the Unified Form Language (UFL) for FEniCS,
    which involve defining both energetic and probabilistic quantities.
    It inherits all attributes and methods from the
    ``CompositeuFJCUFLFEniCS`` class.
    """
    def __init__(self):
        """
        Initializes the ``SmoothstepScissionCompositeuFJCUFLFEniCS``
        class.
        
        Initialize and inherit all attributes and methods from the
        ``CompositeuFJCUFLFEniCS`` class instance.
        """
        CompositeuFJCUFLFEniCS.__init__(self)

    def epsilon_nu_sci_hat_func(self, lmbda_nu_hat):
        """Nondimensional segment scission energy.
        
        This function computes the nondimensional segment scission
        energy as a function of the applied segment stretch.
        """
        lmbda_c_eq_hat = self.lmbda_c_eq_func(lmbda_nu_hat)
        
        return (
            self.psi_cnu_func(lmbda_nu_hat, lmbda_c_eq_hat) + self.zeta_nu_char
        )
    
    def epsilon_cnu_sci_hat_func(self, lmbda_nu_hat):
        """Nondimensional chain scission energy per segment.
        
        This function computes the nondimensional chain scission energy
        per segment as a function of the applied segment stretch.
        """
        return self.epsilon_nu_sci_hat_func(lmbda_nu_hat)
    
    def p_nu_sci_hat_func(self, lmbda_nu_hat):
        """Rate-independent probability of segment scission.
        
        This function computes the rate-independent probability of
        segment scission as a function of the applied segment stretch.
        """
        if lmbda_nu_hat < self.lmbda_nu_crit_min:
            return 0.
        elif lmbda_nu_hat > self.lmbda_nu_crit_max:
            return 1.
        else:
            x = (
                (lmbda_nu_hat-self.lmbda_nu_crit_min)
                / (self.lmbda_nu_crit_max-self.lmbda_nu_crit_min)
            )
            return (
                70 * x**9 - 315 * x**8 + 540 * x**7 - 420 * x**6 + 126 * x**5
            )

    def p_nu_sur_hat_func(self, lmbda_nu_hat):
        """Rate-independent probability of segment survival.
        
        This function computes the rate-independent probability of
        segment survival as a function of the applied segment stretch.
        """
        return 1. - self.p_nu_sci_hat_func(lmbda_nu_hat)
    
    def p_c_sur_hat_func(self, lmbda_nu_hat):
        """Rate-independent probability of chain survival.
        
        This function computes the rate-independent probability of chain
        survival as a function of the applied segment stretch.
        """
        return self.p_nu_sur_hat_func(lmbda_nu_hat)**self.nu
    
    def p_c_sci_hat_func(self, lmbda_nu_hat):
        """Rate-independent probability of chain scission.
        
        This function computes the rate-independent probability of chain
        scission as a function of the applied segment stretch.
        """
        return 1. - self.p_c_sur_hat_func(lmbda_nu_hat)
    
    def p_nu_sci_hat_prime_func(self, lmbda_nu_hat):
        """Derivative of the rate-independent probability of segment
        scission taken with respect to the applied segment stretch.
        
        This function computes the derivative of the rate-independent
        probability of segment scission taken with respect to the
        applied segment stretch as a function of applied segment
        stretch.
        """
        if (lmbda_nu_hat < self.lmbda_nu_crit_min or 
            lmbda_nu_hat > self.lmbda_nu_crit_max):
            return 0.
        else:
            x = (
                (lmbda_nu_hat-self.lmbda_nu_crit_min)
                / (self.lmbda_nu_crit_max-self.lmbda_nu_crit_min)
            )
            return (
                (1./(self.lmbda_nu_crit_max-self.lmbda_nu_crit_min))
                * (630*x**8-2520*x**7+3780*x**6-2520*x**5+630*x**4)
            )
    
    def p_nu_sur_hat_prime_func(self, lmbda_nu_hat):
        """Derivative of the rate-independent probability of segment
        survival taken with respect to the applied segment stretch.
        
        This function computes the derivative of the rate-independent
        probability of segment survival taken with respect to the
        applied segment stretch as a function of applied segment
        stretch.
        """
        return -self.p_nu_sci_hat_prime_func(lmbda_nu_hat)
    
    def p_c_sur_hat_prime_func(self, lmbda_nu_hat):
        """Derivative of the rate-independent probability of chain
        survival taken with respect to the applied segment stretch.
        
        This function computes the derivative of the rate-independent
        probability of chain survival taken with respect to the applied
        segment stretch as a function of applied segment stretch.
        """
        return (
                -self.nu
                * (1.-self.p_nu_sci_hat_func(lmbda_nu_hat))**(self.nu-1)
                * self.p_nu_sci_hat_prime_func(lmbda_nu_hat)
            )

    def p_c_sci_hat_prime_func(self, lmbda_nu_hat):
        """Derivative of the rate-independent probability of chain
        scission taken with respect to the applied segment stretch.
        
        This function computes the derivative of the rate-independent
        probability of chain scission taken with respect to the applied
        segment stretch as a function of applied segment stretch.
        """
        return -self.p_c_sur_hat_prime_func(lmbda_nu_hat)
    
    def epsilon_nu_sci_hat_analytical_func(self, lmbda_nu_hat):
        """Analytical form of the nondimensional segment scission
        energy function.
        
        This function computes the nondimensional segment scission
        energy as a function of the applied segment stretch.
        """
        return self.psi_cnu_analytical_func(lmbda_nu_hat) + self.zeta_nu_char
    
    def epsilon_cnu_sci_hat_analytical_func(self, lmbda_nu_hat):
        """Analytical form of the nondimensional chain scission
        energy function per segment.
        
        This function computes the nondimensional chain scission
        energy per segment as a function of the applied segment stretch.
        """
        return self.epsilon_nu_sci_hat_analytical_func(lmbda_nu_hat)
    
    def p_nu_sci_hat_analytical_func(self, lmbda_nu_hat):
        """Analytical form of the rate-independent probability of
        segment scission.
        
        This function computes the rate-independent probability of
        segment scission as a function of the applied segment stretch.
        """
        return self.p_nu_sci_hat_func(lmbda_nu_hat)

    def p_nu_sur_hat_analytical_func(self, lmbda_nu_hat):
        """Analytical form of the rate-independent probability of
        segment survival.
        
        This function computes the rate-independent probability of
        segment survival as a function of the applied segment stretch.
        """
        return self.p_nu_sur_hat_func(lmbda_nu_hat)
    
    def p_c_sur_hat_analytical_func(self, lmbda_nu_hat):
        """Analytical form of the rate-independent probability of
        chain survival.
        
        This function computes the rate-independent probability of
        chain survival as a function of the applied segment stretch.
        """
        return self.p_c_sur_hat_func(lmbda_nu_hat)
    
    def p_c_sci_hat_analytical_func(self, lmbda_nu_hat):
        """Analytical form of the rate-independent probability of
        chain scission.
        
        This function computes the rate-independent probability of
        chain scission as a function of the applied segment stretch.
        """
        return self.p_c_sci_hat_func( lmbda_nu_hat)
    
    def p_nu_sci_hat_prime_analytical_func(self, lmbda_nu_hat):
        """Analytical form of the derivative of the rate-independent
        probability of segment scission taken with respect to the
        applied segment stretch.
        
        This function computes the derivative of the rate-independent
        probability of segment scission taken with respect to the
        applied segment stretch as a function of applied segment
        stretch.
        """
        return self.p_nu_sci_hat_prime_func(lmbda_nu_hat)
    
    def p_nu_sur_hat_prime_analytical_func(self, lmbda_nu_hat):
        """Analytical form of the derivative of the rate-independent
        probability of segment survival taken with respect to the
        applied segment stretch.
        
        This function computes the derivative of the rate-independent
        probability of segment survival taken with respect to the
        applied segment stretch as a function of applied segment
        stretch.
        """
        return self.p_nu_sur_hat_prime_func(lmbda_nu_hat)
    
    def p_c_sur_hat_prime_analytical_func(self, lmbda_nu_hat):
        """Analytical form of the derivative of the rate-independent
        probability of chain survival taken with respect to the applied
        segment stretch.
        
        This function computes the derivative of the rate-independent
        probability of chain survival taken with respect to the applied
        segment stretch as a function of applied segment stretch.
        """
        return self.p_c_sur_hat_prime_func(lmbda_nu_hat)

    def p_c_sci_hat_prime_analytical_func(self, lmbda_nu_hat):
        """Analytical form of the derivative of the rate-independent
        probability of chain scission taken with respect to the applied
        segment stretch.
        
        This function computes the derivative of the rate-independent
        probability of chain scission taken with respect to the applied
        segment stretch as a function of applied segment stretch.
        """
        return self.p_c_sci_hat_prime_func(lmbda_nu_hat)

class SigmoidScissionCompositeuFJCUFLFEniCS(CompositeuFJCUFLFEniCS):
    """The composite uFJC scission model class implemented in the
    Unified Form Language (UFL) for FEniCS specifying the sigmoid
    scission model.

    This class contains methods specifying the sigmoid scission
    model implemented in the Unified Form Language (UFL) for FEniCS,
    which involve defining both energetic and probabilistic quantities.
    It inherits all attributes and methods from the
    ``CompositeuFJCUFLFEniCS`` class.
    """
    def __init__(self):
        """
        Initializes the ``SigmoidScissionCompositeuFJCUFLFEniCS`` class.
        
        Initialize and inherit all attributes and methods from the
        ``CompositeuFJCUFLFEniCS`` class instance.
        """
        CompositeuFJCUFLFEniCS.__init__(self)

    def epsilon_nu_sci_hat_func(self, lmbda_nu_hat):
        """Nondimensional segment scission energy.
        
        This function computes the nondimensional segment scission
        energy as a function of the applied segment stretch.
        """
        lmbda_c_eq_hat = self.lmbda_c_eq_func(lmbda_nu_hat)
        
        return (
            self.psi_cnu_func(lmbda_nu_hat, lmbda_c_eq_hat) + self.zeta_nu_char
        )
    
    def epsilon_cnu_sci_hat_func(self, lmbda_nu_hat):
        """Nondimensional chain scission energy per segment.
        
        This function computes the nondimensional chain scission energy
        per segment as a function of the applied segment stretch.
        """
        return self.epsilon_nu_sci_hat_func(lmbda_nu_hat)
    
    def p_nu_sci_hat_func(self, lmbda_nu_hat):
        """Rate-independent probability of segment scission.
        
        This function computes the rate-independent probability of
        segment scission as a function of the applied segment stretch.
        """
        exp_arg = -self.tau * (lmbda_nu_hat-self.lmbda_nu_check)
        sqrt_arg = 1. - 1. / (1.+np.exp(exp_arg))
        return 1. - np.sqrt(sqrt_arg)

    def p_nu_sur_hat_func(self, lmbda_nu_hat):
        """Rate-independent probability of segment survival.
        
        This function computes the rate-independent probability of
        segment survival as a function of the applied segment stretch.
        """
        return 1. - self.p_nu_sci_hat_func(lmbda_nu_hat)
    
    def p_c_sur_hat_func(self, lmbda_nu_hat):
        """Rate-independent probability of chain survival.
        
        This function computes the rate-independent probability of chain
        survival as a function of the applied segment stretch.
        """
        return self.p_nu_sur_hat_func(lmbda_nu_hat)**self.nu
    
    def p_c_sci_hat_func(self, lmbda_nu_hat):
        """Rate-independent probability of chain scission.
        
        This function computes the rate-independent probability of chain
        scission as a function of the applied segment stretch.
        """
        return 1. - self.p_c_sur_hat_func(lmbda_nu_hat)
    
    def p_nu_sci_hat_prime_func(self, lmbda_nu_hat):
        """Derivative of the rate-independent probability of segment
        scission taken with respect to the applied segment stretch.
        
        This function computes the derivative of the rate-independent
        probability of segment scission taken with respect to the
        applied segment stretch as a function of applied segment
        stretch.
        """
        exp_arg = -self.tau * (lmbda_nu_hat-self.lmbda_nu_check)
        sqrt_arg = 1. - 1. / (1.+np.exp(exp_arg))
        trm_i = self.tau / 2.
        trm_ii = 1. / np.sqrt(sqrt_arg)
        trm_iii = 1. / (1.+np.exp(exp_arg))**2
        trm_iv = np.exp(exp_arg)
        return trm_i * trm_ii * trm_iii * trm_iv
    
    def p_nu_sur_hat_prime_func(self, lmbda_nu_hat):
        """Derivative of the rate-independent probability of segment
        survival taken with respect to the applied segment stretch.
        
        This function computes the derivative of the rate-independent
        probability of segment survival taken with respect to the
        applied segment stretch as a function of applied segment
        stretch.
        """
        return -self.p_nu_sci_hat_prime_func(lmbda_nu_hat)
    
    def p_c_sur_hat_prime_func(self, lmbda_nu_hat):
        """Derivative of the rate-independent probability of chain
        survival taken with respect to the applied segment stretch.
        
        This function computes the derivative of the rate-independent
        probability of chain survival taken with respect to the applied
        segment stretch as a function of applied segment stretch.
        """
        return (
                -self.nu
                * (1.-self.p_nu_sci_hat_func(lmbda_nu_hat))**(self.nu-1)
                * self.p_nu_sci_hat_prime_func(lmbda_nu_hat)
            )

    def p_c_sci_hat_prime_func(self, lmbda_nu_hat):
        """Derivative of the rate-independent probability of chain
        scission taken with respect to the applied segment stretch.
        
        This function computes the derivative of the rate-independent
        probability of chain scission taken with respect to the applied
        segment stretch as a function of applied segment stretch.
        """
        return -self.p_c_sur_hat_prime_func(lmbda_nu_hat)
    
    def epsilon_nu_sci_hat_analytical_func(self, lmbda_nu_hat):
        """Analytical form of the nondimensional segment scission
        energy function.
        
        This function computes the nondimensional segment scission
        energy as a function of the applied segment stretch.
        """
        return self.psi_cnu_analytical_func(lmbda_nu_hat) + self.zeta_nu_char
    
    def epsilon_cnu_sci_hat_analytical_func(self, lmbda_nu_hat):
        """Analytical form of the nondimensional chain scission
        energy function per segment.
        
        This function computes the nondimensional chain scission
        energy per segment as a function of the applied segment stretch.
        """
        return self.epsilon_nu_sci_hat_analytical_func(lmbda_nu_hat)
    
    def p_nu_sci_hat_analytical_func(self, lmbda_nu_hat):
        """Analytical form of the rate-independent probability of
        segment scission.
        
        This function computes the rate-independent probability of
        segment scission as a function of the applied segment stretch.
        """
        return self.p_nu_sci_hat_func(lmbda_nu_hat)

    def p_nu_sur_hat_analytical_func(self, lmbda_nu_hat):
        """Analytical form of the rate-independent probability of
        segment survival.
        
        This function computes the rate-independent probability of
        segment survival as a function of the applied segment stretch.
        """
        return self.p_nu_sur_hat_func(lmbda_nu_hat)
    
    def p_c_sur_hat_analytical_func(self, lmbda_nu_hat):
        """Analytical form of the rate-independent probability of
        chain survival.
        
        This function computes the rate-independent probability of
        chain survival as a function of the applied segment stretch.
        """
        return self.p_c_sur_hat_func(lmbda_nu_hat)
    
    def p_c_sci_hat_analytical_func(self, lmbda_nu_hat):
        """Analytical form of the rate-independent probability of
        chain scission.
        
        This function computes the rate-independent probability of
        chain scission as a function of the applied segment stretch.
        """
        return self.p_c_sci_hat_func( lmbda_nu_hat)
    
    def p_nu_sci_hat_prime_analytical_func(self, lmbda_nu_hat):
        """Analytical form of the derivative of the rate-independent
        probability of segment scission taken with respect to the
        applied segment stretch.
        
        This function computes the derivative of the rate-independent
        probability of segment scission taken with respect to the
        applied segment stretch as a function of applied segment
        stretch.
        """
        return self.p_nu_sci_hat_prime_func(lmbda_nu_hat)
    
    def p_nu_sur_hat_prime_analytical_func(self, lmbda_nu_hat):
        """Analytical form of the derivative of the rate-independent
        probability of segment survival taken with respect to the
        applied segment stretch.
        
        This function computes the derivative of the rate-independent
        probability of segment survival taken with respect to the
        applied segment stretch as a function of applied segment
        stretch.
        """
        return self.p_nu_sur_hat_prime_func(lmbda_nu_hat)
    
    def p_c_sur_hat_prime_analytical_func(self, lmbda_nu_hat):
        """Analytical form of the derivative of the rate-independent
        probability of chain survival taken with respect to the applied
        segment stretch.
        
        This function computes the derivative of the rate-independent
        probability of chain survival taken with respect to the applied
        segment stretch as a function of applied segment stretch.
        """
        return self.p_c_sur_hat_prime_func(lmbda_nu_hat)

    def p_c_sci_hat_prime_analytical_func(self, lmbda_nu_hat):
        """Analytical form of the derivative of the rate-independent
        probability of chain scission taken with respect to the applied
        segment stretch.
        
        This function computes the derivative of the rate-independent
        probability of chain scission taken with respect to the applied
        segment stretch as a function of applied segment stretch.
        """
        return self.p_c_sci_hat_prime_func(lmbda_nu_hat)