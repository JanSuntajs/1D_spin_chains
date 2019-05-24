"""
A module that implements the
hamiltonian class which constructs
the model hamiltonian from a given
set of input parameters.

"""

import numpy as np
import scipy as sp
from scipy import sparse as ssp

import ham_ops


_ops = ham_ops.operators


class hamiltonian(object):
    """
    Creates a class which constructs the
    spin chain hamiltonian.

    Parameters
    ----------

    L: int
        An integer specifying the spin chain length.

    static_ham: list
        A nested list of the operator description strings
        and site-coupling lists for the time-independent
        part of the Hamiltonian. An example of the
        static_ham list would be:

            static_ham = [['zz', J_zz]]

        Here, 'zz' is the operator descriptor string specifiying
        2-spin interaction along the z-axis direction. For a chain-
        of L sites with constant nearest-neighbour exchange
        J and PBC, the site coupling list would be given by:

            J_zz = [[J, i, (i+1)%L] for i in range(L)]

        In the upper expression, J is the term describing
        the interaction strength and the following entries
        in the inner list specify the positions of the coupled
        sites. The upper example should serve as a general
        template which should allow for simple extension to
        the case of n-spin interaction and varying couplings.

    dynamic_ham: list
        A nested list of the operator description strings. The
        description is similar to the static_ham description,
        however, additional terms are needed to incorporate
        the time dependence: an example would be:

            dynamic_ham = [['zz', J_zz, time_fun, time_fun_args]]

        Here, 'zz' is the same operator descriptor string as the
        one in the static case. J_zz, however, now refers to
        the initial (dynamic) site coupling list at t=0. time_fun
        is a function object describing a protocol for
        time-dependent part of the hamiltonian with the following
        interface: time_fun(t, *time_fun_args) where time_fun_args
        are the possible additional arguments of the time-dependence.




    """

    def __init__(self, L, static_ham, dynamic_ham, Nu=None):
        super(hamiltonian, self).__init__()

        self.L = L
        self.static = static_ham
        self.dynamic = dynamic_ham

    @property
    def L(self):
        return self._L

    @L.setter
    def L(self, L):
        if type(L) is not int:
            raise TypeError("L must be an integer!")
        if L <= 0:
            raise ValueError("L must be greater than zero!")

        self._L = L

    @property
    def static(self):

        return self._static

    @static.setter
    def static(self, static_ham):
        """
        Perform checking on the shapes and values
        of the static_ham input nested list.

        """

        for term in static_ham:

            # operator descriptions and coupling lists
            op_desc, coups = term

            # check if the operator descriptors are ok
            if not isinstance(op_desc, str):
                raise TypeError(('Operator descriptor {} should'
                                 'be a string!').format(op_desc))

            # correct all preceeding/trailing whitespaces, if needed
            term[0] = op_desc.strip('')

            # number of interacting spins in the hamiltonian term
            n_inter = len(list(term[0]))

            # check if the entries in the operator descriptor list are ok
            if not set(list(op_desc)).issubset(_ops.keys()):
                raise ValueError(('Operator descriptor {}'
                                  ' contains invalid entries.'
                                  ' Allowed values: {}'
                                  ).format(op_desc, list(_ops.keys())))

            coups = np.array(coups)

            if coups[:, 1:].shape[1] != n_inter:
                raise ValueError(('Number of sites in '
                                  'the site-coupling list '
                                  'should match the number of terms '
                                  'in the operator descriptor string!'))

            coups[:, 1:] = np.sort(coups[:, 1:], axis=1)
            term[1] = coups

        self._static = static_ham

    # build the hamiltonian
    def _build_hamiltonian(self):
        """
        Build the entire hamiltonian from the static
        list.

        The idea of this code is to build the entire
        hamiltonian as a tensor product of single spin
        operators which automatically also ensures the
        validity of periodic boundary conditions if those
        are specified. No special PBC flag is needed in
        this case, one only needs to properly format
        the couplings list.

        In case we had a Hamiltonian defined on a chain
        of length L = 5 with two spins on sites 1 and 3
        interacting via exchange interaction along the z-axis,
        we would do the following:

        Id_2 x Sz x Id_2 x Sz x Id_2

        Here x denotes the tensor product of the Hilbert
        spaces, Id_2 is the identity over a single spin
        Hilbert space and we have enumerated the states
        according to python's indexing (0, 1, ... , L - 1)

        """

        # the dimensionality of the default placeholder
        # Hamiltonian must match the Hilbert space dimension
        # which scales exponentially with system size as 2 ** L
        ham = 0 * ssp.eye(2 ** self.L)

        # iterate over different hamiltonian
        # terms in the static list
        for ham_term in self.static:

            # which operators comprise the
            # given Hamiltonian term
            op_strings = list(ham_term[0])
            # coupling constants and sites
            couplings = ham_term[1]

            for coupling in couplings:

                # first entry of the coupling array
                # is the exchange constant
                exchange = coupling[0]
                # the remaining entries of the coupling
                # array are the sites with nontrivial
                # (non-identity) operators
                sites = coupling[1:]

                # determine the dimensionalities of the
                # intermediate identity operators which
                # 'act' between the spin operators at
                # specified sites
                dims = np.diff(sites) - 1
                # make sure that boundary cases are also
                # properly considered
                dims = np.insert(dims, 0, sites[0])
                dims = np.append(dims, self.L - 1 - sites[-1])

                # create the intermediate identity matrices
                eyes = [ssp.eye(2 ** dim) for dim in dims]
                # NOTE: the above construction ensures that the
                # cases where nontrivial operators are not present
                # at the edge sites are also properly considered

                # now the actual hamiltonian construction
                # temporary kronecker product matrix
                temp = ssp.eye(1)  # defaults to an identity
                for i, eye in enumerate(eyes[:-1]):

                    # an iterative step term -> consisting
                    # of an identity matrix and an operator
                    temp_ = ssp.kron(eye, _ops[op_strings[i]])

                    # update the temporary kronecker product array
                    # with the new term
                    temp = ssp.kron(temp, temp_)

                temp = ssp.kron(temp, eyes[-1])

                ham += temp * exchange

            self.ham = ham
