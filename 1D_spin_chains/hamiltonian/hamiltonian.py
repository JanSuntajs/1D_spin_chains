"""
A module that implements the
hamiltonian class which constructs
the model hamiltonian from a given
set of input parameters.g

"""

import numpy as np
import scipy as sp
from scipy import sparse as ssp
import functools

import ham_ops


_ops = ham_ops.operators


class decorators_mixin(object):
    """
    A mixin class for decorators which
    are mainly used for input checking.

    """

    @classmethod
    def check_ham_lists(cls, decorated):
        """
        A decorator for checking whether
        the static and dynamic lists provided
        are of the proper shape and if the
        input is ok.

        """
        @functools.wraps(decorated)
        def wrap_check_ham_lists(*args):

            ham_list = args[1]
            for term in ham_list:

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

                # sort sites in the site coupling list
                # coups[:, 1:] = np.sort(coups[:, 1:], axis=1)
                term[1] = coups

            res = decorated(*args)
            return res

        return wrap_check_ham_lists


class hamiltonian(decorators_mixin):
    """
    Creates a class which constructs the
    spin chain hamiltonian.

    Parameters
    ----------

    L: int
        An integer specifying the spin chain length.

    static_list: list
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

    dynamic_list: list
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

    Nu: {int, None}
        Number of up spins, relevant for the hamiltonians where the
        total spin z projection is a conserved quantity. Defaults to
        None.


    """

    def __init__(self, L, static_list, dynamic_list, t=0, Nu=None):
        super(hamiltonian, self).__init__()

        self.L = L
        self.static_list = static_list
        self.dynamic_list = dynamic_list
        self.Nu = Nu

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
    def num_states(self):

        return 1 << self.L

    @property
    def Nu(self):

        return self._Nu

    @Nu.setter
    def Nu(self, Nu):
        if Nu is None:
            Nu_list = None
        elif type(Nu) is int:
            Nu_list = [Nu]
        else:
            try:
                Nu_list = list(Nu)
            except TypeError:
                raise TypeError(("Nf must be an"
                                 "iterable returning integers!"))

            if any((type(Nu) is not int) for Nu in Nu_list):
                raise TypeError("Nf must be iterable returning integers")
            if any(((Nu < 0) or (Nu >= self.L)) for Nu in Nu_list):
                raise ValueError("Nf cannot be greater than L or smaller than "
                                 "zero!")
            if len(Nu_list) != len(set(Nu_list)):
                raise ValueError(("There must be no duplicates"
                                  " in a list of Nf values!"))
        self._Nu = Nu_list

    @property
    def static_list(self):

        return self._static_list

    @static_list.setter
    @decorators_mixin.check_ham_lists
    def static_list(self, static_list):
        """
        Perform checking on the shapes and values
        of the static_ham input nested list.

        INPUT:

        static_ham: list
            A nested list of the operator description
            strings and site-coupling lists for the
            time-independent part of the Hamiltonian.
            See class' docstring for more details.

        """
        self._static_list = static_list

    # build the hamiltonian
    @property
    def _ham_stat(self):
        """
        Build the entire (static) hamiltonian from the static
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

        Returns
        -------

        ham_static: dict
            A dict of key-value pairs where keys are
            the operator descriptor strings and values
            are the hamiltonian terms

        """

        # initialize an empty dict
        ham_static = {}

        # iterate over different hamiltonian
        # terms in the static list
        for ham_term in self.static_list:

            static_key = ham_term[0]
            # the dimensionality of the default placeholder
            # Hamiltonian must match the Hilbert space dimension
            # which scales exponentially with system size as 2 ** L
            ham = 0 * ssp.eye(2 ** self.L)

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
                sites = np.sort(coupling[1:])
                # in the case of pbc, one needs to
                # take care of the operator ordering
                # if the operators "wrap around",
                # one needs to consider this and
                # properly reorder the operator
                # descriptor string list
                sites_sorted = np.argsort(coupling[1:])
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
                    temp_ = ssp.kron(eye, _ops[op_strings[sites_sorted[i]]])

                    # update the temporary kronecker product array
                    # with the new term
                    temp = ssp.kron(temp, temp_)

                temp = ssp.kron(temp, eyes[-1])

                ham += temp * exchange

            if static_key in ham_static.keys():
                static_key = static_key + '_'
            ham_static[static_key] = ham

        return ham_static

    @property
    def dynamic_list(self):

        return self._dynamic_list

    @dynamic_list.setter
    @decorators_mixin.check_ham_lists
    def dynamic_list(self, dynamic_list):
        """
        Perform checking on the shapes and values
        of the dynamic_ham input nested list.

        INPUT:

        dynamic_ham: list
            A nested list of the operator description
            strings and site-coupling lists for the
            time-independent part of the Hamiltonian.
            See class' docstring for more details.

        """
        self._dynamic_list = dynamic_list

    @property
    def ham(self):

        ham = 0

        for value in self._ham_stat.values():

            ham += value

        return ham

    # @property
    # def dynamic(self):

    #     return self._dynamic

    # @dynamic.setter
    # def dynamic(self, dynamic_ham):



