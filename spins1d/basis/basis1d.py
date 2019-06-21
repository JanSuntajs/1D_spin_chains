import numpy as _np
from scipy.special import comb
import numba as nb


class basis_spin(object):

    def __init__(self, L, Nu=None, dtype=_np.uint32):
        super(basis_spin, self).__init__()

        self._dtype = dtype
        self.L = L
        self.Nu = Nu
        self._construct_basis()

    @property
    def L(self):
        return self.__L

    @L.setter
    def L(self, L):
        if type(L) is not int:
            raise TypeError("L must be an integer!")
        if L <= 0:
            raise ValueError("L must be greater than zero!")

        self.__L = L

    @property
    def Nu(self):

        return self.__Nu

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
        self.__Nu = Nu_list


    def _construct_basis(self):

        Nu = self.Nu

        # basis construction
        if Nu is None:

            self.Ns = 1 << self.L
            self.basis = _np.arange(0, 1 << self.L, 1, dtype=self._dtype)
            self._conserve_Nu = False

        else:

            # cumulative number of states in blocks with different
            # fermion numbers, starts with zero
            blocks = _np.insert(_np.cumsum([comb(self.L, numspin, True)
                                            for numspin in Nu]), 0, 0)

            # ordering of count blocks - Nf values should not repeat
            block_ind = {Nu: i for (i, Nu) in enumerate(Nu)}
            indices = _np.zeros((len(Nu)), dtype=self._dtype)

            self.Ns = blocks[-1]
            self.basis = _np.empty((self.Ns,), dtype=self._dtype)
            self._conserve_Nu = False

            for i in range(1 << self.L):

                count = bin(i).count("1")
                if count in Nu:

                    blk_cntr = block_ind[count]
                    state_idx = blocks[blk_cntr] + indices[blk_cntr]
                    self.basis[state_idx] = i
                    indices[blk_cntr] += 1
