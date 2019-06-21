"""
A module with functions that enable
the control of the number of threads
used in numpy and scipy routines.

"""

import ctypes

mkl_rt = ctypes.CDLL('libmkl_rt.so')


def mkl_set_num_threads(cores):
    # Set # of MKL threads
    mkl_rt.MKL_Set_Num_Threads(cores)


def mkl_get_max_threads():
    # # of used MKL threads
    print(mkl_rt.MKL_Get_Max_Threads())
