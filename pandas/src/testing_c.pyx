from pandas import compat
from pandas.core.common import isnull

import numpy as np
import time

cdef DECIMAL_5_CONSTANT = 0.5 * 10.0 ** -5
cdef DECIMAL_3_CONSTANT = 0.5 * 10.0 ** -3

cpdef assert_almost_equal(a, b, check_less_precise=False):
    cdef:
        EqualityWithError ret

    ret = _almost_equal(a, b, check_less_precise=check_less_precise)
    
    if not ret.is_equal:
        assert False, ret.message

    return True

cdef inline bint isiterable(obj):
    return hasattr(obj, '__iter__')

cdef bint assert_dict_equal(dict a, dict b, bint compare_keys=True):
    a_keys = frozenset(a.keys())
    b_keys = frozenset(b.keys())

    if compare_keys:
        if not a_keys == b_keys:
            return False

    for k in a_keys:
        if not _almost_equal(a[k], b[k]).is_equal:
            return False
    return True

cdef bint _decimal_almost_equal(double desired, double actual, int decimal):
    # Code from
    # http://docs.scipy.org/doc/numpy/reference/generated
    # /numpy.testing.assert_almost_equal.html#numpy.testing.assert_almost_equal
    cdef:
        double constant

    if decimal == 3:
        constant = DECIMAL_3_CONSTANT
    elif decimal == 5:
        constant = DECIMAL_5_CONSTANT
    else:
        constant = 0.5 * 10.0 ** -decimal

    return abs(desired - actual) < constant

cdef struct EqualityWithError:
    bint is_equal
    char* message

cdef inline EqualityWithError wrap_response(bint is_equal, char* message):
    cdef:
        EqualityWithError ret

    if is_equal:
        ret.is_equal = True
        ret.message = ''
    else:
        ret.is_equal = False
        ret.message = message

    return ret

cdef EqualityWithError _almost_equal(
        object a, object b, bint check_less_precise=False):
    cdef:
        int i, na, nb

    if isinstance(a, dict) or isinstance(b, dict):
        return wrap_response(
            assert_dict_equal(a, b), 'A'
        )

    if isinstance(a, compat.string_types):
        return wrap_response(a == b, 'B')

    if isiterable(a):
        if not isiterable(b):
            return wrap_response(False, 'C')

        na, nb = len(a), len(b)

        if not na == nb:
            return wrap_response(False, 'D')
        # TODO: Figure out why I thought this needed instance cheacks...
        # if (isinstance(a, np.ndarray) and isinstance(b, np.ndarray) and
        #     np.array_equal(a, b)):
        if np.array_equal(a, b):
            return wrap_response(True, 'E')
        for i in range(na):
            if not _almost_equal(a[i], b[i], check_less_precise).is_equal:
                return wrap_response(False, 'F')
        return wrap_response(True, 'G')

    if isnull(a):
        return wrap_response(isnull(b), 'H')
    elif isnull(b):
        return wrap_response(isnull(a), 'I')

    if isinstance(a, (bool, float, int, np.float32)):
        decimal = 5

        # deal with differing dtypes
        if check_less_precise:
            dtype_a = np.dtype(type(a))
            dtype_b = np.dtype(type(b))
            if dtype_a.kind == 'f' and dtype_b == 'f':
                if dtype_a.itemsize <= 4 and dtype_b.itemsize <= 4:
                    decimal = 3

        if np.isinf(a):
            return wrap_response(np.isinf(b), 'J')
        elif abs(a) < 1e-5:
            return wrap_response(_decimal_almost_equal(a, b, decimal), 'K')
        else:
            return wrap_response(_decimal_almost_equal(1, a / b, decimal), 'L')
    else:
        return wrap_response(a == b, 'M')
