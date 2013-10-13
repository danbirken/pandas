#!/usr/bin/python
# -*- coding: utf-8 -*-
import pandas as pd
import unittest
import warnings
import nose
import numpy as np
import sys

from pandas.util.testing import (
    assert_almost_equal, assertRaisesRegexp, raise_with_traceback
)

# let's get meta.

class TestAssertAlmostEqual(unittest.TestCase):
    _multiprocess_can_split_ = True

    def _assert_not_almost_equal(self, a, b, **kwargs):
        self.assertRaises(AssertionError, assert_almost_equal, a, b, **kwargs)

    def test_assert_almost_equal_numbers(self):
        assert_almost_equal(1.1, 1.1)
        assert_almost_equal(1.1, 1.100001)
        assert_almost_equal(0, 0)
        assert_almost_equal(0, 0.000001)
        assert_almost_equal(0.000001, 0)

        self._assert_not_almost_equal(1.1, 1)
        self._assert_not_almost_equal(1.1, True)
        self._assert_not_almost_equal(1, 2)

    def test_assert_almost_equal_dicts(self):
        assert_almost_equal({'a': 1, 'b': 2}, {'a': 1, 'b': 2})
        self._assert_not_almost_equal({'a': 1, 'b': 2}, {'a': 1, 'b': 3})
        self._assert_not_almost_equal(
            {'a': 1, 'b': 2}, {'a': 1, 'b': 2, 'c': 3}
        )

    def test_assert_almost_equal_strings(self):
        assert_almost_equal('abc', 'abc')
        self._assert_not_almost_equal('abc', 'abcd')
        self._assert_not_almost_equal('abc', 'abd')

    def test_assert_almost_equal_iterables(self):
        assert_almost_equal([1, 2, 3], [1, 2, 3])
        self._assert_not_almost_equal([1, 2, 3], [1, 2, 4])
        self._assert_not_almost_equal([1, 2, 3], [1, 2, 3, 4])
        self._assert_not_almost_equal([1, 2, 3], 1)

    def test_assert_almost_equal_null(self):
        assert_almost_equal(None, None)
        assert_almost_equal(None, np.NaN)
        self._assert_not_almost_equal(None, 0)
        self._assert_not_almost_equal(0, None)
        self._assert_not_almost_equal(np.NaN, 0)


class TestUtilTesting(unittest.TestCase):
    _multiprocess_can_split_ = True

    def test_raise_with_traceback(self):
        with assertRaisesRegexp(LookupError, "error_text"):
            try:
                raise ValueError("THIS IS AN ERROR")
            except ValueError as e:
                e = LookupError("error_text")
                raise_with_traceback(e)
        with assertRaisesRegexp(LookupError, "error_text"):
            try:
                raise ValueError("This is another error")
            except ValueError:
                e = LookupError("error_text")
                _, _, traceback = sys.exc_info()
                raise_with_traceback(e, traceback)
