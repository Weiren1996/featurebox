import unittest

import numpy as np
from featurebox.symbol.dim import Dim, dless, dnan, dim_map


class MyTestCase(unittest.TestCase):
    def setUp(self):
        self.a = Dim([1, 2, 3, 4, 5, 6, 7])
        self.b = Dim([2, 2, 3, 4, 5, 6, 7])

        self.dl = dless
        self.dn = dnan

        self.c = Dim([[1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 6, 7]])

        self.dim_map = dim_map()

    def test_add(self):
        a, b, c, dl, dn = self.a, self.b, self.c, self.dl, self.dn

        assert a != b
        assert a != dl
        assert a != dn
        assert a == c
        self.assertEqual(a, a + 1)
        self.assertEqual(a, 1 + a)
        self.assertEqual(a, a + a)

        # assert (a + b).anyisnan()
        self.assertEqual(a, a + dl)
        self.assertEqual(a, dl + a)
        assert (a + dn).anyisnan()
        assert (dn + a).anyisnan()

        self.assertEqual(2, (1 + c).ndim)
        self.assertEqual(2, (c + 1).ndim)
        self.assertEqual(1, (1 + a).ndim)
        self.assertEqual(1, (a + 1).ndim)
        self.assertEqual(2, (a + c).ndim)
        self.assertEqual(1, (a + b).ndim)
        self.assertEqual(2, (c + dl).ndim)
        self.assertEqual(2, (c + b).ndim)
        self.assertEqual(2, (c + dn).ndim)
        self.assertEqual(2, (dn + c).ndim)
        self.assertEqual(1, (dn + dn).ndim)

    def test_mul(self):
        a, b, c, dl, dn = self.a, self.b, self.c, self.dl, self.dn
        self.assertEqual(a, a * 1)
        self.assertEqual(a, 1 * a)
        self.assertEqual(np.array(a) * 2, a * a)

        self.assertEqual(np.array([3, 4, 6, 8, 10, 12, 14]), a * b)
        self.assertNotEqual(np.array([2, 4, 6, 8, 10, 12, 14]), a * b)
        self.assertEqual(a, a * dl)
        self.assertEqual(a, dl * a)
        assert ((a * dn).anyisnan())
        assert ((dn * a).anyisnan())

        self.assertEqual(2, (1 * c).ndim)
        self.assertEqual(2, (c * 1).ndim)
        self.assertEqual(1, (1 * a).ndim)
        self.assertEqual(1, (a * 1).ndim)
        self.assertEqual(2, (a * c).ndim)
        self.assertEqual(1, (a * b).ndim)
        self.assertEqual(2, (c * dl).ndim)
        self.assertEqual(2, (c * b).ndim)
        self.assertEqual(2, (c * dn).ndim)
        self.assertEqual(2, (dn * c).ndim)
        self.assertEqual(1, (dn * dn).ndim)
        self.assertEqual(1, (dn * dl).ndim)
        self.assertEqual(1, (b * dl).ndim)

    #
    def test_div(self):
        a, b, c, dl, dn = self.a, self.b, self.c, self.dl, self.dn
        self.assertEqual(a, a / 1)
        self.assertEqual(-np.array(a), 1 / a)
        self.assertEqual(dless, a / a)
        self.a = Dim([1, 2, 3, 4, 5, 6, 7])
        self.b = Dim([2, 2, 3, 4, 5, 6, 7])
        self.assertEqual(np.array([-1, 0, 0, 0, 0, 0, 0]), a / b)
        self.assertNotEqual(np.array([2, 4, 6, 8, 10, 12, 14]), a / b)
        self.assertEqual(a, a / dl)
        self.assertEqual(-np.array(a), dl / a)
        assert ((a / dn).anyisnan())
        assert ((dn / a).anyisnan())

        self.assertEqual(2, (1 / c).ndim)
        self.assertEqual(2, (c / 1).ndim)
        self.assertEqual(1, (1 / a).ndim)
        self.assertEqual(1, (a / 1).ndim)
        self.assertEqual(2, (a / c).ndim)
        self.assertEqual(1, (a / b).ndim)
        self.assertEqual(2, (c / dl).ndim)
        self.assertEqual(2, (c / b).ndim)
        self.assertEqual(2, (c / dn).ndim)
        self.assertEqual(2, (dn / c).ndim)
        self.assertEqual(1, (dn / dn).ndim)
        self.assertEqual(1, (dn / dl).ndim)
        self.assertEqual(1, (b / dl).ndim)

    #
    def test_pow(self):
        a, b, c, dl, dn = self.a, self.b, self.c, self.dl, self.dn
        self.assertEqual(np.array(a) * 2, a ** 2)
        self.assertEqual(np.array(a) / 2, a ** 0.5)
        assert ((4 ** a).anyisnan())
        assert ((a ** b).anyisnan())

        self.assertEqual(1, (1 ** c).ndim)
        self.assertEqual(2, (c ** 1).ndim)
        self.assertEqual(1, (1 ** a).ndim)
        self.assertEqual(1, (a ** 1).ndim)
        self.assertEqual(1, (a ** c).ndim)
        self.assertEqual(1, (a ** b).ndim)
        self.assertEqual(1, (c ** dl).ndim)
        self.assertEqual(1, (c ** b).ndim)
        self.assertEqual(1, (c ** dn).ndim)
        self.assertEqual(1, (dn ** c).ndim)
        self.assertEqual(1, (dn ** dn).ndim)
        self.assertEqual(1, (dn ** dl).ndim)
        self.assertEqual(1, (b ** dl).ndim)

    def test_abs(self):
        # my_funcs = {"Abs": my_abs, "exp": my_exp, "log": my_log, 'cos': my_cos, 'sin': my_sin,
        #             'sqrt': my_sqrt, "Flat": my_flat, "Comp": my_comp, "Diff": my_diff,
        #             "Quot": my_quot, "Self": my_self}
        a, b, c, dl, dn = self.a, self.b, self.c, self.dl, self.dn
        my_funcs = self.dim_map
        func = my_funcs["Abs"]
        self.assertEqual(a, func(a))
        self.assertEqual(b, func(b))
        self.assertEqual(dl, func(4))

        self.assertEqual(2, func(c).ndim)
        self.assertEqual(2, func(a + c).ndim)
        self.assertEqual(2, func(c + b).ndim)
        self.assertEqual(1, func(b).ndim)

    #
    def test_exp(self):
        # my_funcs = {"Abs": my_abs, "exp": my_exp, "log": my_log, 'cos': my_cos, 'sin': my_sin,
        #             'sqrt': my_sqrt, "Flat": my_flat, "Comp": my_comp, "Diff": my_diff,
        #             "Quot": my_quot, "Self": my_self}
        a, b, c, dl, dn = self.a, self.b, self.c, self.dl, self.dn
        my_funcs = self.dim_map
        func = my_funcs["exp"]
        assert (func(a).anyisnan())
        assert (func(b).anyisnan())
        assert (func(dn).anyisnan())
        self.assertEqual(dl, func(4))
        self.assertEqual(dl, func(dl))

        self.assertEqual(2, func(c).ndim)
        self.assertEqual(2, func(a + c).ndim)
        self.assertEqual(2, func(c + b).ndim)
        self.assertEqual(1, func(dl).ndim)
        self.assertEqual(1, func(a).ndim)

    def test_Flat(self):
        # my_funcs = {"Abs": my_abs, "exp": my_exp, "log": my_log, 'cos': my_cos, 'sin': my_sin,
        #             'sqrt': my_sqrt, "Flat": my_flat, "Comp": my_comp, "Diff": my_diff,
        #             "Quot": my_quot, "Self": my_self}
        a, b, c, dl, dn = self.a, self.b, self.c, self.dl, self.dn
        my_funcs = self.dim_map
        func = my_funcs["MAdd"]

        self.assertEqual(a, func(a))
        self.assertEqual(b, func(b))
        self.assertEqual(dl, func(dl))
        assert (func(dn).anyisnan())
        self.assertEqual(c, func(c))
        self.assertEqual(dl, func(3))

        self.assertEqual(1, func(a).ndim)
        self.assertEqual(1, func(b).ndim)
        self.assertEqual(1, func(dl).ndim)
        self.assertEqual(1, func(dn).ndim)
        self.assertEqual(1, func(c).ndim)

    def test_Diff(self):
        # my_funcs = {"Abs": my_abs, "exp": my_exp, "log": my_log, 'cos': my_cos, 'sin': my_sin,
        #             'sqrt': my_sqrt, "Flat": my_flat, "Comp": my_comp, "Diff": my_diff,
        #             "Quot": my_quot, "Self": my_self}
        a, b, c, dl, dn = self.a, self.b, self.c, self.dl, self.dn
        c5 = a.copy()
        c5 = Dim(np.array([c5] * 5))

        my_funcs = self.dim_map
        func = my_funcs["MSub"]

        self.assertEqual(a, func(a))
        self.assertEqual(b, func(b))
        self.assertEqual(dl, func(dl))
        assert (func(dn).anyisnan())
        self.assertEqual(c, func(c))
        self.assertEqual(dl, func(3))

        self.assertEqual(1, func(a).ndim)
        self.assertEqual(1, func(b).ndim)
        self.assertEqual(1, func(dl).ndim)
        self.assertEqual(1, func(dn).ndim)
        self.assertEqual(1, func(c).ndim)
        self.assertEqual(2, func(c5).ndim)

    def test_comp(self):
        # my_funcs = {"Abs": my_abs, "exp": my_exp, "log": my_log, 'cos': my_cos, 'sin': my_sin,
        #             'sqrt': my_sqrt, "Flat": my_flat, "Comp": my_comp, "Diff": my_diff,
        #             "Quot": my_quot, "Self": my_self}
        a, b, c, dl, dn = self.a, self.b, self.c, self.dl, self.dn
        c5 = a.copy()
        c5 = Dim(np.array([c5] * 5))

        my_funcs = self.dim_map
        func = my_funcs["MMul"]

        self.assertEqual(a, func(a))
        self.assertEqual(b, func(b))
        self.assertEqual(dl, func(dl))
        assert (func(dn).anyisnan())
        self.assertEqual(2 * np.array(c), func(c))
        self.assertEqual(5 * np.array(c), func(c5))
        self.assertEqual(dl, func(3))

        self.assertEqual(1, func(a).ndim)
        self.assertEqual(1, func(b).ndim)
        self.assertEqual(1, func(dl).ndim)
        self.assertEqual(1, func(dn).ndim)
        self.assertEqual(1, func(c).ndim)
        self.assertEqual(1, func(c5).ndim)

    #
    def test_quot(self):
        # my_funcs = {"Abs": my_abs, "exp": my_exp, "log": my_log, 'cos': my_cos, 'sin': my_sin,
        #             'sqrt': my_sqrt, "Flat": my_flat, "Comp": my_comp, "Diff": my_diff,
        #             "Quot": my_quot, "Self": my_self}
        a, b, c, dl, dn = self.a, self.b, self.c, self.dl, self.dn
        c5 = a.copy()
        c5 = Dim(np.array([c5] * 5))

        my_funcs = self.dim_map
        func = my_funcs["MDiv"]

        self.assertEqual(a, func(a))
        self.assertEqual(b, func(b))
        self.assertEqual(dl, func(dl))
        assert (func(dn).anyisnan())
        self.assertEqual(dless, func(c))
        self.assertEqual(c5, func(c5))
        self.assertEqual(dl, func(3))

        self.assertEqual(1, func(a).ndim)
        self.assertEqual(1, func(b).ndim)
        self.assertEqual(1, func(dl).ndim)
        self.assertEqual(1, func(dn).ndim)
        self.assertEqual(1, func(c).ndim)
        self.assertEqual(2, func(c5).ndim)


if __name__ == '__main__':
    unittest.main()
