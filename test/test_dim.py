import unittest


class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)


if __name__ == '__main__':
    unittest.main()
# from sympy.physics.units import J, eV, Dimension, m,mm, kg, mol,N
# a = m
# b = Dim.convert_to_Dim(a)
# c = Dim.inverse_convert(b[1],b[0],target_units=[mm])
# if __name__=="__main__":
#
#
# ta = time.time()
#
# x = Dim([1, 2, 3, 4, 5, 6, 7])

# b = Dim(np.array([1, 1, 1, 1, 1, 1, 1]))

# c = b + 8
# d = 8 + b
# e = b + a
# f = a + b
# g = b + x
#
# g1 = a + dless
# g2 = dless+a
# g3 = dless + x
# g4 = x+dless
# g5 = dless + 1
# g6 = 1 + dless

# c = b - 8
# d = 8 - b
# e = b - a
# f = a - b
# g = b - x
#
# c = b * 8
# d = 8 * b
# e = b * a
# f = a * b
# g = b * x
#
# c = b / 8
# d = 8 / b
# e = b / a
# f = a / b
# g = b / x
#
# c = b ** 8
# d = 8 ** b
# e = b ** a
# f = a ** b
# g = b ** x

# h = abs(b)
# j = abs(a)
#
# h = -b
# j = -a

# xx = np.copy(a)
# print(a == dnan)
# print(a is a)
# print(a != b)
# print(a == 1)
# #
# k = dim_func()["exp"](a)
# l = dim_func()["exp"](b)
# m = dim_func()["exp"](dless)
#
#
# tb = time.time()
# print(tb - ta)

# print(SI._collect_factor_and_dimension(a))
# print(SI.get_quantity_scale_factor(a))
# print(SI.get_default_unit_system())
# print(SI.get_dimension_system())
# print(SI.get_quantity_dimension(a))

# bb=b[1]
# print(bb.dim)
# print(bb.unit)
# print(bb.inverse_convert(scale=b[0], target_units=None, unit_system="SI"))