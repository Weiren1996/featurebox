# -*- coding: utf-8 -*-

# @Time    : 2019/11/12 23:36
# @Email   : 986798607@qq.com
# @Software: PyCharm
# @License: BSD 3-Clause
import array
import copy
import warnings

class_replacers = {}

try:
    import numpy

    _ = (numpy.ndarray, numpy.array)
except ImportError:
    # Numpy is not present, skip the definition of the replacement class.
    pass
except AttributeError:
    # Numpy is present, but there is either no ndarray or array in numpy,
    # also skip the definition of the replacement class.
    pass
else:
    class _numpy_array(numpy.ndarray):
        def __deepcopy__(self, memo):
            """Overrides the deepcopy from numpy.ndarray that does not copy
            the object's attributes. This one will deepcopy the array and its
            :attr:`__dict__` attribute.
            """
            copy_ = numpy.ndarray.copy(self)
            copy_.__dict__.update(copy.deepcopy(self.__dict__, memo))
            return copy_

        @staticmethod
        def __new__(cls, iterable):
            """Creates a new instance of a numpy.ndarray from a function call.
            Adds the possibility to instanciate from an iterable."""
            return numpy.array(list(iterable)).view(cls)

        def __setstate__(self, state):
            self.__dict__.update(state)

        def __reduce__(self):
            return (self.__class__, (list(self),), self.__dict__)


    class_replacers[numpy.ndarray] = _numpy_array


class _array(array.array):
    @staticmethod
    def __new__(cls, seq=()):
        return super(_array, cls).__new__(cls, cls.typecode, seq)

    def __deepcopy__(self, memo):
        """Overrides the deepcopy from array.array that does not copy
        the object's attributes and class type.
        """
        cls = self.__class__
        copy_ = cls.__new__(cls, self)
        memo[id(self)] = copy_
        copy_.__dict__.update(copy.deepcopy(self.__dict__, memo))
        return copy_

    def __reduce__(self):
        return (self.__class__, (list(self),), self.__dict__)


class_replacers[array.array] = _array


def create(name, base, **kargs):
    """Creates a new class named *name* inheriting from *base* in the
    :mod:`~deap.creator` module. The new class can have attributes defined by
    the subsequent keyword arguments passed to the function create. If the
    argument is a class (without the parenthesis), the __init__ function is
    called in the initialization of an instance of the new object and the
    returned instance is added as an attribute of the class' instance.
    Otherwise, if the argument is not a class, (for example an :class:`int`),
    it is added as a "static" attribute of the class.

    :param name: The name of the class to create.
    :param base: A base class from which to inherit.
    :param attribute: One or more attributes to add on instantiation of this
                      class, optional.

    The following is used to create a class :class:`Foo` inheriting from the
    standard :class:`list` and having an attribute :attr:`bar` being an empty
    dictionary and a static attribute :attr:`spam` initialized to 1. ::

        create("Foo", list, bar=dict, spam=1)

    This above line is exactly the same as defining in the :mod:`creator`
    module something like the following. ::

        class Foo(list):
            spam = 1

            def __init__(spath):
                spath.bar = dict()

    The :ref:`creating-types` tutorial gives more examples of the creator
    usage.

    .. warning::

       If your are inheriting from :class:`numpy.ndarray` see the
       :doc:`tutorials/advanced/numpy` tutorial and the
       :doc:`/examples/ga_onemax_numpy` example.

    """

    if name in globals():
        warnings.warn("A class named '{0}' has already been created and it "
                      "will be overwritten. Consider deleting previous "
                      "creation of that class or rename it.".format(name),
                      RuntimeWarning)

    dict_inst = {}
    dict_cls = {}
    for obj_name, obj in kargs.items():
        if isinstance(obj, type):
            dict_inst[obj_name] = obj
        else:
            dict_cls[obj_name] = obj

    # Check if the base class has to be replaced
    if base in class_replacers:
        base = class_replacers[base]

    # A DeprecationWarning is raised when the object inherits from the
    # class "object" which leave the option of passing arguments, but
    # raise a warning stating that it will eventually stop permitting
    # this option. Usually this happens when the base class does not
    # override the __init__ method from object.
    def initType(self, *args, **kargs):
        """Replace the __init__ function of the new type, in order to
        add attributes that were defined with **kargs to the instance.
        """
        for obj_name, obj in dict_inst.items():
            setattr(self, obj_name, obj())
        if base.__init__ is not object.__init__:
            base.__init__(self, *args, **kargs)

    objtype = type(str(name), (base,), dict_cls)
    objtype.__init__ = initType
    # globals()[name] = objtype
    return objtype
