#!/usr/bin/env python3

"""test_minnetonka.py: test the minnetonka language for value modeling"""

__author__ = "Dave Bridgeland"
__copyright__ = "Copyright 2017-2018, Hanging Steel Productions LLC"
__credits__ = ["Dave Bridgeland"]

__version__ = "1"
__maintainer__ = "Dave Bridgeland"
__email__ = "dave@hangingsteel.com"
__status__ = "Prototype"

import unittest 
import unittest.mock
import io
import sys
import random
import collections
import numpy as np 

from minnetonka import *

class ModelCreationTest(unittest.TestCase):
    """Create a model, in a couple of ways. Access the treatments"""
    def test_create_with_one_treatment(self):
        """Create a model with a single treatment"""
        m = model(treatments=['As is'])
        ts = list(m.treatments())
        self.assertEqual(len(ts), 1)
        self.assertEqual(ts[0].name, 'As is')

    def test_create_with_two_treatments(self):
        """Create a model with a single treatment"""
        m = model(treatments=['As is', 'To be'])
        ts = list(m.treatments())
        self.assertEqual(len(ts), 2)
        self.assertEqual(ts[0].name, 'As is')
        self.assertEqual(ts[1].name, 'To be')

    def test_create_model_with_no_explicit_treatments(self):
        """Create a model with no explicit treatments"""
        m = model()
        ts = list(m.treatments())
        self.assertEqual(len(ts), 1)
        self.assertEqual(ts[0].name, '')

    def test_create_model_with_descriptions(self):
        """Create a model with treatment descriptions"""
        m = model(treatments=[('As is', 'The current situation'),
                                   ('To be', 'The future')])
        ts = list(m.treatments())
        self.assertEqual(len(ts), 2)
        self.assertEqual(ts[0].name, 'As is')
        self.assertEqual(ts[0].description, 'The current situation')
        self.assertEqual(ts[1].name, 'To be')
        self.assertEqual(ts[1].description, 'The future')

    def test_four_mixed_treatments(self):
        """Create a model with four treatments, some of which are described"""
        m = model(treatments=[('As is', 'The current situation'), 
                                   'To be', 
                                   'Alternative 1',
                                   ('Alternative 2', 'Another possibility')])
        ts = list(m.treatments())
        self.assertEqual(len(ts), 4)
        self.assertEqual(ts[0].name, 'As is')
        self.assertEqual(ts[0].description, 'The current situation')
        self.assertEqual(ts[1].name, 'To be')
        self.assertIsNone(ts[1].description)
        self.assertEqual(ts[2].name, 'Alternative 1')
        self.assertIsNone(ts[2].description)
        self.assertEqual(ts[3].name, 'Alternative 2')
        self.assertEqual(ts[3].description, 'Another possibility')


class ModelTreatmentAccess(unittest.TestCase):
    """Access the treatments from a model"""
    def test_access_treatments(self):
        """Access the treatments from a model"""
        m = model(treatments=[('As is', 'The current situation'), 
                               ('To be', 'The future')])
        self.assertEqual(m.treatment('As is').name, 'As is')
        self.assertEqual(
            m.treatment('As is').description, 'The current situation')
        self.assertEqual(m.treatment('To be').description, 'The future')


class ModelVariableAccess(unittest.TestCase):
    """Access the variable (classes) of a model"""
    def test_variable_access(self):
        """Access a variable with .variable()"""
        DischargeBegins = variable('DischargeBegins', 12)
        DischargeEnds = variable('DischargeEnds', 18)
        m = model([DischargeBegins, DischargeEnds])
        self.assertEqual(m.variable('DischargeBegins'), DischargeBegins)
        self.assertEqual(m.variable('DischargeEnds'), DischargeEnds)

    def test_variable_access(self):
        """Access a variable that does not exist"""
        DischargeBegins = variable('DischargeBegins', 12)
        DischargeEnds = variable('DischargeEnds', 18)
        m = model([DischargeBegins, DischargeEnds])
        with self.assertRaises(MinnetonkaError) as me:
            m.variable('DischargeAbides')
        self.assertEqual(
            me.exception.message, 'Unknown variable DischargeAbides')

    def test_subscripts(self):
        DischargeBegins = variable('DischargeBegins', 12)
        DischargeEnds = variable('DischargeEnds', 18)
        m = model([DischargeBegins, DischargeEnds])
        self.assertEqual(m['DischargeBegins'], DischargeBegins)
        self.assertEqual(m['DischargeEnds'], DischargeEnds)
        with self.assertRaises(MinnetonkaError) as me:
            m['DischargeAbides']
        self.assertEqual(
            me.exception.message, 'Unknown variable DischargeAbides')

    def test_redefined_variable(self):
        DischargeBegins = variable('DischargeBegins', 12)
        DB2 = variable('DischargeBegins', 13)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            m = model([DischargeBegins, DB2])
            self.assertEqual(len(w), 1)
            self.assertEqual(w[-1].category, MinnetonkaWarning)
            self.assertEqual(
                str(w[-1].message), 'Variable DischargeBegins redefined')
        self.assertEqual(m['DischargeBegins'][''], 13)


class TreatmentTest(unittest.TestCase):
    """Basic test of treatments"""
    def test_repr(self): 
        """Are the treatments repred correctly?"""
        m = model(treatments=[
            'As is', ('Value at risk', 'Total value that could be achieved')])
        nullTreatment, valueAtRisk = m.treatments()
        self.assertEqual(repr(nullTreatment), "Treatment('As is')")
        self.assertEqual(
            repr(valueAtRisk),
            "Treatment('Value at risk', 'Total value that could be achieved')")

    def test_by_name(self):
        """Is the class Treatment keeping track of all the treatments?"""
        m = model(treatments=[
            'As is', ('Value at risk', 'Total value that could be achieved')])
        nullTreatment, valueAtRisk = m.treatments()
        self.assertEqual(m.treatment('As is'), nullTreatment)
        self.assertEqual(m.treatment('Value at risk'), valueAtRisk)

    def test_by_name_not_found(self):
        """Does Treatment raise an error if the treatment is not found?"""
        m = model(treatments=[
            'As is', ('Value at risk', 'Total value that could be achieved')])
        with self.assertRaises(MinnetonkaError) as me:
            foo = m.treatment('Could be')
        self.assertEqual(
            me.exception.message, 'Model has no treatment Could be')


class SimpleQuantityTest(unittest.TestCase):
    """Tests for simple quantities"""
    def test_single_simple_quantity(self):
        """Does the simple quantity know its value?"""
        DischargeBegins = variable('DischargeBegins', 12)
        m = model([DischargeBegins])
        self.assertEqual(DischargeBegins.by_treatment('').amount(), 12)

    def test_single_simple_quantity_via_subscript(self):
        """Does the simple quantity know its value?"""
        DischargeBegins = variable('DischargeBegins', 12)
        m = model([DischargeBegins])
        self.assertEqual(DischargeBegins[''], 12)

    def test_simple_equality_two_treatments(self):
        """Does a simple quantity know its values in 2 different treatments?"""
        DischargeBegins = variable('DischargeBegins', 
            PerTreatment({'As is': 12, 'To be': 2}))
        m = model([DischargeBegins], ['As is', 'To be'])
        self.assertEqual(m['DischargeBegins']['As is'], 12)
        self.assertEqual(m['DischargeBegins']['To be'], 2)

    def test_constant_with_default_across_treatments(self):
        DischargeEnds = variable('DischargeEnds', 15)
        DischargeBegins = variable('DischargeBegins', 
            PerTreatment({'As is': 12, 'To be': 2}))
        m = model([DischargeBegins, DischargeEnds], ['As is', 'To be'])
        self.assertEqual(DischargeEnds['As is'], 15)
        self.assertEqual(DischargeEnds['To be'], 15)
        self.assertEqual(m['DischargeBegins']['As is'], 12)
        self.assertEqual(m['DischargeBegins']['To be'], 2)

    def test_incomplete_varying_quantity(self):
        """Does an incomplete varying quantity know it's incomplete?"""
        DischargeBegins = variable('DischargeBegins', 
            PerTreatment({'As is': 12, '2B': 2}))
        with self.assertRaises(MinnetonkaError) as cm:
            m = model([DischargeBegins], ['As is', 'To be'])
        self.assertEqual(cm.exception.message,
                        "Treatment 'To be' not defined")

    def test_quantity_knows_treatment(self): 
        """Does the simple quantity know its treatment?"""
        DischargeBegins = variable('DischargeBegins', 
            PerTreatment({'As is': 12, 'To be': 2}))
        m = model([DischargeBegins], ['As is', 'To be'])
        self.assertEqual(m['DischargeBegins'].by_treatment('As is').treatment(),
                         m.treatment('As is'))
        self.assertEqual(m['DischargeBegins'].by_treatment('To be').treatment(),
                         m.treatment('To be'))

    def test_treatment_knows_quantity(self):
        """Does the treatment know its simple quantity?"""
        DischargeBegins = variable('DischargeBegins', 
            PerTreatment({'As is': 12, 'To be': 2}))
        m = model([DischargeBegins], ['As is', 'To be'])
        self.assertEqual(m['DischargeBegins'].by_treatment('As is'),
                         m.treatment('As is')['DischargeBegins'])
        self.assertEqual(m['DischargeBegins'].by_treatment('To be'),
                      m.treatment('To be')['DischargeBegins'])

    def test_reset(self):
        """Can a simple quantity reset correctly?"""
        DischargeBegins = variable('DischargeBegins', 
            PerTreatment({'As is': 12, 'To be': 2}))
        m = model([DischargeBegins], ['As is', 'To be'])
        DischargeBegins['As is'] = 11
        m.reset()
        self.assertEqual(DischargeBegins['As is'], 12)

    def test_docstring(self):
        """Can a simple quantity have a docstring?"""
        DischargeEnds = variable('DischargeEnds', 
            """The quarter when discharging ends""",
            15)
        self.assertEqual(
            DischargeEnds.__doc__, 'The quarter when discharging ends')


class ContextManagerTest(unittest.TestCase):
    """Create a model and variables using a contex manager"""
    def test_variable_access_within_context_manager(self):
        """Does a model defined as a context mgr know about the variables?"""
        with model() as m:
            DischargeBegins = variable('DischargeBegins', 12)
        self.assertEqual(m.variable('DischargeBegins'), DischargeBegins)

    def test_model_initialization_via_context_manager(self):
        """Does a model defined as a context manager initialize?"""
        with model():
            DischargeBegins = variable('DischargeBegins', 12)
            DischargeEnds = variable('DischargeEnds', 15)
        self.assertEqual(DischargeBegins[''], 12)
        self.assertEqual(DischargeEnds[''], 15)

    def test_variables_without_python_vars_within_context_manager(self):
        """Does a model context manager need vars to have python vars?"""
        with model() as m:
            variable('DischargeBegins', 12)
            variable('DischargeEnds', 15)
        self.assertEqual(m['DischargeBegins'][''], 12)
        self.assertEqual(m['DischargeEnds'][''], 15)

    def test_reopen_context_manager(self):
        """Can I reopen a previously defined context manager and add a var?"""
        with model() as m:
            variable('DischargeBegins', 12)
        with m:
            variable('DischargeEnds', 15)
        self.assertEqual(m['DischargeEnds'][''], 15)

    def test_reopen_context_manager_after_step(self):
        """Can I reopen a previously defined context manager and add a var?"""
        with model() as m:
            variable('DischargeBegins', 12)
        m.step()
        with m:
            variable('DischargeEnds', 15)
        self.assertEqual(m['DischargeEnds'][''], 15)
        self.assertEqual(m['DischargeBegins'][''], 12)

    def test_redefine_variable(self):
        """Can I redefine a variable in a subsequent context?"""
        with model() as m:
            variable('DischargeBegins', 12)
        self.assertEqual(m['DischargeBegins'][''], 12)
        # Yuck. Need to encapsulate all this code to check for warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            with m:
                variable('DischargeBegins', 13)
            self.assertEqual(len(w), 1)
            self.assertEqual(w[-1].category, MinnetonkaWarning)
            self.assertEqual(
                str(w[-1].message), 'Variable DischargeBegins redefined')
        self.assertEqual(m['DischargeBegins'][''], 13)


class NoArgVariableTest(unittest.TestCase):
    """Tests for variables without any arguments"""
    def test_simple_noarg(self):
        """Does a simple no arg variable work?"""
        with model():
            DischargeProgress = variable('DischargeProgress', lambda: 0.5)
        self.assertEqual(DischargeProgress[''], 0.5)

    def test_simple_no_arg_with_docstring(self):
        """Does a simple no arg variable wiht a docstring work?"""
        with model():
            DischargeProgress = variable('DischargeProgress',  
                """Between the beginning and the end, how much progress?""",
                lambda: 0.5)
        self.assertEqual(
            DischargeProgress.__doc__, 
            'Between the beginning and the end, how much progress?')
        self.assertEqual(DischargeProgress[''], 0.5)

    def test_changing_no_arg(self):
        """Can the no arg variable change its behavior?"""
        with model(treatments=['As is']) as m:
            progress = 0.5
            DischargeProgress = variable('DischargeProgress', lambda: progress)
        self.assertEqual(DischargeProgress['As is'], 0.5)
        progress = 0.7
        m.step()
        self.assertEqual(DischargeProgress['As is'], 0.7)

    def test_reset_no_arg(self):
        """Can a no arg variable reset?"""
        with model() as m:
            DischargeProgress = variable('DischargeProgress', lambda: 0.5)
        self.assertEqual(DischargeProgress[''], 0.5)
        m.reset()
        self.assertEqual(DischargeProgress[''], 0.5)

    def test_embedded_fn(self):
        """Can a function be defined within a model context?"""
        with model() as m:
            def _fn(x):
                return x + 1

            Foo = variable('Foo', _fn, 'Bar')
            variable('Bar', 9)

        self.assertEqual(Foo[''], 10)

    def test_different_treatments_different_callables(self):
        """Can different treatments be given different callables"""
        with model(treatments=['As is', 'To be']) as m:
            Foo = variable('Foo', PerTreatment(
                {'As is': lambda: 12, 'To be': lambda: 13}))

        self.assertEqual(Foo['As is'], 12)
        self.assertEqual(Foo['To be'], 13)

    def test_callable_and_constant(self):
        """Can one treatment have a callable and another a constant"""
        with model(treatments=['As is', 'To be']) as m:
            Foo = variable('Foo', PerTreatment(
                {'As is': lambda: 12, 'To be': 13}))

        self.assertEqual(Foo['As is'], 12)
        self.assertEqual(Foo['To be'], 13)


class OneArgVariableTest(unittest.TestCase):
    """Tests for variables that have a single argument"""
    def test_sunny_day(self):
        with model(treatments=['As is', 'To be']):
            DischargeBegins = variable('DischargeBegins', 
                PerTreatment({'As is': 4, 'To be': 3}))
            current_step = 7
            DischargeProgress = variable('DischargeProgress', 
                """Between the beginning and the end, how much progress?""",
                lambda db: (current_step - db) / 4,
                'DischargeBegins') 
        self.assertEqual(DischargeProgress['As is'], 0.75)
        self.assertEqual(DischargeProgress['To be'], 1.0)
        self.assertEqual(
            DischargeProgress.__doc__, 
            'Between the beginning and the end, how much progress?')

    def test_different_treatments_different_callables(self):
        """Can different treatments be given different callables"""
        with model(treatments=['As is', 'To be']) as m:
            DischargeBegins = variable('DischargeBegins', 4)
            current_step = 7
            DischargeProgress = variable('DischargeProgress', 
                """Between the beginning and the end, how much progress?""",
                PerTreatment(
                    {'As is': lambda db: (current_step - db) / 4,
                     'To be': lambda db: (current_step - db + 1) / 4}),
                'DischargeBegins')

        self.assertEqual(DischargeProgress['As is'], 0.75)
        self.assertEqual(DischargeProgress['To be'], 1)

    def test_populate_no_depends(self):
        current_step = 7
        with self.assertRaises(MinnetonkaError):
            with model(treatments=['As is', 'To be']):
                variable('Progress', lambda db: (current_step - db) / 4)

    def test_simple_circularity(self):
        """Test for detecting a variable that depends on itself"""
        with self.assertRaises(MinnetonkaError) as me:
            with model():
                variable('Reflect', lambda r: r+2, 'Reflect')
        self.assertEqual(me.exception.message,
            'Circularity among variables: Reflect <- Reflect')
    

class TwoArgsTest(unittest.TestCase):
    """Tests for variables that have two arguments"""
    def test_two_args_sunny_day(self):
        with model():
            variable('DischargeBegins', 5)
            variable('DischargeEnds', 9)
            current_step = 7
            DischargeProgress = variable('DischargeProgress',
                """Between the beginning and the end, how much progress?""",
                lambda db, de: (current_step - db) / (de - db),
                'DischargeBegins', 'DischargeEnds') 
        self.assertEqual(DischargeProgress[''], 0.5)
        self.assertEqual(
            DischargeProgress.__doc__, 
            'Between the beginning and the end, how much progress?')

    def test_two_args_2_treatments_sunny_day(self):
        with model(treatments=['As is', 'Just might work']):
            variable('DischargeBegins', 5)
            variable('DischargeEnds', 
                PerTreatment({'As is': 13, 'Just might work': 11}))
            current_step = 7
            DischargeProgress = variable('DischargeProgress',
                lambda db, de: (current_step - db) / (de - db),
                'DischargeBegins', 'DischargeEnds') 
        self.assertEqual(DischargeProgress['As is'], 0.25)
        self.assertEqual(DischargeProgress['Just might work'], 1/3)

    def test_two_arg_circularity(self):
        """Can variable detect dependency on var that depends on the first?"""
        with self.assertRaises(MinnetonkaError) as me:
            with model():
                variable('Foo', lambda b: b + 2, 'Bar')
                variable('Bar', lambda f: f - 2, 'Foo')
        self.assertEqual(me.exception.message,
            'Circularity among variables: Foo <- Bar <- Foo')

    def test_three_arg_circularity(self):
        """Can variable detect dependency on var that depends on the first?"""
        with self.assertRaises(MinnetonkaError) as me:
            with model():
                variable('Foo', lambda b: b + 2, 'Bar')
                variable('Bar', lambda b: b - 2, 'Baz')
                variable('Baz', lambda f: f + 10, 'Foo')
        self.assertEqual(me.exception.message,
            'Circularity among variables: Foo <- Bar <- Baz <- Foo')


class ExpressionCacheTest(unittest.TestCase):
    """Test the behavior of the cache"""
    def test_cache_retention(self):
        """Test cache retention"""
        with model():
            hidden = 12
            variable('Cached', lambda: hidden)
            UsesCached = variable('UsesCached', lambda x: x, 'Cached')
        self.assertEqual(UsesCached[''], 12)
        hidden = 14
        self.assertEqual(UsesCached[''], 12)


class TimeTest(unittest.TestCase):
    """Test step(), on all kinds of simple variables"""
    def test_step_constant(self):
        """Test step on a constant variable"""
        with model() as m:
            HoursPerDay = variable('HoursPerDay', 24)
        self.assertEqual(HoursPerDay[''], 24)
        m.step()
        self.assertEqual(HoursPerDay[''], 24)

    def test_TIME(self):
        """Test usage of the TIME value in a lambda"""
        with model() as m:
            Tm = variable('Tm', lambda md: md.TIME, '__model__')
        self.assertEqual(Tm[''], 0)
        m.step()
        self.assertEqual(Tm[''], 1)
        m.reset()
        self.assertEqual(Tm[''], 0)

    def test_STEP(self):
        """Test usage of the STEP value in a lambda"""
        with model() as m:
            St = variable('St', lambda md: md.STEP, '__model__')
        self.assertEqual(St[''], 0)
        m.step()
        self.assertEqual(St[''], 1)
        m.reset()
        self.assertEqual(St[''], 0)

    def test_TIME_smaller_timestep(self):
        """Test usage of the TIME value when timestep is not 1"""
        with model(timestep=0.5) as m:
            Time = variable('Time', lambda md: md.TIME, '__model__')
            Step = variable('Step', lambda md: md.STEP, '__model__')
        self.assertEqual(Time[''], 0)
        self.assertEqual(Step[''], 0)
        m.step()
        self.assertEqual(Time[''], 0.5)
        self.assertEqual(Step[''], 1)
        m.step()
        self.assertEqual(Time[''], 1)
        self.assertEqual(Step[''], 2)
        m.reset()
        self.assertEqual(Time[''], 0)
        self.assertEqual(Step[''], 0)

    def test_TIME_n(self):
        """Test usage of the step(n)"""
        with model() as m:
            Time = variable('Time', lambda md: md.TIME, '__model__')
            Step = variable('Step', lambda md: md.STEP, '__model__')
        self.assertEqual(Time[''], 0)
        self.assertEqual(Step[''], 0)
        m.step(5)
        self.assertEqual(Time[''], 5)
        self.assertEqual(Step[''], 5)
        m.step(3)
        self.assertEqual(Time[''], 8)
        self.assertEqual(Step[''], 8)
        m.reset()
        self.assertEqual(Time[''], 0)
        self.assertEqual(Step[''], 0)
        m.step(4)
        self.assertEqual(Time[''], 4)
        self.assertEqual(Step[''], 4)

    def test_TIME_n_smaller(self):
        """Test usage of the step(n) with a non-unitary timestep"""
        with model(timestep=0.25) as m:
            Time = variable('Time', lambda md: md.TIME, '__model__')
        self.assertEqual(Time[''], 0)
        m.step(5)
        self.assertEqual(Time[''], 1.25)
        m.step(3)
        self.assertEqual(Time[''], 2)
        m.reset()
        self.assertEqual(Time[''], 0)
        m.step(4)
        self.assertEqual(Time[''], 1)

    def test_step_usage(self):
        """Test step usage in a more complex situation."""
        with model(treatments=['As is', 'To be']) as m:
            variable('DischargeBegins', 5)
            variable('DischargeEnds', 
                PerTreatment({'As is': 13, 'To be': 11}))
            DischargeProgress = variable(
                'DischargeProgress', 
                lambda db, de, md: max(0, min(1, (md.TIME - db) / (de - db))),
                'DischargeBegins', 'DischargeEnds', '__model__') 
                  
        self.assertEqual(DischargeProgress['As is'], 0)
        self.assertEqual(DischargeProgress['To be'], 0)
        m.step(6)
        self.assertEqual(DischargeProgress['As is'], 0.125)
        self.assertEqual(DischargeProgress['To be'], 1/6)
        m.step()
        self.assertEqual(DischargeProgress['As is'], 0.25)
        self.assertEqual(DischargeProgress['To be'], 1/3)
        m.step(4)
        self.assertEqual(DischargeProgress['As is'], 0.75)
        self.assertEqual(DischargeProgress['To be'], 1)
        m.step(2)
        self.assertEqual(DischargeProgress['As is'], 1)
        self.assertEqual(DischargeProgress['To be'], 1)

    def test_depends_on_step(self):
        """Test various kinds of variables depending on step."""
        with model() as m:
            Step = variable('Step', lambda md: md.STEP, '__model__')
            StockStep = stock('StockStep', 
                lambda s: s, ('Step',), 
                lambda s: s, ('Step',))
            AccumStep = accum('AccumStep', lambda s: s, ('Step',), 0)
            PreviousStep = previous('PreviousStep', 'Step', 0)

        self.assertEqual(StockStep[''], 0)
        self.assertEqual(AccumStep[''], 0)
        self.assertEqual(PreviousStep[''], 0)
        m.step()
        self.assertEqual(StockStep[''], 0)
        self.assertEqual(AccumStep[''], 1)
        self.assertEqual(PreviousStep[''], 0)
        m.step()
        self.assertEqual(StockStep[''], 1)
        self.assertEqual(AccumStep[''], 3)
        self.assertEqual(PreviousStep[''], 1)
        m.step()
        self.assertEqual(StockStep[''], 3)
        self.assertEqual(AccumStep[''], 6)
        self.assertEqual(PreviousStep[''], 2)


class StartAndEndTest(unittest.TestCase):
    """Test step(), on all kinds of simple variables"""

    def test_start_time_simple(self):
        """Test step usage with non-zero start."""
        with model(start_time=2019) as m:
            Time = variable('Time', lambda md: md.TIME, '__model__')
            Step = variable('Step', lambda md: md.STEP, '__model__')

        self.assertEqual(Time[''], 2019)
        self.assertEqual(Step[''], 0)
        m.step()
        self.assertEqual(Time[''], 2020)
        self.assertEqual(Step[''], 1)
        m.step()
        self.assertEqual(Time[''], 2021)
        self.assertEqual(Step[''], 2)
        m.reset()
        self.assertEqual(Time[''], 2019)
        self.assertEqual(Step[''], 0)

    def test_start_time_with_timestep(self):
        """Test step usage with non-zero start and timestep."""
        with model(start_time=2019, timestep=0.25) as m:
            Time = variable('Time', lambda md: md.TIME, '__model__')
            Step = variable('Step', lambda md: md.STEP, '__model__')

        self.assertEqual(Time[''], 2019)
        self.assertEqual(Step[''], 0)
        m.step()
        self.assertEqual(Time[''], 2019.25)
        self.assertEqual(Step[''], 1)
        m.step()
        self.assertEqual(Time[''], 2019.5)
        self.assertEqual(Step[''], 2)
        m.reset()
        self.assertEqual(Time[''], 2019)
        self.assertEqual(Step[''], 0)

    def test_end_time(self):
        """Test step usage with end time."""
        with model(end_time=5) as m:
            Time = variable('Time', lambda md: md.TIME, '__model__')
            Step = variable('Step', lambda md: md.STEP, '__model__')
            Foo = stock('Foo', 1, 0)

        self.assertEqual(Time[''], 0)
        self.assertEqual(Step[''], 0)
        self.assertEqual(Foo[''], 0)
        m.step(5)        
        self.assertEqual(Time[''], 5)
        self.assertEqual(Step[''], 5)
        self.assertEqual(Foo[''], 5)
        with self.assertRaises(MinnetonkaError) as err:
            m.step()
        self.assertEqual(err.exception.message,
                        "Attempted to simulation beyond end_time: 5")
        self.assertEqual(Time[''], 5)
        self.assertEqual(Step[''], 5)
        self.assertEqual(Foo[''], 5)

    def test_step_to_end(self):
        """Test simple case of stepping to end."""
        with model(end_time=5) as m:
            Time = variable('Time', lambda md: md.TIME, '__model__')
            Step = variable('Step', lambda md: md.STEP, '__model__')
            Foo = stock('Foo', 1, 0)

        m.step(to_end=True)
        self.assertEqual(Time[''], 5)
        self.assertEqual(Step[''], 5)
        self.assertEqual(Foo[''], 5)
        m.reset()
        m.step()
        self.assertEqual(Time[''], 1)
        self.assertEqual(Step[''], 1)
        self.assertEqual(Foo[''], 1)
        m.step(to_end=True)
        self.assertEqual(Time[''], 5)
        self.assertEqual(Step[''], 5)
        self.assertEqual(Foo[''], 5)

    def test_step_to_end_twice(self):
        """Test step to end redundantly."""
        with model(end_time=5) as m:
            Time = variable('Time', lambda md: md.TIME, '__model__')
            Step = variable('Step', lambda md: md.STEP, '__model__')
            Foo = stock('Foo', 1, 0)

        m.step(to_end=True)
        m.step(to_end=True)
        self.assertEqual(Time[''], 5)
        self.assertEqual(Step[''], 5)
        self.assertEqual(Foo[''], 5)
        m.reset()
        m.step()
        self.assertEqual(Time[''], 1)
        self.assertEqual(Step[''], 1)
        self.assertEqual(Foo[''], 1)
        m.step(to_end=True)
        m.step(to_end=True)
        self.assertEqual(Time[''], 5)
        self.assertEqual(Step[''], 5)
        self.assertEqual(Foo[''], 5)

    def test_step_to_end_with_timestep(self):
        """Test step to end with a non-one timestep."""
        with model(end_time=5, timestep=0.25) as m:
            Time = variable('Time', lambda md: md.TIME, '__model__')
            Step = variable('Step', lambda md: md.STEP, '__model__')
            Foo = stock('Foo', 1, 0)

        m.step(to_end=True)
        self.assertEqual(Time[''], 5)
        self.assertEqual(Step[''], 20)
        self.assertEqual(Foo[''], 5)
        m.reset()
        m.step()
        self.assertEqual(Time[''], 0.25)
        self.assertEqual(Step[''], 1)
        self.assertEqual(Foo[''], 0.25)
        m.step(to_end=True)
        self.assertEqual(Time[''], 5)
        self.assertEqual(Step[''], 20)
        self.assertEqual(Foo[''], 5)

    def test_step_to_end_with_incompatible_timestep(self):
        """Test step to end with incompatible timestep."""
        with model(end_time=4.6, timestep=0.5) as m:
            Time = variable('Time', lambda md: md.TIME, '__model__')
            Step = variable('Step', lambda md: md.STEP, '__model__')
            Foo = stock('Foo', 1, 0)

        m.step(to_end=True)
        self.assertEqual(Time[''], 4.5)
        self.assertEqual(Step[''], 9)
        self.assertEqual(Foo[''], 4.5)
        m.reset()
        m.step()
        self.assertEqual(Time[''], 0.5)
        self.assertEqual(Step[''], 1)
        self.assertEqual(Foo[''], 0.5)
        m.step(to_end=True)
        self.assertEqual(Time[''], 4.5)
        self.assertEqual(Step[''], 9)
        self.assertEqual(Foo[''], 4.5)

    def test_start_and_end(self):
        """Test a model that has both a start time and an end time."""
        with model(start_time=2018, end_time=2022) as m:
            Time = variable('Time', lambda md: md.TIME, '__model__')
            Step = variable('Step', lambda md: md.STEP, '__model__')
            Foo = stock('Foo', 1, 0)

        self.assertEqual(Time[''], 2018)
        self.assertEqual(Foo[''], 0)
        m.step()
        self.assertEqual(Time[''], 2019)
        self.assertEqual(Foo[''], 1)
        m.step(to_end=True)
        self.assertEqual(Time[''], 2022)
        self.assertEqual(Foo[''], 4)
        m.reset()
        self.assertEqual(Time[''], 2018)
        self.assertEqual(Foo[''], 0)

    def test_incompatible_start_and_end(self):
        """Test a model that has an incompatible start_time and end_time."""
        with self.assertRaises(MinnetonkaError) as err:
            with model(start_time=2018, end_time=2017) as m:
                Time = variable('Time', lambda md: md.TIME, '__model__')
                Step = variable('Step', lambda md: md.STEP, '__model__')
                Foo = stock('Foo', 1, 0)
        self.assertEqual(err.exception.message,
                         'End time 2017 is before start time 2018')

    def test_STARTTIME_and_ENDTIME(self):
        """Test access of start and end variables."""
        with model(start_time=2019, end_time=2022) as m:
            Start = variable('Start', lambda md: md.STARTTIME, '__model__')
            End = variable('End', lambda md: md.ENDTIME, '__model__')

        self.assertEqual(Start[''], 2019)
        self.assertEqual(End[''], 2022)
        m.step()
        self.assertEqual(Start[''], 2019)
        self.assertEqual(End[''], 2022)
        m.reset()
        self.assertEqual(Start[''], 2019)
        self.assertEqual(End[''], 2022)


class ConstantTest(unittest.TestCase):
    """Test constants, that are initiallized and then don't change"""
    def test_simple_constant(self):
        """Does a simple constant have the right value?"""
        with model() as m:
            DischargeBegins = constant('DischargeBegins', 12)
        self.assertEqual(m['DischargeBegins'][''], 12)
        m.step()
        self.assertEqual(m['DischargeBegins'][''], 12)
        m.step(4)
        self.assertEqual(m['DischargeBegins'][''], 12)
        m.reset()
        self.assertEqual(m['DischargeBegins'][''], 12)

    def test_lambda_constant(self):
        """Is a constant only evaluated once?"""
        how_many = 0
        def eval_once():
            nonlocal how_many
            assert(how_many == 0)
            how_many += 1
            return 12

        with model() as m:
            DischargeBegins = constant('DischargeBegins', eval_once)

        self.assertEqual(m['DischargeBegins'][''], 12)
        m.step()
        self.assertEqual(m['DischargeBegins'][''], 12)
        m.step(4)
        self.assertEqual(m['DischargeBegins'][''], 12) 
        how_many = 0
        m.reset()
        self.assertEqual(m['DischargeBegins'][''], 12)
        m.step()
        self.assertEqual(m['DischargeBegins'][''], 12)


    def test_lambda_constant_multiple_args(self):
        """Is a constant with multiple arguments only evaluated once?"""
        how_many = 0
        def eval_once(a, b):
            nonlocal how_many
            assert(how_many == 0)
            how_many += 1
            return a + b

        with model() as m:
            variable('Foo', 9)
            variable('Bar', 3)
            DischargeBegins = constant(
                'DischargeBegins', eval_once, 'Foo', 'Bar')

        self.assertEqual(m['DischargeBegins'][''], 12)
        m.step()
        self.assertEqual(m['DischargeBegins'][''], 12)
        m.step(4)
        self.assertEqual(m['DischargeBegins'][''], 12) 
        how_many = 0
        m.reset()
        self.assertEqual(m['DischargeBegins'][''], 12)
        m.step()
        self.assertEqual(m['DischargeBegins'][''], 12)


    def test_constant_and_treatments(self):
        """Can a constant take different values in different treatments?"""
        with model(treatments=['Bar', 'Baz']) as m:
            DischargeBegins = constant('DischargeBegins', 
                                        PerTreatment({'Bar': 9, 'Baz':10}))

        self.assertEqual(m['DischargeBegins']['Bar'], 9)
        self.assertEqual(m['DischargeBegins']['Baz'], 10)
        m.step()
        self.assertEqual(m['DischargeBegins']['Bar'], 9)
        self.assertEqual(m['DischargeBegins']['Baz'], 10)
        
        
class BasicStockTest(unittest.TestCase):
    """Test stocks"""

    def test_simple_stock_zero_initial(self):
        """Stock with no callables and no initial"""
        with model() as m:
            S = stock('S', 5)
        self.assertEqual(S[''], 0)
        m.step()
        self.assertEqual(S[''], 5)
        m.step()
        self.assertEqual(S[''], 10)
        m.reset()
        self.assertEqual(S[''], 0)
        m.step(3)
        self.assertEqual(S[''], 15)
        self.assertEqual(S.__doc__, '')

    def test_simple_stock_zero_initial_half_step(self):
        """Stock with no callables, no initial, and timestep = 0.5"""
        with model(timestep=0.5) as m:
            S = stock('S', 5)
        self.assertEqual(S[''], 0)
        m.step()
        self.assertEqual(S[''], 2.5)
        m.step()
        self.assertEqual(S[''], 5)
        m.reset()
        self.assertEqual(S[''], 0)
        m.step(3)
        self.assertEqual(S[''], 7.5)

    def test_simple_stock_zero_initial_and_docstring(self):
        """Stock with no callables and no initial, but with docstring"""
        with model() as m:
            S = stock('S', """Increase by 5 every step""", 5)
        self.assertEqual(S[''], 0)
        m.step()
        self.assertEqual(S[''], 5)
        self.assertEqual(S.__doc__, 'Increase by 5 every step')

    def test_simple_stock_with_initial(self):
        """Stock with no callables but with an initial"""
        with model() as m:
            S = stock('S', 1, 22)
        self.assertEqual(S[''], 22)
        m.step()
        self.assertEqual(S[''], 23)
        m.step()
        self.assertEqual(S[''], 24)
        m.reset()
        self.assertEqual(S[''], 22)
        m.step(3)
        self.assertEqual(S[''], 25)

    def test_simple_stock_with_initial_and_docstring(self):
        """Stock with no callables but with an initial"""
        with model() as m:
            S = stock('S', """Start at 22 and increase by 1""", 1, 22)
        self.assertEqual(S[''], 22)
        m.step()
        self.assertEqual(S[''], 23)
        self.assertEqual(S.__doc__, 'Start at 22 and increase by 1')

    def test_simple_stock_with_varying_initial(self):
        """Stock with no callables but with a treatment-varying initial"""
        with model(treatments=['As is', 'To be']) as m:
            S = stock('S', 1, PerTreatment({'As is': 22, 'To be': 23}))
        self.assertEqual(S['As is'], 22)
        self.assertEqual(S['To be'], 23)
        m.step()
        self.assertEqual(S['As is'], 23)
        self.assertEqual(S['To be'], 24)
        m.step()
        self.assertEqual(S['As is'], 24)
        self.assertEqual(S['To be'], 25)
        m.reset()
        self.assertEqual(S['As is'], 22)
        self.assertEqual(S['To be'], 23)
        m.step(3)
        self.assertEqual(S['As is'], 25)
        self.assertEqual(S['To be'], 26)

    def test_stock_with_callable_flow(self):
        """Stock with callable flow, but depends on nothing"""
        with model() as m:
            S = stock('S', lambda: 1, (), 22)
        self.assertEqual(S[''], 22)
        m.step()
        self.assertEqual(S[''], 23)
        m.step()
        self.assertEqual(S[''], 24)
        m.reset()
        self.assertEqual(S[''], 22)
        m.step(3)
        self.assertEqual(S[''], 25)

    def test_stock_with_callable_flow_and_init(self):
        """Stock with callable flow and callable init, depends on nothing"""
        with model() as m:
            S = stock('S', 
                """Start at 22 and increase by 1""",
                lambda: 1, (), lambda: 22, ())
        self.assertEqual(S[''], 22)
        m.step()
        self.assertEqual(S[''], 23)
        m.step()
        self.assertEqual(S[''], 24)
        m.reset()
        self.assertEqual(S[''], 22)
        m.step(3)
        self.assertEqual(S[''], 25)
        self.assertEqual(S.__doc__, 'Start at 22 and increase by 1')

    def test_stock_with_simple_increment_variable(self):
        """Stock with very simple variable dependency"""
        with model() as m:
            variable('X', 1)
            S = stock('S', lambda x: x, ('X',), 22)
        self.assertEqual(S[''], 22)
        m.step()
        self.assertEqual(S[''], 23)
        m.step()
        self.assertEqual(S[''], 24)
        m.reset()
        self.assertEqual(S[''], 22)
        m.step(3)
        self.assertEqual(S[''], 25)

    def test_stock_with_nontuple_dependency(self):
        """Test stock with a nontuple dependency, translated to tuple.""" 
        with model() as m:
            variable('XY', 1)
            S = stock('S', lambda x: x, 'XY', 22)

        with model() as m:
            variable('XY', 1)
            S = stock('S', lambda: 1, (), lambda x: x, 'XY')


    def test_stock_with_two_callables_with_depends(self):
        """Stock with depends vars for both flow and initial"""
        with model() as m:
            variable('X', 1)
            variable('Y', 22)
            S = stock('S',
                """Start at 22 and increase by 1""",
                 lambda x: x, ('X',), lambda x: x, ('Y',))
        self.assertEqual(S[''], 22)
        m.step()
        self.assertEqual(S[''], 23)
        m.step()
        self.assertEqual(S[''], 24)
        m.reset()
        self.assertEqual(S[''], 22)
        m.step(3)
        self.assertEqual(S[''], 25)
        self.assertEqual(S.__doc__, 'Start at 22 and increase by 1')

    def test_stock_with_variable_increase(self):
        with model() as m:
            variable('Time', lambda md: md.TIME, '__model__')
            S = stock('S', lambda s: s, ('Time',), 0)
        m.step()
        self.assertEqual(S[''], 0)
        m.step()
        self.assertEqual(S[''], 1)
        m.step()
        self.assertEqual(S[''], 3)
        m.step()
        self.assertEqual(S[''], 6)

    def test_stock_with_positive_feedback(self):
        """Classic interest stock"""
        with model() as m:
            Savings = stock(
                'Savings', lambda interest: interest, ('Interest',), 1000)
            variable('Rate', 0.05)
            variable(
                'Interest', lambda savings, rate: savings * rate, 
                'Savings', 'Rate')
        self.assertEqual(Savings[''], 1000)
        m.step()
        self.assertEqual(Savings[''], 1050)
        m.step()
        self.assertEqual(Savings[''], 1102.5)
        m.step()
        self.assertEqual(Savings[''], 1157.625)
        m.reset()
        self.assertEqual(Savings[''], 1000)
        m.step()
        self.assertEqual(Savings[''], 1050)

    def test_stock_with_positive_feedback_small_timestep(self):
        """Classic interest stock with a smaller timestep"""
        with model(timestep=0.25) as m:
            Savings = stock('Savings', 
                lambda interest: interest, ('Interest',), 1000)
            variable('Rate', 0.05)
            variable('Interest', 
                lambda savings, rate: savings * rate,
                'Savings', 'Rate')
        self.assertEqual(Savings[''], 1000)
        m.step()
        self.assertEqual(Savings[''], 1012.5)
        m.step()
        self.assertAlmostEqual(Savings[''], 1025.156, places=3)
        m.step()
        self.assertAlmostEqual(Savings[''], 1037.971, places=3)
        m.step()
        self.assertAlmostEqual(Savings[''], 1050.945, places=3)
        m.step()
        self.assertAlmostEqual(Savings[''], 1064.082, places=3)

    def test_stock_with_positive_feedback_and_treatments(self):
        """Classic interest stock"""
        with model(treatments=['Good', 'Better', 'Best']) as m:
            Savings = stock('Savings', 
                lambda interest: interest, ('Interest',), 1000)
            variable('Rate', 
                PerTreatment({'Good': 0.04, 'Better': 0.05, 'Best': 0.06}))
            variable('Interest', 
                lambda savings, rate: savings * rate,
                'Savings', 'Rate')
        self.assertEqual(Savings['Good'], 1000)
        self.assertEqual(Savings['Better'], 1000)
        self.assertEqual(Savings['Best'], 1000)
        m.step()
        self.assertEqual(Savings['Good'], 1040)
        self.assertEqual(Savings['Better'], 1050)
        self.assertEqual(Savings['Best'], 1060)
        m.step()
        self.assertEqual(Savings['Good'], 1081.6)
        self.assertEqual(Savings['Better'], 1102.5)
        self.assertEqual(Savings['Best'], 1123.6)
        m.reset()
        self.assertEqual(Savings['Good'], 1000)
        self.assertEqual(Savings['Better'], 1000)
        self.assertEqual(Savings['Best'], 1000)
        m.step()
        self.assertEqual(Savings['Good'], 1040)
        self.assertEqual(Savings['Better'], 1050)
        self.assertEqual(Savings['Best'], 1060)

    def test_stock_with_many_depends(self):
        """Test stock that depends on a lot of callables"""
        with model() as m:
            ABCDEFGH = stock(
                'ABCDEFGH', 
                lambda a, b, c, d, e, f, g, h: a + b + c + d + e + f + g + h,
                ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'),
                0)
            variable('A', lambda md: md.TIME, '__model__')
            variable('B', lambda md: md.TIME, '__model__')
            variable('C', lambda md: md.TIME, '__model__')
            variable('D', lambda md: md.TIME, '__model__')
            variable('E', lambda md: md.TIME, '__model__')
            variable('F', lambda md: md.TIME, '__model__')
            variable('G', lambda md: md.TIME, '__model__')
            variable('H', lambda md: md.TIME, '__model__')
        self.assertEqual(ABCDEFGH[''], 0)
        m.step()
        self.assertEqual(ABCDEFGH[''], 0)
        m.step()
        self.assertEqual(ABCDEFGH[''], 8)
        m.step()
        self.assertEqual(ABCDEFGH[''], 24)
        m.step()
        self.assertEqual(ABCDEFGH[''], 48)

    def test_stock_order(self):
        """Test a stock that uses variables defined before and after"""
        with model() as m:
            variable('Before', lambda md: md.TIME, '__model__')
            stock('UsingTimes', 
                lambda before, after: before + after, ('Before', 'After'), 
                0)
            variable('After', lambda md: md.TIME, '__model__')

        self.assertEqual(m['UsingTimes'][''], 0)
        m.step()
        self.assertEqual(m['UsingTimes'][''], 0)
        m.step()
        self.assertEqual(m['UsingTimes'][''], 2)
        m.step()
        self.assertEqual(m['UsingTimes'][''], 6)
        m.step()
        self.assertEqual(m['UsingTimes'][''], 12)
        m.step(2)
        self.assertEqual(m['UsingTimes'][''], 30)

    def test_eval_count(self):
        """Test a stock that uses two variables that count # of calls"""
        before_count = 0
        after_count = 0

        def before():
            nonlocal before_count
            before_count += 1
            return before_count

        def after():
            nonlocal after_count
            after_count += 1
            return after_count

        with model() as m:
            variable('Before', before)
            stock('UsingBeforeAndAfter',
                lambda b, a: b + a, ('Before', 'After'),
                0)
            variable('After', after)

        self.assertEqual(m['UsingBeforeAndAfter'][''], 0)
        m.step()
        self.assertEqual(m['UsingBeforeAndAfter'][''], 2)
        m.step()
        self.assertEqual(m['UsingBeforeAndAfter'][''], 6)
        m.step()
        self.assertEqual(m['UsingBeforeAndAfter'][''], 12)
        m.step()
        self.assertEqual(m['UsingBeforeAndAfter'][''], 20)
        m.step(2)
        self.assertEqual(m['UsingBeforeAndAfter'][''], 42)
        m.reset()
        self.assertEqual(m['UsingBeforeAndAfter'][''], 0)
        m.step()
        self.assertEqual(m['UsingBeforeAndAfter'][''], 16)

    def test_variable_using_stock(self):
        """Test whether a variable can use an stock value"""
        with model() as m:
            stock('Revenue', 5, 0)
            variable('Cost', 10)
            variable('Earnings', lambda r, c: r - c, 'Revenue', 'Cost')

        self.assertEqual(m['Earnings'][''], -10)
        m.step()
        self.assertEqual(m['Earnings'][''], -5)
        m.step()
        self.assertEqual(m['Earnings'][''], 0)
        m.step()
        self.assertEqual(m['Earnings'][''], 5)
        m.reset()
        self.assertEqual(m['Earnings'][''], -10)

    def test_stock_using_stock(self):
        """Test a stock that uses another stock"""
        with model() as m:
            stock('First', 1)
            stock('Second', lambda f: f, ('First',), 0)
            stock('Third', lambda f, s: f + s, ('First', 'Second'), 0)

        m.step()
        self.assertEqual(m['First'][''], 1)
        self.assertEqual(m['Second'][''], 0)
        self.assertEqual(m['Third'][''], 0)
        m.step()
        self.assertEqual(m['First'][''], 2)
        self.assertEqual(m['Second'][''], 1)
        self.assertEqual(m['Third'][''], 1)
        m.step()
        self.assertEqual(m['First'][''], 3)
        self.assertEqual(m['Second'][''], 3)
        self.assertEqual(m['Third'][''], 4)
        m.step()
        self.assertEqual(m['First'][''], 4)
        self.assertEqual(m['Second'][''], 6)
        self.assertEqual(m['Third'][''], 10)
        m.step()
        self.assertEqual(m['First'][''], 5)
        self.assertEqual(m['Second'][''], 10)
        self.assertEqual(m['Third'][''], 20)

    def test_stock_using_stock_alt_ordering(self):
        """Test a stock using another stock, with user defined first"""
        with model() as m:
            stock('Third', lambda f, s: f + s, ('First', 'Second'), 0)
            stock('Second', lambda f: f, ('First',), 0)
            stock('First', 1)

        m.step()
        self.assertEqual(m['First'][''], 1)
        self.assertEqual(m['Second'][''], 0)
        self.assertEqual(m['Third'][''], 0)
        m.step()
        self.assertEqual(m['First'][''], 2)
        self.assertEqual(m['Second'][''], 1)
        self.assertEqual(m['Third'][''], 1)
        m.step()
        self.assertEqual(m['First'][''], 3)
        self.assertEqual(m['Second'][''], 3)
        self.assertEqual(m['Third'][''], 4)
        m.step()
        self.assertEqual(m['First'][''], 4)
        self.assertEqual(m['Second'][''], 6)
        self.assertEqual(m['Third'][''], 10)
        m.step()
        self.assertEqual(m['First'][''], 5)
        self.assertEqual(m['Second'][''], 10)
        self.assertEqual(m['Third'][''], 20)

    def test_stock_init_circularity(self):
        """Test a variable circularity involving stocks"""
        with self.assertRaises(MinnetonkaError) as me:
            with model() as m:
                stock('Foo', lambda b: b, ('Bar',), lambda b: b, ('Bar',))
                variable('Bar', lambda f: f, 'Foo')
        self.assertEqual(me.exception.message,
            'Circularity among variables: Foo <- Bar <- Foo')

    def test_stock_one_treatment_only(self):
        """Variable that uses a stock for 1 treatment and constant 4 another"""
        with model(treatments=['As is', 'To be']) as m:
            ValueAtRisk = variable('ValueAtRisk',
                PerTreatment({'As is': lambda x: x, 'To be': 0}),
                'ValueAtRiskAsIs')

            stock('ValueAtRiskAsIs', 1, 1)

        self.assertEqual(ValueAtRisk['To be'], 0)
        self.assertEqual(ValueAtRisk['As is'], 1)
        m.step()
        self.assertEqual(ValueAtRisk['To be'], 0)
        self.assertEqual(ValueAtRisk['As is'], 2)
        m.step(2)
        self.assertEqual(ValueAtRisk['To be'], 0)
        self.assertEqual(ValueAtRisk['As is'], 4)
        m.reset()
        self.assertEqual(ValueAtRisk['To be'], 0)
        self.assertEqual(ValueAtRisk['As is'], 1)

    def test_one_treatment_stock_both_sides(self):
        """A stock that has both init and incr defined with treatments""",
        with model(treatments=['As is', 'To be']) as m:
            Foo = stock('Foo',
                PerTreatment({'As is': lambda x: x, 'To be': 1}),
                ('Bar',),
                PerTreatment({'As is': 0, 'To be': lambda x: x + 1}),
                ('Baz',))
            variable('Bar', 2)
            variable('Baz', 1)

        self.assertEqual(Foo['To be'], 2)
        self.assertEqual(Foo['As is'], 0)
        m.step()
        self.assertEqual(Foo['To be'], 3)
        self.assertEqual(Foo['As is'], 2)
        m.step()
        self.assertEqual(Foo['To be'], 4)
        self.assertEqual(Foo['As is'], 4)


class BasicAccumTest(unittest.TestCase):
    """Test accums"""

    def test_simple_accum_zero_initial(self):
        """Accum with no callables and no initial"""
        with model() as m:
            A = accum('A', 5)

        self.assertEqual(A[''], 0)
        m.step()
        self.assertEqual(A[''], 5)        
        m.step()
        self.assertEqual(A[''], 10)
        m.reset()
        self.assertEqual(A[''], 0)
        m.step(3)
        self.assertEqual(A[''], 15)
        self.assertEqual(A.__doc__, '')

    def test_simple_accum_zero_initial_and_docstring(self):
        """Accum with no callables and no initial, but with a docstring"""
        with model() as m:
            A = accum('A', """Increase by 5 every step""", 5)

        self.assertEqual(A[''], 0)
        m.step()
        self.assertEqual(A[''], 5)
        self.assertEqual(A.__doc__, 'Increase by 5 every step')

    def test_simple_accum_with_initial(self):
        """Accum with no callables but with an initial"""
        with model() as m:
            A = accum('A', 1, 22)
    
        self.assertEqual(A[''], 22)
        m.step()
        self.assertEqual(A[''], 23)
        m.step()
        self.assertEqual(A[''], 24)
        m.reset()
        self.assertEqual(A[''], 22)
        m.step(3)
        self.assertEqual(A[''], 25)

    def test_simple_accum_with_initial_and_docstring(self):
        """Accum with no callables but with an initial"""
        with model() as m:
            A = accum('A', """Start at 22 and increase by 1""", 1, 22)

        self.assertEqual(A[''], 22)
        m.step()
        self.assertEqual(A[''], 23)
        self.assertEqual(A.__doc__, 'Start at 22 and increase by 1')

    def test_simple_accum_with_varying_initial(self):
        """Accum with no callables but with a treatment-varying initial"""
        with model(treatments=['As is', 'To be']) as m:
            A = accum('A', 1, PerTreatment({'As is': 22, 'To be': 23}))

        self.assertEqual(A['As is'], 22)
        self.assertEqual(A['To be'], 23)
        m.step() 
        self.assertEqual(A['As is'], 23)
        self.assertEqual(A['To be'], 24)
        m.reset() 
        m.step(3)
        self.assertEqual(A['As is'], 25)
        self.assertEqual(A['To be'], 26)

    def test_simple_accum_zero_initial_small_timestep(self):
        """Accum with no callables and no initial"""
        with model(timestep=0.25) as m:
            A = accum('A', 5)

        self.assertEqual(A[''], 0)
        m.step()
        self.assertEqual(A[''], 5)        
        m.step()
        self.assertEqual(A[''], 10)
        m.reset()
        self.assertEqual(A[''], 0)
        m.step(3)
        self.assertEqual(A[''], 15)

    def test_accum_with_callable_flow(self):
        """accum with callable flow, but depends on nothing"""
        with model() as m:
            A = accum('A', lambda: 1, (), 22)

        self.assertEqual(A[''], 22)
        m.step()
        self.assertEqual(A[''], 23)
        m.step()
        self.assertEqual(A[''], 24)
        m.reset()
        self.assertEqual(A[''], 22)
        m.step(3)
        self.assertEqual(A[''], 25)

    def test_accum_with_callable_flow_and_init(self):
        """accum with callable flow and callable init, but depends on nothing
        """
        with model() as m:
            A = accum('A', 
                """Start at 22 and increase by 1""",
                lambda: 1, (), lambda: 22, ())

        self.assertEqual(A[''], 22)
        m.step()
        self.assertEqual(A[''], 23)
        m.step()
        self.assertEqual(A[''], 24)
        m.reset()
        self.assertEqual(A[''], 22)
        m.step(3)
        self.assertEqual(A[''], 25)
        self.assertEqual(A.__doc__, 'Start at 22 and increase by 1')

    def test_accum_with_simple_increment_variable(self):
        """accum with very simple variable dependency"""
        with model() as m:
            variable('X', 1)
            A = accum('A', lambda x: x, ('X',), 22)

        self.assertEqual(A[''], 22)
        m.step()
        self.assertEqual(A[''], 23)
        m.step()
        self.assertEqual(A[''], 24)
        m.reset()
        self.assertEqual(A[''], 22)
        m.step(3)
        self.assertEqual(A[''], 25)

    def test_accum_with_two_callables_with_depends(self):
        """accum with depends vars for both flow and initial"""
        with model() as m:
            variable('X', 1)
            variable('Y', 22)
            A = accum('A',
                """Start at 22 and increase by 1""",
                 lambda x: x, ('X',), lambda x: x, ('Y',))

        self.assertEqual(A[''], 22)
        m.step()
        self.assertEqual(A[''], 23)
        m.step()
        self.assertEqual(A[''], 24)
        m.reset()
        self.assertEqual(A[''], 22)
        m.step(3)
        self.assertEqual(A[''], 25)
        self.assertEqual(A.__doc__, 'Start at 22 and increase by 1')

    def test_accum_with_nontuple_dependency(self):
        """Test accum with a nontuple dependency, translated to tuple.""" 
        with model() as m:
            variable('X1', 1)
            variable('Y2', 22)
            A = accum('A',
                """Start at 22 and increase by 1""",
                 lambda x: x, 'X1', lambda x: x, 'Y2') 

    def test_accum_with_variable_increase(self):
        with model() as m:
            variable('Time', lambda md: md.TIME, '__model__')
            A = accum('A', lambda s: s, ('Time',), 0)

        self.assertEqual(A[''], 0)
        m.step()
        self.assertEqual(A[''], 1)
        m.step()
        self.assertEqual(A[''], 3)
        m.step()
        self.assertEqual(A[''], 6)

    def test_accum_with_many_depends(self):
        """Test accum that depends on a lot of callables"""
        with model() as m:
            ABCDEFGH = accum(
                'ABCDEFGH', 
                lambda a, b, c, d, e, f, g, h: a + b + c + d + e + f + g + h,
                ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'),
                0)
            variable('A', lambda md: md.TIME, '__model__')
            variable('B', lambda md: md.TIME, '__model__')
            variable('C', lambda md: md.TIME, '__model__')
            variable('D', lambda md: md.TIME, '__model__')
            variable('E', lambda md: md.TIME, '__model__')
            variable('F', lambda md: md.TIME, '__model__')
            variable('G', lambda md: md.TIME, '__model__')
            variable('H', lambda md: md.TIME, '__model__')

        self.assertEqual(ABCDEFGH[''], 0)
        m.step()
        self.assertEqual(ABCDEFGH[''], 8)
        m.step()
        self.assertEqual(ABCDEFGH[''], 24)
        m.step()
        self.assertEqual(ABCDEFGH[''], 48)

    def test_accum_order(self):
        """Test a accum that uses variables defined before and after"""
        with model() as m:
            variable('Before', lambda md: md.TIME, '__model__')
            accum('UsingTimes', 
                lambda before, after: before + after, ('Before', 'After'), 
                0)
            variable('After', lambda md: md.TIME, '__model__')

        self.assertEqual(m['UsingTimes'][''], 0)
        m.step()
        self.assertEqual(m['UsingTimes'][''], 2)
        m.step()
        self.assertEqual(m['UsingTimes'][''], 6)
        m.step()
        self.assertEqual(m['UsingTimes'][''], 12)
        m.step(2)
        self.assertEqual(m['UsingTimes'][''], 30)

    def test_eval_count(self):
        """Test a accum that uses two variables that count # of calls"""
        before_count = 0
        after_count = 0

        def before():
            nonlocal before_count
            before_count += 1
            return before_count

        def after():
            nonlocal after_count
            after_count += 1
            return after_count

        with model() as m:
            variable('Before', before)
            accum('UsingBeforeAndAfter',
                lambda b, a: b + a, ('Before', 'After'),
                0)
            variable('After', after)

        self.assertEqual(m['UsingBeforeAndAfter'][''], 0)
        m.step()
        self.assertEqual(m['UsingBeforeAndAfter'][''], 4)
        m.step()
        self.assertEqual(m['UsingBeforeAndAfter'][''], 10)
        m.step()
        self.assertEqual(m['UsingBeforeAndAfter'][''], 18)
        m.step(2)
        self.assertEqual(m['UsingBeforeAndAfter'][''], 40)
        m.reset()
        self.assertEqual(m['UsingBeforeAndAfter'][''], 0)
        m.step()
        self.assertEqual(m['UsingBeforeAndAfter'][''], 16)

    def test_variable_using_accum(self):
        """Test whether a variable can use an accum value"""
        with model() as m:
            accum('Revenue', 5, 0)
            variable('Cost', 10)
            variable('Earnings', lambda r, c: r - c, 'Revenue', 'Cost')

        self.assertEqual(m['Earnings'][''], -10)
        self.assertEqual(m['Revenue'][''], 0)
        m.step()
        self.assertEqual(m['Earnings'][''], -5)
        self.assertEqual(m['Revenue'][''], 5)
        m.step()
        self.assertEqual(m['Earnings'][''], 0)
        m.step()
        self.assertEqual(m['Earnings'][''], 5)
        m.step()
        self.assertEqual(m['Earnings'][''], 10)
        m.reset()
        self.assertEqual(m['Earnings'][''], -10)

    def test_accum_using_accum(self):
        """Test an accume that uses a variable that is another accum"""
        with model() as m:
            accum('First', 1)
            accum('Second', lambda f: f, ('First',), 0)
            accum('Third', lambda f, s: f + s, ('First', 'Second'), 0)

        m.step()
        self.assertEqual(m['First'][''], 1)
        self.assertEqual(m['Second'][''], 1)
        self.assertEqual(m['Third'][''], 2)
        m.step()
        self.assertEqual(m['First'][''], 2)
        self.assertEqual(m['Second'][''], 3)
        self.assertEqual(m['Third'][''], 7)
        m.step()
        self.assertEqual(m['First'][''], 3)
        self.assertEqual(m['Second'][''], 6)
        self.assertEqual(m['Third'][''], 16)
        m.step()
        self.assertEqual(m['First'][''], 4)
        self.assertEqual(m['Second'][''], 10)
        self.assertEqual(m['Third'][''], 30)
        m.step()
        self.assertEqual(m['First'][''], 5)
        self.assertEqual(m['Second'][''], 15)
        self.assertEqual(m['Third'][''], 50)
        m.step()
        self.assertEqual(m['First'][''], 6)
        self.assertEqual(m['Second'][''], 21)
        self.assertEqual(m['Third'][''], 77)

    def test_accum_using_accum_alt_ordering(self):
        """Test accum that uses a previously defined accum"""
        with model() as m:
            accum('Third', lambda f, s: f + s, ('First', 'Second'), 0)
            accum('Second', lambda f: f, ('First',), 0)
            accum('First', 1)

        m.step()
        self.assertEqual(m['First'][''], 1)
        self.assertEqual(m['Second'][''], 1)
        self.assertEqual(m['Third'][''], 2)
        m.step()
        self.assertEqual(m['First'][''], 2)
        self.assertEqual(m['Second'][''], 3)
        self.assertEqual(m['Third'][''], 7)
        m.step()
        self.assertEqual(m['First'][''], 3)
        self.assertEqual(m['Second'][''], 6)
        self.assertEqual(m['Third'][''], 16)
        m.step()
        self.assertEqual(m['First'][''], 4)
        self.assertEqual(m['Second'][''], 10)
        self.assertEqual(m['Third'][''], 30)
        m.step()
        self.assertEqual(m['First'][''], 5)
        self.assertEqual(m['Second'][''], 15)
        self.assertEqual(m['Third'][''], 50)
        m.step()
        self.assertEqual(m['First'][''], 6)
        self.assertEqual(m['Second'][''], 21)
        self.assertEqual(m['Third'][''], 77)

    def test_accum_with_circularity(self):
        """Accum does not support the kind of circularity that stock does"""
        with self.assertRaises(MinnetonkaError) as cm:
            with model() as m:
                accum('Savings', 
                    lambda interest: interest, ('Interest',), 1000)
                variable('Rate', 0.05)
                variable('Interest', lambda savings, rate: savings * rate,
                         'Savings', 'Rate')

        self.assertEqual(cm.exception.message,
            'Circularity among variables: Savings <- Interest <- Savings')
    
    def test_accum_one_treatment_both_sides(self):
        """An accum that has both init and incr defined with treatments""",
        with model(treatments=['As is', 'To be']) as m:
            Foo = accum('Foo',
                PerTreatment({'As is': lambda x: x, 'To be': 1}),
                ('Bar',),
                PerTreatment({'As is': 0, 'To be': lambda x: x + 1}),
                ('Baz',))
            variable('Bar', 2)
            variable('Baz', 1)

        self.assertEqual(Foo['To be'], 2)
        self.assertEqual(Foo['As is'], 0)
        m.step()
        self.assertEqual(Foo['To be'], 3)
        self.assertEqual(Foo['As is'], 2)
        m.step()
        self.assertEqual(Foo['To be'], 4)
        self.assertEqual(Foo['As is'], 4)

class StandardSystemDynamicsTest(unittest.TestCase):
    """Test a few basic SD models"""

    def test_population(self):
        """Test basic population growth model"""
        with model() as m:
            Population = stock('Population', 
                lambda births: births, ('Births',),
                10000)
            Births = variable('Births', 
                lambda pop, rate: pop * rate, 
                'Population', 'BirthRate')
            variable('BirthRate', 0.1)

        self.assertEqual(Population[''], 10000)
        self.assertEqual(Births[''], 1000)
        m.step()
        self.assertEqual(Births[''], 1100)
        self.assertEqual(Population[''], 11000)
        m.step(2)
        self.assertEqual(Births[''], 1331)
        self.assertEqual(Population[''], 13310)
        m.reset()
        self.assertEqual(Population[''], 10000)
        self.assertEqual(Births[''], 1000)
        m.step()
        self.assertEqual(Births[''], 1100)
        self.assertEqual(Population[''], 11000)

    def test_mice(self):
        """Test standard birth and death model"""
        with model() as m:
            MicePopulation = stock('MicePopulation',
                lambda births, deaths: births - deaths, 
                ('MiceBirths', 'MiceDeaths'),
                10000)
            MiceBirths = variable('MiceBirths', 
                lambda pop, rate: pop * rate, 'MicePopulation', 'MiceBirthRate')
            variable('MiceBirthRate', 0.1)
            MiceDeaths = variable('MiceDeaths', 
                lambda pop, rate: pop * rate, 'MicePopulation', 'MiceDeathRate')
            variable('MiceDeathRate', 0.05)
   
        self.assertEqual(MicePopulation[''], 10000)
        self.assertEqual(MiceBirths[''], 1000)
        self.assertEqual(MiceDeaths[''], 500)
        m.step()
        self.assertEqual(MicePopulation[''], 10500)
        m.step()
        self.assertEqual(MicePopulation[''], 11025)

# a bit less verbose
def assert_array_equal(array1, array2):
    np.testing.assert_array_equal(array1, array2)

def assert_array_almost_equal(array1, array2):
    np.testing.assert_allclose(array1, array2)


class OneDimensionalArrayTest(unittest.TestCase):
    """Test one dimensional numpy arrays"""
    def test_array_access(self):
        """Test whether a variable can take an array value"""
        with model() as m:
            revenue = variable('Revenue', np.array([30.1, 15, 20]))

        self.assertEqual(revenue[''][0], 30.1)
        self.assertEqual(revenue[''][1], 15)
        self.assertEqual(revenue[''][2], 20)

    def test_expression(self):
        """Test whether an array supports simple expressions"""
        with model() as m:
            variable('Revenue', np.array([30.1, 15, 20]))
            variable('Cost', np.array([10, 10, 10]))
            variable('Earnings', lambda r, c: r - c, 'Revenue', 'Cost')

        assert_array_equal(m['Earnings'][''], np.array([20.1, 5, 10]))

    def test_mixed_array_and_scalar(self):
        """Test whether an array and a scalar can be combined w/o trouble"""
        with model() as m:
            variable('Revenue', np.array([30.1, 15, 20]))
            variable('Cost', 10)
            variable('Earnings', lambda r, c: r - c, 'Revenue', 'Cost')

        assert_array_equal(m['Earnings'][''], np.array([20.1, 5, 10]))

    def test_simple_stock(self):
        """Test whether an array supports simple stocks"""
        with model() as m:
            stock('Revenue', np.array([5, 5, 10]), np.array([0, 0, 0]))
            variable('Cost', np.array([10, 10, 10]))
            variable('Earnings', lambda r, c: r - c, 'Revenue', 'Cost')

        assert_array_equal(m['Revenue'][''], np.array([0, 0, 0]))
        assert_array_equal(m['Earnings'][''], np.array([-10, -10, -10]))
        m.step()
        assert_array_equal(m['Revenue'][''], np.array([5, 5, 10]))
        assert_array_equal(m['Earnings'][''], np.array([-5, -5, 0]))
        m.step()
        assert_array_equal(m['Revenue'][''], np.array([10, 10, 20]))
        assert_array_equal(m['Earnings'][''], np.array([0, 0, 10]))
        m.reset()
        assert_array_equal(m['Revenue'][''], np.array([0, 0, 0]))
        assert_array_equal(m['Earnings'][''], np.array([-10, -10, -10]))

    def test_simple_stock_small_timestep(self):
        """Test whether an array supports simple stocks"""
        with model(timestep=0.25) as m:
            stock('Revenue', np.array([5, 5, 10]), np.array([0, 0, 0]))
            variable('Cost', np.array([10, 10, 10]))
            variable('Earnings', lambda r, c: r - c, 'Revenue', 'Cost')

        assert_array_equal(m['Revenue'][''], np.array([0, 0, 0]))
        assert_array_equal(m['Earnings'][''], np.array([-10, -10, -10]))
        m.step()
        assert_array_equal(m['Revenue'][''], np.array([1.25, 1.25, 2.5]))
        assert_array_equal(m['Earnings'][''], np.array([-8.75, -8.75, -7.5]))
        m.step()
        assert_array_equal(m['Revenue'][''], np.array([2.5, 2.5, 5]))
        assert_array_equal(m['Earnings'][''], np.array([-7.5, -7.5, -5]))
        m.reset()
        assert_array_equal(m['Revenue'][''], np.array([0, 0, 0]))
        assert_array_equal(m['Earnings'][''], np.array([-10, -10, -10]))

    def test_simple_stock_with_treatments(self):
        """Test whether an array supports simple stocks, with treatments"""
        with model(treatments=['As is', 'To be']) as m: 
            stock('Revenue', np.array([5, 5, 10]), np.array([0, 0, 0]))
            variable('Cost', 
                PerTreatment({'As is': np.array([10, 10, 10]), 
                              'To be': np.array([9, 8, 6])}))
            variable('Earnings', lambda r, c: r - c, 'Revenue', 'Cost')

        assert_array_equal(m['Revenue']['As is'], np.array([0, 0, 0]))
        assert_array_equal(m['Earnings']['As is'], np.array([-10, -10, -10]))
        assert_array_equal(m['Earnings']['To be'], np.array([-9, -8, -6]))
        m.step()
        assert_array_equal(m['Earnings']['As is'], np.array([-5, -5, 0]))
        assert_array_equal(m['Earnings']['To be'], np.array([-4, -3, 4]))
        m.step()
        assert_array_equal(m['Earnings']['As is'], np.array([0, 0, 10]))
        assert_array_equal(m['Earnings']['To be'], np.array([1, 2, 14]))
        m.reset() 
        assert_array_equal(m['Earnings']['As is'], np.array([-10, -10, -10]))
        assert_array_equal(m['Earnings']['To be'], np.array([-9, -8, -6]))

    def test_simple_accum(self):
        """Test whether an array supports simple accums"""
        with model() as m:
            accum('Revenue', np.array([5, 5, 10]), np.array([0, 0, 0]))
            variable('Cost', np.array([10, 10, 10]))
            variable('Earnings', lambda r, c: r - c, 'Revenue', 'Cost')

        assert_array_equal(m['Revenue'][''], np.array([0, 0, 0]))
        assert_array_equal(m['Earnings'][''], np.array([-10, -10, -10]))
        m.step()
        assert_array_equal(m['Revenue'][''], np.array([5, 5, 10]))
        assert_array_equal(m['Earnings'][''], np.array([-5, -5, 0]))
        m.step()
        assert_array_equal(m['Revenue'][''], np.array([10, 10, 20]))
        assert_array_equal(m['Earnings'][''], np.array([0, 0, 10]))
        m.reset()
        assert_array_equal(m['Revenue'][''], np.array([0, 0, 0]))
        assert_array_equal(m['Earnings'][''], np.array([-10, -10, -10]))

    def test_array_sum(self):
        """Can I sum over an array?"""
        with model() as m:
            variable('Cost', np.array([10, 10, 5]))
            variable('TotalCost', np.sum, 'Cost')

        self.assertEqual(m['TotalCost'][''], 25)


class TwoDimensionalArrayTest(unittest.TestCase):
    """Test 2D numpy arrays"""
    def test_array_access(self):
        """Test whether a variable can take a 2D array value"""
        with model() as m:
            Revenue = variable('Revenue', np.array([[30.1, 15, 20], [1, 2, 0]]))

        assert_array_equal(Revenue[''], np.array([[30.1, 15, 20], [1, 2, 0]]))

    def test_mixed_array_and_scalar(self):
        """Test whether a lambda variable can take 2D arrays"""
        with model() as m:
            variable('Revenue', np.array([[30.1, 15, 20], [1, 2, 0]]))
            variable('Cost', 10)
            variable('Earnings', lambda r, c: r - c, 'Revenue', 'Cost')

        assert_array_equal(
            m['Earnings'][''], np.array([[20.1, 5, 10], [-9, -8, -10]]))

    def test_simple_stock(self):
        """Test whether a 2D array supports simple stocks"""
        with model() as m:
            stock('Revenue', np.array([[5, 5], [10, 15]]), np.zeros((2, 2)))
            variable('Cost', np.array([[10, 10], [0, 9]]))
            variable('Earnings', lambda r, c: r - c, 'Revenue', 'Cost')
            stock('AccumulatedEarnings', 
                  lambda r: r, ('Revenue',), 
                  np.zeros((2, 2)))

        revenue = m['Revenue']
        earnings = m['Earnings']

        assert_array_equal(revenue[''], np.array([[0, 0], [0, 0]]))
        assert_array_equal(earnings[''], np.array([[-10, -10], [0, -9]]))
        m.step()
        assert_array_equal(revenue[''], np.array([[5, 5], [10, 15]]))
        assert_array_equal(earnings[''], np.array([[-5, -5], [10, 6]]))
        m.step()
        assert_array_equal(revenue[''], np.array([[10, 10], [20, 30]]))
        assert_array_equal(earnings[''], np.array([[0, 0], [20, 21]]))
        m.reset()
        assert_array_equal(revenue[''], np.array([[0, 0], [0, 0]]))
        assert_array_equal(earnings[''], np.array([[-10, -10], [0, -9]]))

    def test_simple_stock_short_timestep(self):
        """Test whether a 2D array supports simple stocks"""
        with model(timestep=0.5) as m:
            stock('Revenue', np.array([[5, 5], [10, 15]]), np.zeros((2, 2)))
            variable('Cost', np.array([[10, 10], [0, 9]]))
            variable('Earnings', lambda r, c: r - c, 'Revenue', 'Cost')
            stock('AccumulatedEarnings', 
                lambda r: r, ('Revenue',), 
                np.zeros((2, 2)))

        revenue = m['Revenue']
        earnings = m['Earnings']

        assert_array_equal(revenue[''], np.array([[0, 0], [0, 0]]))
        assert_array_equal(earnings[''], np.array([[-10, -10], [0, -9]]))
        m.step(2)
        assert_array_equal(revenue[''], np.array([[5, 5], [10, 15]]))
        assert_array_equal(earnings[''], np.array([[-5, -5], [10, 6]]))
        m.step(2)
        assert_array_equal(revenue[''], np.array([[10, 10], [20, 30]]))
        assert_array_equal(earnings[''], np.array([[0, 0], [20, 21]]))
        m.reset()
        assert_array_equal(revenue[''], np.array([[0, 0], [0, 0]]))
        assert_array_equal(earnings[''], np.array([[-10, -10], [0, -9]]))

    def test_array_sum(self):
        """Test sum over 2D array"""
        with model() as m:
            variable('Revenue', np.array([[30.1, 15, 20], [1, 2, 0]]))
            variable('Cost', 10)
            variable('Earnings', lambda r, c: r - c, 'Revenue', 'Cost')
            variable('TotalEarnings', np.sum, 'Earnings')
            variable('TotalEarningsByCostCenter', 
                lambda e: np.sum(e, axis=0), 'Earnings')

        self.assertAlmostEqual(m['TotalEarnings'][''], 8.1)
        assert_array_almost_equal(
            m['TotalEarningsByCostCenter'][''], [11.1, -3, 0])


class FunctionTest(unittest.TestCase):
    """Test use of free-standing function"""
    def test_function_with_constant(self):
        """Test a variable initialized with a function call"""
        # Is this even a useful test??
        def is_special(facility, situation, criterion):
            return (facility == 1) and (situation == 0) and (criterion == 2)

        def create_attractiveness():
            attr = np.empty((3, 3, 3))
            for index in np.ndindex(*(attr.shape)):
                if is_special(*index):
                    attr[index] = index[0] * 100 + index[1] * 10 + index[2]
                else:
                    attr[index] = index[0] * 10 + index[1] 
            return attr

        with model() as m:
            variable('Attractiveness', create_attractiveness())

        self.assertEqual(m['Attractiveness'][''][0, 0, 0], 0)
        self.assertEqual(m['Attractiveness'][''][2, 2, 2], 22)
        self.assertEqual(m['Attractiveness'][''][2, 0, 1], 20)
        self.assertEqual(m['Attractiveness'][''][1, 0, 2], 102)

    def test_function_with_variable(self):
        """Test a variable updated with a function as a callable"""
        def is_special(facility, situation, criterion):
            return (facility == 1) and (situation == 0) and (criterion == 2)

        def attractiveness(md):
            attr = np.empty((3, 3, 3))
            for index in np.ndindex(*(attr.shape)):
                if is_special(*index):
                    attr[index] = md.TIME
                else:
                    attr[index] = index[0] * 10 + index[1] 
            return attr

        with model() as m:
            variable('Attractiveness', attractiveness, '__model__')

        self.assertEqual(m['Attractiveness'][''][0, 0, 0], 0)
        self.assertEqual(m['Attractiveness'][''][2, 2, 2], 22)
        self.assertEqual(m['Attractiveness'][''][2, 0, 1], 20)
        self.assertEqual(m['Attractiveness'][''][1, 0, 2], 0)
        m.step()
        self.assertEqual(m['Attractiveness'][''][0, 0, 0], 0)
        self.assertEqual(m['Attractiveness'][''][2, 2, 2], 22)
        self.assertEqual(m['Attractiveness'][''][2, 0, 1], 20)
        self.assertEqual(m['Attractiveness'][''][1, 0, 2], 1)
        m.step()
        self.assertEqual(m['Attractiveness'][''][0, 0, 0], 0)
        self.assertEqual(m['Attractiveness'][''][2, 2, 2], 22)
        self.assertEqual(m['Attractiveness'][''][2, 0, 1], 20)
        self.assertEqual(m['Attractiveness'][''][1, 0, 2], 2)
        m.step(10)
        self.assertEqual(m['Attractiveness'][''][0, 0, 0], 0)
        self.assertEqual(m['Attractiveness'][''][2, 2, 2], 22)
        self.assertEqual(m['Attractiveness'][''][2, 0, 1], 20)
        self.assertEqual(m['Attractiveness'][''][1, 0, 2], 12)
        m.reset()
        self.assertEqual(m['Attractiveness'][''][0, 0, 0], 0)
        self.assertEqual(m['Attractiveness'][''][2, 2, 2], 22)
        self.assertEqual(m['Attractiveness'][''][2, 0, 1], 20)
        self.assertEqual(m['Attractiveness'][''][1, 0, 2], 0)
        m.step()
        self.assertEqual(m['Attractiveness'][''][0, 0, 0], 0)
        self.assertEqual(m['Attractiveness'][''][2, 2, 2], 22)
        self.assertEqual(m['Attractiveness'][''][2, 0, 1], 20)
        self.assertEqual(m['Attractiveness'][''][1, 0, 2], 1)

    def test_function_with_stock(self):
        """Test a stock initialized and updated with functions"""
        def is_special(facility, situation, criterion):
            return (facility == 1) and (situation == 0) and (criterion == 2)

        def create_attractiveness():
            return np.zeros((3, 3, 3))

        def update_attractiveness():
            update = np.zeros((3, 3, 3))
            for index in np.ndindex(*(update.shape)):
                if is_special(*index):
                    update[index] = 1
            return update 

        with model() as m:
            stock('Attractiveness', 
                update_attractiveness, (), create_attractiveness, ())

        self.assertEqual(m['Attractiveness'][''][0, 0, 0], 0) 
        self.assertEqual(m['Attractiveness'][''][1, 0, 2], 0)
        m.step() 
        self.assertEqual(m['Attractiveness'][''][2, 2, 2], 0) 
        self.assertEqual(m['Attractiveness'][''][1, 0, 2], 1)
        m.step() 
        self.assertEqual(m['Attractiveness'][''][2, 0, 1], 0)
        self.assertEqual(m['Attractiveness'][''][1, 0, 2], 2)
        m.step(10)
        self.assertEqual(m['Attractiveness'][''][0, 0, 0], 0) 
        self.assertEqual(m['Attractiveness'][''][1, 0, 2], 12)
        m.reset()
        self.assertEqual(m['Attractiveness'][''][0, 0, 0], 0) 
        self.assertEqual(m['Attractiveness'][''][1, 0, 2], 0)
        m.step()
        self.assertEqual(m['Attractiveness'][''][0, 0, 0], 0) 
        self.assertEqual(m['Attractiveness'][''][1, 0, 2], 1)


class NamedTuplesFundasTest(unittest.TestCase):
    """Test functionality for named tuples"""
    def setUp(self):
        self.OneType = mn_namedtuple('OneType', ['Foo', 'Bar', 'Baz'])
        self.AnotherType = mn_namedtuple('AnotherType', ['Foo', 'Bar'])

    def test_add(self):
        """Test addition for Minnetonka named tuples"""
        self.assertEqual(self.OneType(1, 2, 3) + self.OneType(4, 5, 0), 
                         self.OneType(5, 7, 3))

    def test_add_scalar(self):
        """Test addition for MN named tuple and a scalar"""
        self.assertEqual(self.OneType(1, 2, 3) + 2, self.OneType(3, 4, 5))
        self.assertEqual(2 + self.OneType(1, 2, 0), self.OneType(3, 4, 2))

    def test_add_failure(self):
        """Test addition failure for Minnetonka named tuples"""
        with self.assertRaises(TypeError):
            self.OneType(1, 2, 3) + self.AnotherType(3, 4)

    def test_subtract(self):
        """Test subtraction for Minnetonka named tuples"""
        self.assertEqual(self.OneType(1, 2, 3) - self.OneType(4, 5, 0), 
                         self.OneType(-3, -3, 3))

    def test_subtract_scalar(self):
        """Test subtraction of scalar value from MN named tuple"""
        self.assertEqual(self.OneType(1, 2, 3) - 2, self.OneType(-1, 0, 1))
        self.assertEqual(5 - self.OneType(1, 2, 0), self.OneType(4, 3, 5))

    def test_subtract_failure (self):
        """Test subtraction failure for Minnetonka named tuples"""
        with self.assertRaises(TypeError):
            self.OneType(1, 2, 3) - self.AnotherType(3, 4)

    def test_multiply(self):
        """Test multiplication of two mn named tuples together"""
        self.assertEqual(self.OneType(1, 2, 3) * self.OneType(4, 5, 6),
                         self.OneType(4, 10, 18))

    def test_multiply_scalar(self):
        """Test multiplication of named tuple by scalar"""
        self.assertEqual(self.OneType(1, 2, 3) * 2, self.OneType(2, 4, 6))

    def test_multiply_failure(self):
        """Test multiplication failure for Minnetonka named tuples"""
        with self.assertRaises(TypeError):
            self.OneType(1, 2, 3) * self.AnotherType(5, 6)

    def test_divide(self):
        """Test divide of two mn named tuples together"""
        self.assertEqual(self.OneType(4, 5, 6) / self.OneType(1, 2, 3),
                         self.OneType(4, 2.5, 2))

    def test_divide_scalar(self):
        """Test divide of named tuple by scalar"""
        self.assertEqual(self.OneType(1, 2, 3) / 2, self.OneType(0.5, 1, 1.5))

    def test_divide_failure(self):
        """Test divide failure for Minnetonka named tuples"""
        with self.assertRaises(TypeError):
            self.OneType(1, 2, 3) / self.AnotherType(5, 6)

    def test_round(self):
        """Test rounding a mn named tuple."""
        self.assertEqual(
            round(self.OneType(3.2, 2, 14.65)), 
            self.OneType(3, 2, 15))
        self.assertEqual(
            round(self.OneType(3.2, 2, 14.65), 1), 
            self.OneType(3.2, 2, 14.7))

    def test_le(self):
        """Test <= on two named tuples."""
        self.assertTrue(self.OneType(1.4, 2, 14.65) <= self.OneType(1.4, 4, 15))
        self.assertFalse(self.OneType(1.4, 2, 14.65) <= self.OneType(1, 4, 15))

    def test_le_scalar(self):
        """Test <= on two named tuples."""
        self.assertTrue(self.OneType(1.4, 2, 14.65) <= 20)
        self.assertFalse(self.OneType(1.4, 2, 14.65) <= 14.0)
        self.assertTrue(1.4 <= self.OneType(1.4, 2, 14.65))
        self.assertFalse(2.0 <= self.OneType(1.4, 2, 14.65))

    def test_le_failure(self):
        """Test <= on two incomparable named tuples."""
        with self.assertRaises(TypeError):
            self.OneType(1, 2, 3) <= self.AnotherType(5, 6),

    def test_equal(self):
        """Test whether two equivalent namedtuples are judged equal"""
        self.assertEqual(self.OneType(0, 10, -10), self.OneType(0, 10, -10))

    def test_not_equal(self):
        """Test whether two uneqal namedtuples are judged unequal"""
        self.assertNotEqual(self.OneType(0, 10, -10), self.OneType(0, 10, -9))

    def test_create(self):
        """Test whether the new method _create works"""
        self.assertEqual(self.OneType._create(1), self.OneType(1, 1, 1))
        self.assertEqual(self.AnotherType._create(0), self.AnotherType(0, 0))


class UseOfNamedTupleTest(unittest.TestCase):
    """Test Minnetonka functionality for named tuples, instead of scalars"""
    def setUp(self):
        self.Payer = mn_namedtuple(
            'Payer', ['Medicare', 'Medicaid', 'Commercial'])

    def test_constant(self):
        """Test whether a constant can be a named tuple"""
        with model() as m:
            Revenue = variable('Revenue', self.Payer(30, 15, 20))

        self.assertEqual(Revenue[''].Medicare, 30)
        self.assertEqual(Revenue[''].Medicaid, 15)
        self.assertEqual(Revenue[''].Commercial, 20)

    def test_expression(self):
        """Test whether a variable with a callable can be a named tuple"""
        with model() as m:
            variable('Revenue', self.Payer(30, 15, 20))
            variable('Cost', self.Payer(10, 10, 10))
            Earnings = variable('Earnings', 
                lambda r, c: r - c, 'Revenue', 'Cost')

        self.assertEqual(Earnings[''], self.Payer(20, 5, 10))

    def test_sum(self):
        """Does a sum over a named tuple work?"""
        with model() as m:
            variable('Revenue', self.Payer(30, 15, 20))
            TotalRevenue = variable('TotalRevenue', sum, 'Revenue')
        self.assertEqual(TotalRevenue[''], 65)


    def test_simple_stock(self):
        """Test whether a simple stock can be a named tuple"""
        with model() as m:
            Revenue = stock('Revenue', 
                self.Payer(5, 5, 10), self.Payer(0, 0, 0))
            variable('Cost', self.Payer(10, 10, 10))
            Earnings = variable('Earnings', 
                lambda r, c: r - c, 'Revenue', 'Cost')

        self.assertEqual(Revenue[''], self.Payer(0, 0, 0))
        self.assertEqual(Earnings[''], self.Payer(-10, -10, -10))
        m.step()
        self.assertEqual(Revenue[''], self.Payer(5, 5, 10))
        self.assertEqual(Earnings[''], self.Payer(-5, -5, 0))
        m.step()
        self.assertEqual(Revenue[''], self.Payer(10, 10, 20))
        self.assertEqual(Earnings[''], self.Payer(0, 0, 10))
        m.reset()
        self.assertEqual(Revenue[''], self.Payer(0, 0, 0))
        self.assertEqual(Earnings[''], self.Payer(-10, -10, -10))

    def test_simple_stock_short_timestep(self):
        """Test whether a simple stock can be a named tuple; non-1 timestep"""
        with model(timestep=0.5) as m:
            Revenue = stock('Revenue', 
                self.Payer(5, 5, 10), self.Payer(0, 0, 0))
            variable('Cost', self.Payer(10, 10, 10))
            Earnings = variable('Earnings', 
                lambda r, c: r - c, 'Revenue', 'Cost')

        self.assertEqual(Revenue[''], self.Payer(0, 0, 0))
        self.assertEqual(Earnings[''], self.Payer(-10, -10, -10))
        m.step(2)
        self.assertEqual(Revenue[''], self.Payer(5, 5, 10))
        self.assertEqual(Earnings[''], self.Payer(-5, -5, 0))
        m.step(2)
        self.assertEqual(Revenue[''], self.Payer(10, 10, 20))
        self.assertEqual(Earnings[''], self.Payer(0, 0, 10))
        m.reset()
        self.assertEqual(Revenue[''], self.Payer(0, 0, 0))
        self.assertEqual(Earnings[''], self.Payer(-10, -10, -10))

    def test_stock_with_callables(self):
        """Test whether a stock with callables can use named tuples"""
        with model() as m:
            stock('Revenue', self.Payer(5, 5, 10), self.Payer(0, 0, 0))
            variable('Cost', self.Payer(10, 10, 10))
            variable('Earnings', lambda r, c: r - c, 'Revenue', 'Cost')
            AccumulatedEarnings = stock('AccumulatedEarnings',
                lambda e: e, ('Earnings',),
                self.Payer(0, 0, 0))

        self.assertEqual(AccumulatedEarnings[''], self.Payer(0, 0, 0))
        m.step()
        self.assertEqual(AccumulatedEarnings[''], self.Payer(-10, -10, -10))
        m.step()
        self.assertEqual(AccumulatedEarnings[''], self.Payer(-15, -15, -10))
        m.step()
        self.assertEqual(AccumulatedEarnings[''], self.Payer(-15, -15, 0))
        m.reset()
        self.assertEqual(AccumulatedEarnings[''], self.Payer(0, 0, 0))

    def test_namedtuple_per_treatment(self):
        """Test whether a treatment can accept a namedtuple"""
        with model(treatments=['As is', 'To be']) as m:
            BaseRevenue = variable('BaseRevenue',
                PerTreatment({'As is': self.Payer(12, 13, 14),
                              'To be': self.Payer(2, 4, 6)}))
            TotalRevenue = variable('TotalRevenue',
                lambda br: br + 2,
                'BaseRevenue')

        self.assertEqual(BaseRevenue['As is'], self.Payer(12, 13, 14))
        self.assertEqual(BaseRevenue['To be'], self.Payer(2, 4, 6))
        self.assertEqual(TotalRevenue['As is'], self.Payer(14, 15, 16))
        self.assertEqual(TotalRevenue['To be'], self.Payer(4, 6, 8))

class TwoSimulations(unittest.TestCase):
    """Very simple situations of having two simulations"""
    def test_creating_two_constants_with_same_name(self):
        """Test creating two constants within two different simulations"""
        work_model = model([variable('HoursPerDay', 8)])
        calendar_model = model([variable('HoursPerDay', 24)])
        self.assertEqual(work_model['HoursPerDay'][''], 8)
        self.assertEqual(calendar_model['HoursPerDay'][''], 24)

    def test_two_variables_with_same_name(self):
        """Test whether two variables with the same name will udpate amouns"""
        with model() as work_model:
            variable('HoursPerDay', 8)
            variable('HoursPerWeek', lambda hpd: hpd*5, 'HoursPerDay',)

        with model() as calendar_model:
            variable('HoursPerDay', 24)
            stock('AccumulatedHours', lambda hpd: hpd, ('HoursPerDay',), 0)

        self.assertEqual(work_model['HoursPerWeek'][''], 40)
        self.assertEqual(calendar_model['AccumulatedHours'][''], 0)
        work_model.step(); calendar_model.step()
        self.assertEqual(work_model['HoursPerWeek'][''], 40)
        self.assertEqual(calendar_model['AccumulatedHours'][''], 24)
        work_model.step(); calendar_model.step()
        self.assertEqual(work_model['HoursPerWeek'][''], 40)
        self.assertEqual(calendar_model['AccumulatedHours'][''], 48)
        work_model.reset(); calendar_model.reset()
        self.assertEqual(work_model['HoursPerWeek'][''], 40)
        self.assertEqual(calendar_model['AccumulatedHours'][''], 0)

    def test_two_models_different_timing(self):
        """Test whether two models work with different timing of steps"""
        with model() as calendar_model_1:
            variable('HoursPerDay', 24)
            stock('AccumulatedHours', lambda hpd: hpd, ('HoursPerDay',), 0)

        with model() as calendar_model_2: 
            variable('HoursPerDay', 24)
            stock('AccumulatedHours', lambda hpd: hpd, ('HoursPerDay',), 0)

        self.assertEqual(calendar_model_1['AccumulatedHours'][''], 0)
        self.assertEqual(calendar_model_2['AccumulatedHours'][''], 0)
        calendar_model_1.step()
        self.assertEqual(calendar_model_1['AccumulatedHours'][''], 24)
        self.assertEqual(calendar_model_2['AccumulatedHours'][''], 0)
        calendar_model_1.step(); calendar_model_2.step()
        self.assertEqual(calendar_model_1['AccumulatedHours'][''], 48)
        self.assertEqual(calendar_model_2['AccumulatedHours'][''], 24)
        calendar_model_2.step(3)
        self.assertEqual(calendar_model_1['AccumulatedHours'][''], 48)
        self.assertEqual(calendar_model_2['AccumulatedHours'][''], 96)
        calendar_model_1.reset()
        self.assertEqual(calendar_model_1['AccumulatedHours'][''], 0)
        self.assertEqual(calendar_model_2['AccumulatedHours'][''], 96)


class UserSetAmount(unittest.TestCase):
    """For testing whether users can set the amount of variables"""

    def test_update_amount(self):
        """Can a constant take a new value?"""
        with model(treatments=['As is', 'To be']) as m:
            DischargeBegins = variable('DischargeBegins', 
                PerTreatment({'As is': 12, 'To be': 2}))

        self.assertEqual(DischargeBegins['As is'], 12)
        DischargeBegins['As is'] = 11
        self.assertEqual(DischargeBegins['As is'], 11)
        self.assertEqual(DischargeBegins['To be'], 2)
        m.reset(reset_external_vars=False)
        self.assertEqual(DischargeBegins['As is'], 11)
        self.assertEqual(DischargeBegins['To be'], 2)
        m.reset()
        self.assertEqual(DischargeBegins['As is'], 12)
        self.assertEqual(DischargeBegins['To be'], 2)

    def test_update_amount_no_arg(self):
        """Can a no arg variable take a new value?"""
        with model() as m:
            DischargeProgress = variable('DischargeProgress', lambda: 0.5)

        self.assertEqual(DischargeProgress[''], 0.5)
        DischargeProgress[''] = 0.75
        self.assertEqual(DischargeProgress[''], 0.75)
        m.step()
        self.assertEqual(DischargeProgress[''], 0.75)
        m.reset(reset_external_vars=False)
        self.assertEqual(DischargeProgress[''], 0.75)
        m.reset()
        self.assertEqual(DischargeProgress[''], 0.5)

    def test_update_amount_depends(self):
        """Can a variable take a new value when it has dependencies?"""
        with model() as m:
            Foo = variable('Foo', 9)
            Bar = variable('Bar', lambda f: f, 'Foo')

        self.assertEqual(Bar[''], 9)
        Foo[''] = 2.4
        m.recalculate()
        self.assertEqual(Bar[''], 2.4)
        m.reset(reset_external_vars=False)
        self.assertEqual(Bar[''], 2.4)
        Bar[''] = 8
        m.recalculate()
        self.assertEqual(Bar[''], 8)
        m.reset()
        self.assertEqual(Bar[''], 9)

    def test_update_amount_depends_constants(self):
        """Can a constant with constant dependencies take a new value?"""
        with model() as m:
            Foo = constant('Foo', 9)
            Bar = constant('Bar', lambda f: f, 'Foo')

        self.assertEqual(Bar[''], 9)
        Foo[''] = 2.4
        m.recalculate()
        self.assertEqual(Bar[''], 2.4)
        m.reset(reset_external_vars=False)
        self.assertEqual(Bar[''], 2.4)
        m.reset()
        self.assertEqual(Bar[''], 9)

    def test_update_depends_stock(self):
        """Can a stock with constant dependencies take a new value?"""
        with model() as m:
            Foo = stock('Foo', lambda: 1, (), lambda x: x, ('Bar',))
            Bar = constant('Bar', 99)

        self.assertEqual(m['Foo'][''], 99)
        m['Bar'][''] = 90
        m.recalculate()
        self.assertEqual(m['Foo'][''], 90)
        m.step()
        self.assertEqual(m['Foo'][''], 91)

    def test_update_depends_stock_chain(self):
        """Can a stock with change of constant dependencies take a new value?"""
        with model() as m:
            Foo = stock('Foo', lambda: 1, (), lambda x: x, ('Bar',))
            Bar = constant('Bar', lambda x: x, 'Baz')
            Baz = constant('Baz', 99)

        self.assertEqual(m['Foo'][''], 99)
        m['Baz'][''] = 90
        m.recalculate()
        self.assertEqual(m['Foo'][''], 90)
        m.step()
        self.assertEqual(m['Foo'][''], 91)

    def test_stock_with_user_setting_amount(self):
        """Test stock with user setting amount"""
        with model() as m:
            Foo = stock('Foo', 1, 0)

        m.step()
        self.assertEqual(Foo[''], 1)
        Foo[''] = 10
        self.assertEqual(Foo[''], 10)
        m.step()
        self.assertEqual(Foo[''], 11)
        m.reset()
        m.step()
        self.assertEqual(Foo[''], 1)
        Foo[''] = 7
        m.reset(reset_external_vars=False)
        self.assertEqual(Foo[''], 0)

    def test_user_setting_constant_multiple_treatments(self):
        """Can a user set the amount of a constant for multiple treatments?"""
        with model(treatments={'As is', 'To be'}) as m:
            DischargeBegins = variable('DischargeBegins', 
                PerTreatment({'As is': 10, 'To be': 8, 'Might be': 6}))
            DischargeEnds = variable('DischargeEnds', 14)
            DischargeDuration = variable('DischargeDuration', 
                lambda e, b: e - b, 'DischargeEnds', 'DischargeBegins')

        self.assertEqual(DischargeBegins['As is'], 10)
        self.assertEqual(DischargeDuration['To be'], 6)
        DischargeBegins['__all__'] = 9
        m.recalculate()
        self.assertEqual(DischargeBegins['As is'], 9)
        self.assertEqual(DischargeDuration['To be'], 5)

class ForeachDict(unittest.TestCase):
    """For testing the foreach construct with dictionaries"""
    def test_simple(self):
        """Does the simplest possible foreach work?"""
        with model():
            variable('Baz', {'foo': 12, 'bar': 13})
            Quz = variable('Quz', foreach(lambda f: f + 1), 'Baz')
        self.assertEqual(Quz[''], {'foo': 13, 'bar': 14})

    def test_two_arg_foreach(self):
        """Does a two arg callable to a foreach work?"""
        with model():
            variable('Baz', {'foo': 12, 'bar': 13})
            variable('Corge', {'foo': 0, 'bar': 99})
            Quz = variable('Quz', foreach(lambda b, c: b + c), 'Baz', 'Corge')
        self.assertEqual(Quz[''], {'foo': 12, 'bar': 112})

    def test_foreach_with_mismatch(self):
        """Does a two arg foreach with mismatched dicts error correctly?"""
        with self.assertRaisesRegex(MinnetonkaError, 
                'Foreach encountered mismatched dicts'):
            with model():
                variable('Baz', {'foo': 12, 'bar': 13})
                variable('Corge', {'foo': 0, 'wtf': 99})
                Quz = variable('Quz', 
                    foreach(lambda b, c: b + c), 'Baz', 'Corge')

    def test_big_dict_foreach(self):
        """Does foreach work with a 1000 element dict?"""
        with model():
            variable('Biggus', {'ind{:03}'.format(n): n for n in range(1000)})
            Dickus = variable('Dickus', foreach(lambda x: x*2), 'Biggus')
        self.assertEqual(Dickus['']['ind002'], 4)
        self.assertEqual(Dickus['']['ind999'], 1998)

    def test_foreach_nondict_error(self):
        """Does foreach raise error when first variable is not a dict?"""
        with self.assertRaisesRegex(MinnetonkaError,
                'First arg of foreach 23 must be dictionary or tuple'):
            with model():
                variable('Baz', 23)
                Quz = variable('Quz', foreach(lambda f: f + 1), 'Baz')

    def test_foreach_nondict_sunny_day(self):
        """Does foreach do the right thing with a nondict as second element?"""
        with model():
            variable('Baz', {'foo': 12, 'bar': 13})
            variable('Corge', 12)
            Quz = variable('Quz', foreach(lambda b, c: b + c), 'Baz', 'Corge')
        self.assertEqual(Quz[''], {'foo': 24, 'bar': 25})

    def test_foreach_stock(self):
        """Does foreach work with stocks and dicts?"""
        with model() as m:
            variable('Baz', {'foo': 12, 'bar': 13})
            variable('Waldo', {'foo': 1, 'bar': 2})
            Corge = stock('Corge', 
                foreach(lambda b: b+2), ('Baz',), 
                foreach(lambda w: w), ('Waldo',))
        m.step()
        self.assertEqual(Corge[''], {'foo':15, 'bar': 17} )
        m.step(2)
        self.assertEqual(Corge[''], {'foo':43, 'bar': 47} )

    def test_nested_foreach_stock(self):
        """Do nested foreaches work with stocks and dicts?"""
        with model() as m:
            Baz = variable('Baz', 
                {'drg001': {'trad': 7, 'rrc': 9},
                 'drg003': {'trad': 18, 'rrc': 4},
                 'drg257': {'trad': 6, 'rrc': 11}})
            Corge = stock('Corge',
                foreach(foreach(lambda x: x+1)), ('Baz',),
                {'drg001': {'trad': 0, 'rrc': 0},
                 'drg003': {'trad': 0, 'rrc': 0},
                 'drg257': {'trad': 0, 'rrc': 0}})
        m.step()
        self.assertEqual(
            Corge[''], 
            {'drg001': {'trad': 8, 'rrc': 10},
             'drg003': {'trad': 19, 'rrc': 5},
             'drg257': {'trad': 7, 'rrc': 12}})
        m.step(2)
        self.assertEqual(
            Corge[''], 
            {'drg001': {'trad': 24, 'rrc': 30},
             'drg003': {'trad': 57, 'rrc': 15},
             'drg257': {'trad': 21, 'rrc': 36}})

    def test_foreach_stock_timestep(self):
        """Does foreach work with stocks and dicts, and smaller timestep?"""
        with model(timestep=0.5) as m:
            variable('Baz', {'foo': 12, 'bar': 13})
            Corge = stock('Corge', 
                foreach(lambda b: b+2), ('Baz',), 
                {'foo': 0, 'bar': 0})
        m.step()
        self.assertEqual(Corge[''], {'foo':7, 'bar': 7.5} )
        m.step(2)
        self.assertEqual(Corge[''], {'foo':21, 'bar': 22.5} )

    def test_foreach_stock_multivariable(self):
        """Does foreach work with stocks that have multiple variables?"""
        with model() as m:
            variable('Baz', {'foo': 12, 'bar': 13})
            variable('Quz', {'foo': 1, 'bar': 2})
            Corge = stock('Corge', 
                foreach(lambda b, q: b+q), ('Baz', 'Quz'), 
                {'foo': 0, 'bar': 0})
        m.step()
        self.assertEqual(Corge[''], {'foo':13, 'bar': 15} )
        m.step(2)
        self.assertEqual(Corge[''], {'foo':39, 'bar': 45} )

    def test_foreach_accum(self):
        """Does foreach work with accums and dicts?"""
        with model() as m:
            variable('Baz', {'foo': 12, 'bar': 13})
            variable('Waldo', {'foo': 1, 'bar': 2})
            Corge = accum('Corge', 
                foreach(lambda b: b+2), ('Baz',), 
                foreach(lambda w: w), ('Waldo',))
        m.step()
        self.assertEqual(Corge[''], {'foo':15, 'bar': 17} )
        m.step(2)
        self.assertEqual(Corge[''], {'foo':43, 'bar': 47} )

    def test_nested_foreach_accum(self):
        """Do nested foreaches work with accums and dicts?"""
        with model() as m:
            Baz = variable('Baz', 
                {'drg001': {'trad': 7, 'rrc': 9},
                 'drg003': {'trad': 18, 'rrc': 4},
                 'drg257': {'trad': 6, 'rrc': 11}})
            Corge = accum('Corge',
                foreach(foreach(lambda x: x+1)), ('Baz',),
                {'drg001': {'trad': 0, 'rrc': 0},
                 'drg003': {'trad': 0, 'rrc': 0},
                 'drg257': {'trad': 0, 'rrc': 0}})
        m.step()
        self.assertEqual(
            Corge[''], 
            {'drg001': {'trad': 8, 'rrc': 10},
             'drg003': {'trad': 19, 'rrc': 5},
             'drg257': {'trad': 7, 'rrc': 12}})
        m.step(2)
        self.assertEqual(
            Corge[''], 
            {'drg001': {'trad': 24, 'rrc': 30},
             'drg003': {'trad': 57, 'rrc': 15},
             'drg257': {'trad': 21, 'rrc': 36}})

class ForeachTuples(unittest.TestCase):
    """For testing the foreach construct with tuples"""
    def test_simple(self):
        """Does the simplest possible foreach work with named tuples?"""
        with model():
            variable('Baz', (12, 13, 15))
            Quz = variable('Quz', foreach(lambda f: f + 1), 'Baz')
        self.assertEqual(Quz[''], (13, 14, 16))

    def test_two_arg_foreach(self):
        """Does a two arg callable to a foreach work?"""
        with model():
            variable('Baz', (12, 13, 0))
            variable('Corge', (0, 99, 12))
            Quz = variable('Quz', foreach(lambda b, c: b + c), 'Baz', 'Corge')
        self.assertEqual(Quz[''], (12, 112, 12))

    def test_foreach_with_mismatched_tuples(self):
        """Does a two arg foreach with mismatched tuples error correctly?"""
        with model():
            variable('Baz', (12, 13, 0))
            variable('Corge', (0, 99))
            Quz = variable('Quz', foreach(lambda b, c: b + c), 'Baz', 'Corge')
        self.assertEqual(Quz[''], (12, 112))

    def test_big_tuple_foreach(self):
        """Does foreach work with a 1000 element tuple?"""
        with model():
            variable('Biggus', tuple(range(1000)))
            Dickus = variable('Dickus', foreach(lambda x: x*2), 'Biggus')
        self.assertEqual(Dickus[''][3], 6)
        self.assertEqual(Dickus[''][999], 1998)

    def test_foreach_nontuple_sunny_day(self):
        """Does foreach do the right thing with a nontuple as second element?"""
        with model():
            variable('Baz', (12, 13))
            variable('Corge', 12)
            Quz = variable('Quz', foreach(lambda b, c: b + c), 'Baz', 'Corge')
        self.assertEqual(Quz[''], (24, 25))

    def test_foreach_stock(self):
        """Does foreach work with stocks?"""
        with model() as m:
            variable('Baz', (12, 13))
            variable('Waldo', (1, 2))
            Corge = stock('Corge', 
                foreach(lambda b: b+2), ('Baz',), 
                lambda w: w, ('Waldo',))
        m.step()
        self.assertEqual(Corge[''], (15, 17))
        m.step(2)
        self.assertEqual(Corge[''], (43, 47))

    def test_nested_foreach_stock(self):
        """Do nested foreaches work with stocks and tuples?"""
        with model() as m:
            Baz = variable('Baz', ((7, 9), (18, 4), (6, 11)))
            Corge = stock('Corge',
                foreach(foreach(lambda x: x+1)), ('Baz',),
                ((0, 0), (0, 0), (0, 0)))
        m.step()
        self.assertEqual(Corge[''], ((8, 10), (19, 5), (7, 12)))
        m.step(2)
        self.assertEqual(Corge[''], ((24, 30), (57, 15), (21, 36)))

    def test_foreach_stock_timestep(self):
        """Does foreach work with stocks?"""
        with model(timestep=0.5) as m:
            variable('Baz', (12, 13))
            Corge = stock('Corge', 
                foreach(lambda b: b+2), ('Baz',), 
                (0, 0))
        m.step()
        self.assertEqual(Corge[''], (7, 7.5))
        m.step(2)
        self.assertEqual(Corge[''], (21, 22.5))

    def test_foreach_stock_multivariable(self):
        """Does foreach work with stocks that have multiple variables?"""
        with model() as m:
            variable('Baz', (12, 13))
            variable('Quz', (1, 2))
            Corge = stock('Corge', 
                foreach(lambda b, q: b+q), ('Baz', 'Quz'), 
                (0, 0))
        m.step()
        self.assertEqual(Corge[''], (13, 15))
        m.step(2)
        self.assertEqual(Corge[''], (39, 45))

    def test_foreach_accum(self):
        """Does foreach work with accums?"""
        with model() as m:
            variable('Baz', (12, 13))
            variable('Waldo', (1, 2))
            Corge = accum('Corge', 
                foreach(lambda b: b+2), ('Baz',), 
                lambda w: w, ('Waldo',))
        m.step()
        self.assertEqual(Corge[''], (15, 17))
        m.step(2)
        self.assertEqual(Corge[''], (43, 47))

    def test_nested_foreach_accum(self):
        """Do nested foreaches work with accums and tuples?"""
        with model() as m:
            Baz = variable('Baz', ((7, 9), (18, 4), (6, 11)))
            Corge = accum('Corge',
                foreach(foreach(lambda x: x+1)), ('Baz',),
                ((0, 0), (0, 0), (0, 0)))
        m.step()
        self.assertEqual(Corge[''], ((8, 10), (19, 5), (7, 12)))
        m.step(2)
        self.assertEqual(Corge[''], ((24, 30), (57, 15), (21, 36)))


class ForeachNamedTuples(unittest.TestCase):
    """For testing the foreach construct with named tuples"""
    def setUp(self):
        self.drg = collections.namedtuple('drg', ['drg001', 'drg003', 'drg257'])
        self.site = collections.namedtuple('site', ['traditional', 'rrc'])

    def test_simple(self):
        """Does the simplest possible foreach work with named tuples?"""
        with model():
            variable('Baz', self.drg(12, 13, 15))
            Quz = variable('Quz', foreach(lambda f: f + 1), 'Baz')
        self.assertEqual(Quz[''], self.drg(13, 14, 16))

    def test_two_arg_foreach(self):
        """Does a two arg callable to a foreach work?"""
        with model():
            variable('Baz', self.drg(12, 13, 0))
            variable('Corge', self.drg(0, 99, 12))
            Quz = variable('Quz', foreach(lambda b, c: b + c), 'Baz', 'Corge')
        self.assertEqual(Quz[''], self.drg(12, 112, 12))

    def test_foreach_scalar_sunny_day(self):
        """Does foreach do the right thing with a scalar as second element?"""
        with model():
            variable('Baz', self.drg(12, 13, 19))
            variable('Corge', 12)
            Quz = variable('Quz', foreach(lambda b, c: b + c), 'Baz', 'Corge')
        self.assertEqual(Quz[''], self.drg(24, 25, 31))

    def test_foreach_scalar_sunny_day_third_elt(self):
        """Does foreach do the right thing with a scalar as third element?"""
        with model():
            variable('Baz', self.drg(12, 13, 19))
            variable('Grault', self.drg(0, 0, 2))
            variable('Corge', 12)
            Quz = variable('Quz', 
                foreach(lambda b, g, c: b + g + c), 'Baz', 'Grault', 'Corge')
        self.assertEqual(Quz[''], self.drg(24, 25, 33))

    def test_nested_foreach(self):
        """Do nested namedtuple foreaches work?""" 
        with model():
            variable('Baz', 
                self.drg(self.site(12, 9), self.site(13, 4), self.site(19, 18)))
            variable('Grault', 
                self.drg(self.site(1, 2), self.site(3, 4), self.site(5, 6)))
            Qux = variable('Qux',
                foreach(foreach(lambda b, g: b+g)), 'Baz', 'Grault')
        self.assertEqual(
            Qux[''], 
            self.drg(self.site(13, 11), self.site(16, 8), self.site(24, 24)))

    def test_nested_foreach_one_level_const(self):
        """Do nested namedtuple foreaches work, with one level const?""" 
        with model():
            variable('Baz', 
                self.drg(self.site(12, 9), self.site(13, 4), self.site(19, 18)))
            variable('Grault', self.drg(1, 2, 3))
            Qux = variable('Qux',
                foreach(foreach(lambda b, g: b+g)), 'Baz', 'Grault')
        self.assertEqual(
            Qux[''], 
            self.drg(self.site(13, 10), self.site(15, 6), self.site(22, 21)))

    def test_nested_foreach_two_levels_const(self):
        """Do nested namedtuple foreaches work, with two levels const?""" 
        with model():
            variable('Baz', 
                self.drg(self.site(12, 9), self.site(13, 4), self.site(19, 18)))
            variable('Grault', 9)
            Qux = variable('Qux',
                foreach(foreach(lambda b, g: b+g)), 'Baz', 'Grault')
        self.assertEqual(
            Qux[''], 
            self.drg(self.site(21, 18), self.site(22, 13), self.site(28, 27)))

    def test_foreach_stock(self):
        """Does foreach work with stocks and mn named tuples?"""
        with model() as m:
            variable('Baz', self.drg(12, 13, 19))
            Corge = stock('Corge', 
                foreach(lambda b: b+2), ('Baz',), 
                self.drg(0, 0, 0))
        m.step()
        self.assertEqual(Corge[''], self.drg(14, 15, 21))
        m.step(2)
        self.assertEqual(Corge[''], self.drg(42, 45, 63))

    def test_nested_foreach_stock(self):
        """Do nested foreaches work with stocks and named tuples?""" 
        with model() as m:
            Baz = variable('Baz', 
                self.drg(self.site(7, 9), self.site(18, 4), self.site(6, 11)))
            Corge = stock('Corge',
                foreach(foreach(lambda x: x+1)), ('Baz',),
                self.drg(self.site(0, 0), self.site(0, 0), self.site(0, 0)))
        m.step()
        self.assertEqual(
            Corge[''], 
            self.drg(self.site(8, 10), self.site(19, 5), self.site(7, 12)))
        m.step(2)
        self.assertEqual(
            Corge[''], 
            self.drg(self.site(24, 30), self.site(57, 15), self.site(21, 36)))

    def test_foreach_stock_timestep(self):
        """Does foreach work with stocks and mn named tuples?"""
        with model(timestep=0.5) as m:
            variable('Baz', self.drg(12, 13, 19))
            Corge = stock('Corge', 
                foreach(lambda b: b+2), ('Baz',), 
                self.drg(0, 0, 0))
        m.step()
        self.assertEqual(Corge[''], self.drg(7, 7.5, 10.5))
        m.step(2)
        self.assertEqual(Corge[''], self.drg(21, 22.5, 31.5))

    def test_foreach_stock_multivariable(self):
        """Does foreach work with stocks that have multiple variables?"""
        with model() as m:
            variable('Baz', self.drg(12, 13, 19))
            variable('Quz', self.drg(1, 2, 3))
            Corge = stock('Corge', 
                foreach(lambda b, q: b+q), ('Baz', 'Quz'), 
                self.drg(0, 0, 0))
        m.step()
        self.assertEqual(Corge[''], self.drg(13, 15, 22))
        m.step(2)
        self.assertEqual(Corge[''], self.drg(39, 45, 66))

    def test_foreach_accum(self):
        """Does foreach work with accums and mn named tuples?"""
        with model() as m:
            variable('Baz', self.drg(12, 13, 19))
            Corge = accum('Corge', 
                foreach(lambda b: b+2), ('Baz',), 
                self.drg(0, 0, 0))
        m.step()
        self.assertEqual(Corge[''], self.drg(14, 15, 21))
        m.step(2)
        self.assertEqual(Corge[''], self.drg(42, 45, 63))

    def test_nested_foreach_accum(self):
        """Do nested foreaches work with accums and named tuples?""" 
        with model() as m:
            Baz = variable('Baz', 
                self.drg(self.site(7, 9), self.site(18, 4), self.site(6, 11)))
            Corge = accum('Corge',
                foreach(foreach(lambda x: x+1)), ('Baz',),
                self.drg(self.site(0, 0), self.site(0, 0), self.site(0, 0)))
        m.step()
        self.assertEqual(
            Corge[''], 
            self.drg(self.site(8, 10), self.site(19, 5), self.site(7, 12)))
        m.step(2)
        self.assertEqual(
            Corge[''], 
            self.drg(self.site(24, 30), self.site(57, 15), self.site(21, 36)))


class ForeachMixed(unittest.TestCase):
    """For testing the foreach construct on mixed data."""
    def setUp(self):
        self.drg = collections.namedtuple('drg', ['drg001', 'drg003', 'drg257'])
        self.site = collections.namedtuple('site', ['traditional', 'rrc'])

    def test_dict_tuple(self):
        """Do nested foreaches work with tuples inside dicts?"""
        with model() as m:
            Baz = variable('Baz', 
                {'drg001': (7, 9), 'drg003': (18, 4), 'drg257': (6, 11)})
            Corge = stock('Corge',
                foreach(foreach(lambda x: x+1)), ('Baz',),
                {'drg001': (0, 0), 'drg003': (0, 0), 'drg257': (0,0)})
        m.step()
        self.assertEqual(
            Corge[''], 
            {'drg001': (8, 10), 'drg003': (19, 5), 'drg257': (7, 12)})
        m.step(2)
        self.assertEqual(
            Corge[''], 
            {'drg001': (24, 30), 'drg003': (57, 15), 'drg257': (21, 36)})

    def test_dict_namedtuple(self):
        """Does nested foreaches work with named tuples inside dicts?"""
        with model():
            Baz = variable('Baz', foreach(foreach(lambda x: x+1)), 'Grault')
            Grault = constant('Grault',
                {'drg001': self.site(7, 9),
                 'drg003': self.site(18, 4),
                 'drg257': self.site(6, 11)})
        self.assertEqual(
            Baz[''],
            {'drg001': self.site(8, 10),
             'drg003': self.site(19, 5),
             'drg257': self.site(7, 12)})

    def test_namedtuple_tuple(self):
        """Do nested foreaches work with tuples inside named tuples?"""
        with model() as m:
            Baz = variable('Baz', 
                self.drg((7, 9), (18, 4), (6, 11)))
            Corge = stock('Corge',
                foreach(foreach(lambda x: x+1)), ('Baz',),
                self.drg((0, 0), (0, 0), (0, 0)))
        m.step()
        self.assertEqual(
            Corge[''], 
            self.drg((8, 10), (19, 5), (7, 12)))
        m.step(2)
        self.assertEqual(
            Corge[''], 
            self.drg((24, 30), (57, 15), (21, 36)))

    def test_namedtuple_dict(self):
        """Do nested foreaches work with dicts inside named tuples?"""
        with model() as m:
            Baz = variable('Baz', 
                self.drg({'trad': 7, 'rrc': 9}, 
                         {'trad': 18, 'rrc': 4}, 
                         {'trad': 6, 'rrc': 11}))
            Corge = stock('Corge',
                foreach(foreach(lambda x: x+1)), ('Baz',),
                self.drg({'trad': 0, 'rrc': 0}, 
                         {'trad': 0, 'rrc': 0}, 
                        {'trad': 0, 'rrc': 0}))
        m.step()
        self.assertEqual(
            Corge[''], 
            self.drg({'trad': 8, 'rrc': 10}, 
                     {'trad': 19, 'rrc': 5}, 
                     {'trad': 7, 'rrc': 12}))
        m.step(2)
        self.assertEqual(
            Corge[''], 
            self.drg({'trad': 24, 'rrc': 30}, 
                     {'trad': 57, 'rrc': 15}, 
                     {'trad': 21, 'rrc': 36}))

    def test_tuple_namedtuple(self):
        """Do nested foreaches work with named tuples inside tuples?"""
        with model() as m:
            Baz = variable('Baz', 
                (self.site(7, 9), self.site(18, 4), self.site(6, 11)))
            Corge = stock('Corge',
                foreach(foreach(lambda x: x+1)), ('Baz',),
                (self.site(0, 0), self.site(0, 0), self.site(0, 0)))
        m.step()
        self.assertEqual(
            Corge[''], 
            (self.site(8, 10), self.site(19, 5), self.site(7, 12)))
        m.step(2)
        self.assertEqual(
            Corge[''], 
            (self.site(24, 30), self.site(57, 15), self.site(21, 36)))

    def test_tuple_dict(self):
        """Do nested foreaches work with dicts inside tuples?"""
        with model() as m:
            Baz = variable('Baz', 
                ({'trad': 7, 'rrc': 9}, {'trad': 18, 'rrc': 4}, 
                 {'trad': 6, 'rrc': 11}))
            Corge = stock('Corge',
                foreach(foreach(lambda x: x+1)), ('Baz',),
                ({'trad': 0, 'rrc': 0}, {'trad': 0, 'rrc': 0},
                 {'trad': 0, 'rrc': 0}))
        m.step()
        self.assertEqual(
            Corge[''], 
            ({'trad': 8, 'rrc': 10}, {'trad': 19, 'rrc': 5},
             {'trad': 7, 'rrc': 12}))
        m.step(2)
        self.assertEqual(
            Corge[''], 
            ({'trad': 24, 'rrc': 30}, {'trad': 57, 'rrc': 15},
             {'trad': 21, 'rrc': 36}))

class Previous(unittest.TestCase):
    """For testing previous"""
    def test_previous(self):
        """Does a simple value of previous work, with a stock?"""
        with model() as m:
            stock('Foo', 1, 0)
            LastFoo = previous('LastFoo', 'Foo')

        self.assertEqual(LastFoo[''], 0)
        m.step()
        self.assertEqual(LastFoo[''], 0)
        m.step()
        self.assertEqual(LastFoo[''], 1)
        m.step()
        self.assertEqual(LastFoo[''], 2)
        m.reset()
        self.assertEqual(LastFoo[''], 0)

    def test_previous_reversed_order(self):
        """Does a simple value of previous work, with a stock?"""
        with model() as m:
            LastFoo = previous('LastFoo', 'Baz')
            variable('Baz', lambda x: x, 'Foo')
            stock('Foo', 1, 0)

        self.assertEqual(LastFoo[''], 0)
        m.step()
        self.assertEqual(LastFoo[''], 0)
        m.step()
        self.assertEqual(LastFoo[''], 1)
        m.step()
        self.assertEqual(LastFoo[''], 2)
        m.reset()
        self.assertEqual(LastFoo[''], 0)

    def test_previous_with_docstring(self):
        """Does a simple value of previous work, with a stock?"""
        with model() as m:
            stock('Foo', 1, 0)
            LastFoo = previous('LastFoo', 'Simple previous', 'Foo')

        self.assertEqual(LastFoo.__doc__, 'Simple previous')
        self.assertEqual(LastFoo[''], 0)
        m.step()
        self.assertEqual(LastFoo[''], 0)
        m.step()
        self.assertEqual(LastFoo[''], 1)
        m.step()
        self.assertEqual(LastFoo[''], 2)
        m.reset()
        self.assertEqual(LastFoo[''], 0)

    def test_previous_small_timestep(self):
        """Does a simple value of previous work, with non-1 timestep?"""
        with model(timestep=0.5) as m:
            stock('Foo', 1, 0)
            LastFoo = previous('LastFoo', 'Foo')

        self.assertEqual(LastFoo[''], 0)
        m.step()
        self.assertEqual(LastFoo[''], 0)
        m.step()
        self.assertEqual(LastFoo[''], 0.5)
        m.step()
        self.assertEqual(LastFoo[''], 1)
        m.reset()
        self.assertEqual(LastFoo[''], 0)

    def test_previous_with_treatments(self):
        """Does a simple value of previous work, with treatments?"""
        with model(treatments=['As is', 'To be']) as m:
            stock('Foo', PerTreatment({'As is': 1, 'To be': 2}), 0)
            LastFoo = previous('LastFoo', 'Foo')

        self.assertEqual(LastFoo['As is'], 0)
        self.assertEqual(LastFoo['To be'], 0)
        m.step()
        self.assertEqual(LastFoo['As is'], 0)
        self.assertEqual(LastFoo['To be'], 0) 
        m.step()
        self.assertEqual(LastFoo['As is'], 1)
        self.assertEqual(LastFoo['To be'], 2)
        m.step()
        self.assertEqual(LastFoo['As is'], 2)
        self.assertEqual(LastFoo['To be'], 4)
        m.reset()
        self.assertEqual(LastFoo['As is'], 0)
        self.assertEqual(LastFoo['To be'], 0)

    def test_previous_with_namedtuple(self):
        """Does a simple value of previous work, with a mn_namedtuple?"""
        Payer = mn_namedtuple('Payer', ['Medicare', 'Medicaid', 'Commercial'])
        with model() as m:
            stock('Foo', Payer(1, 2, 3), Payer(0, 0, 0))
            LastFoo = previous('LastFoo', 'Foo')

        self.assertEqual(LastFoo[''], Payer(0, 0, 0))
        m.step()
        self.assertEqual(LastFoo[''], Payer(0, 0, 0))
        m.step()
        self.assertEqual(LastFoo[''], Payer(1, 2, 3))
        m.step()
        self.assertEqual(LastFoo[''], Payer(2, 4, 6))
        m.reset()
        self.assertEqual(LastFoo[''], Payer(0, 0, 0))

    def test_previous_with_initial_value(self):
        """Does a simple value of previous work, with an initial value?"""
        with model() as m:
            stock('Foo', 1, 0)
            LastFoo = previous('LastFoo', 'Foo', 0.3)

        self.assertEqual(LastFoo[''], 0.3)
        m.step()
        self.assertEqual(LastFoo[''], 0)
        m.step()
        self.assertEqual(LastFoo[''], 1)
        m.step()
        self.assertEqual(LastFoo[''], 2)
        m.reset()
        self.assertEqual(LastFoo[''], 0.3)

    def test_previous_with_initial_value_reversed_order(self):
        """Does a simple value of previous work, with an initial value?"""
        with model() as m:
            LastFoo = previous('LastFoo', 'Foo', 0.3)
            stock('Foo', 1, 0)

        self.assertEqual(LastFoo[''], 0.3)
        m.step()
        self.assertEqual(LastFoo[''], 0)
        m.step()
        self.assertEqual(LastFoo[''], 1)
        m.step()
        self.assertEqual(LastFoo[''], 2)
        m.reset()
        self.assertEqual(LastFoo[''], 0.3)

    def test_previous_with_initial_value_and_docstring(self):
        """Does a simple value of previous work, with an initial value?"""
        with model() as m:
            stock('Foo', 1, 0)
            LastFoo = previous('LastFoo', 'docstring', 'Foo', 0.3)

        self.assertEqual(LastFoo.__doc__, 'docstring')
        self.assertEqual(LastFoo[''], 0.3)
        m.step()
        self.assertEqual(LastFoo[''], 0)
        m.step()
        self.assertEqual(LastFoo[''], 1)
        m.step()
        self.assertEqual(LastFoo[''], 2)
        m.reset()
        self.assertEqual(LastFoo[''], 0.3)

    def test_previous_with_constant(self):
        """Does a previous of a constant work?"""
        with model() as m:
            constant('Foo', 12)
            LastFoo = previous('LastFoo', 'Foo', 0)

        self.assertEqual(LastFoo[''], 12)
        m.step()
        self.assertEqual(LastFoo[''], 12)
        m.step()
        self.assertEqual(LastFoo[''], 12)
        m.reset()
        self.assertEqual(LastFoo[''], 12)
        m.step()
        self.assertEqual(LastFoo[''], 12)

    def test_previous_with_circularity(self):
        """Does a previous work when it defines an apparent circularity?"""
        with model() as m:
            previous('LastFoo', 'Foo', 0)
            Foo = variable('Foo', lambda x: x + 2, 'LastFoo')

        self.assertEqual(Foo[''], 2)
        m.step()
        self.assertEqual(Foo[''], 4)
        m.step()
        self.assertEqual(Foo[''], 6)
        m.reset()
        self.assertEqual(Foo[''], 2)
        m.step()
        self.assertEqual(Foo[''], 4)

    def test_self_previous(self):
        """Does a previous work when it refers to itself?"""
        with model() as m:
            Foo = previous('Foo', 'Foo', 0)

        self.assertEqual(Foo[''], 0)
        m.step()
        self.assertEqual(Foo[''], 0)
        m.step()
        self.assertEqual(Foo[''], 0)
        m.reset()
        self.assertEqual(Foo[''], 0)
        m.step()
        self.assertEqual(Foo[''], 0)

    def test_set_previous(self):
        """Does setting the amount of a previous raise an error?"""
        with model() as m:
            stock('Foo', 1, 0)
            LastFoo = previous('LastFoo', 'docstring', 'Foo', 0.3)
        with self.assertRaises(MinnetonkaError) as me:
            LastFoo[''] = 12
        self.assertEqual(
            me.exception.message, 
            'Amount of <Previous LastFoo> cannot be changed outside model logic'
            )


class OldValues(unittest.TestCase):
    """For checking that values are stored every step"""
    def test_stock_old_values(self):
        """Does a stock keep all the old values around?"""
        with model(treatments=['Bar', 'Baz']) as m:
            Foo = stock('Foo', PerTreatment({'Bar': 1, 'Baz': 2}), 0)

        m.step(6)
        self.assertEqual(Foo.history('Bar', 0), 0)
        self.assertEqual(Foo.history('Baz', 0), 0)
        self.assertEqual(Foo.history('Bar', 1), 1)
        self.assertEqual(Foo.history('Baz', 1), 2)
        self.assertEqual(Foo.history('Bar', 2), 2)
        self.assertEqual(Foo.history('Baz', 2), 4)
        self.assertEqual(Foo.history('Bar', 3), 3)
        self.assertEqual(Foo.history('Baz', 3), 6)
        self.assertEqual(Foo.history('Bar', 5), 5)
        self.assertEqual(Foo.history('Baz', 5), 10)
        m.reset()
        m.step(2)
        self.assertEqual(Foo.history('Bar', 0), 0)
        self.assertEqual(Foo.history('Baz', 0), 0)
        self.assertEqual(Foo.history('Bar', 1), 1)
        self.assertEqual(Foo.history('Baz', 1), 2)
        with self.assertRaises(MinnetonkaError) as me:
            Foo.history('Bar', 3)
        self.assertEqual(
            me.exception.message, "Foo['Bar'] has no value for step 3")

    def test_variable_old_values(self):
        """Does a variable keep all the old values around?"""
        with model(treatments=['Bar', 'Baz']) as m:
            stock('Foo', PerTreatment({'Bar': 1, 'Baz': 2}), 0)
            Quz = variable('Quz', lambda x: x, 'Foo')

        m.step(6)
        self.assertEqual(Quz.history('Bar', 0), 0)
        self.assertEqual(Quz.history('Baz', 0), 0)
        self.assertEqual(Quz.history('Bar', 1), 1)
        self.assertEqual(Quz.history('Baz', 1), 2)
        self.assertEqual(Quz.history('Bar', 2), 2)
        self.assertEqual(Quz.history('Baz', 2), 4)
        self.assertEqual(Quz.history('Bar', 3), 3)
        self.assertEqual(Quz.history('Baz', 3), 6)
        self.assertEqual(Quz.history('Bar', 5), 5)
        self.assertEqual(Quz.history('Baz', 5), 10)
        m.reset()
        m.step(2)
        self.assertEqual(Quz.history('Bar', 0), 0)
        self.assertEqual(Quz.history('Baz', 0), 0)
        self.assertEqual(Quz.history('Bar', 1), 1)
        self.assertEqual(Quz.history('Baz', 1), 2)
        with self.assertRaises(MinnetonkaError) as me:
            Quz.history('Bar', 3)
        self.assertEqual(
            me.exception.message, "Quz['Bar'] has no value for step 3")

    def test_accum_old_values(self):
        """Does an accum keep all the old values around?"""
        with model(treatments=['Bar', 'Baz']) as m:
            Foo = accum('Foo', PerTreatment({'Bar': 1, 'Baz': 2}), 0)

        m.step(6)
        self.assertEqual(Foo.history('Bar', 0), 0)
        self.assertEqual(Foo.history('Baz', 0), 0)
        self.assertEqual(Foo.history('Bar', 1), 1)
        self.assertEqual(Foo.history('Baz', 1), 2)
        self.assertEqual(Foo.history('Bar', 2), 2)
        self.assertEqual(Foo.history('Baz', 2), 4)
        self.assertEqual(Foo.history('Bar', 3), 3)
        self.assertEqual(Foo.history('Baz', 3), 6)
        self.assertEqual(Foo.history('Bar', 5), 5)
        self.assertEqual(Foo.history('Baz', 5), 10)
        m.reset()
        m.step(2)
        self.assertEqual(Foo.history('Bar', 0), 0)
        self.assertEqual(Foo.history('Baz', 0), 0)
        self.assertEqual(Foo.history('Bar', 1), 1)
        self.assertEqual(Foo.history('Baz', 1), 2)
        with self.assertRaises(MinnetonkaError) as me:
            Foo.history('Bar', 3)
        self.assertEqual(
            me.exception.message, "Foo['Bar'] has no value for step 3")

    def test_constant_old_values(self):
        """Does a constant do the right thing for history() calls?"""
        with model(treatments=['Bar', 'Baz']) as m:
            Quz = constant('Quz', PerTreatment({'Bar': 9, 'Baz':10}))

        m.step(6)
        self.assertEqual(Quz.history('Bar', 0), 9)
        self.assertEqual(Quz.history('Baz', 0), 10) 
        self.assertEqual(Quz.history('Bar', 3), 9)
        self.assertEqual(Quz.history('Baz', 3), 10)
        m.reset()
        m.step(2)
        self.assertEqual(Quz.history('Bar', 0), 9)
        self.assertEqual(Quz.history('Baz', 0), 10) 
        self.assertEqual(Quz.history('Bar', 99), 9)
        self.assertEqual(Quz.history('Baz', 99), 10)

    def test_old_derived_values(self):
        """Does history do the right thing if the treatment is derived?"""
        with model(treatments=['Bar', 'Baz'],
                   derived_treatments={'Quz': AmountBetter('Baz', 'Bar')}
        ) as m:
            Foo = stock('Foo', PerTreatment({'Bar': 1, 'Baz': 2}), 0
            ).derived()

        m.step(6)
        self.assertEqual(Foo.history('Quz', 0), 0) 
        self.assertEqual(Foo.history('Quz', 1), 1) 
        self.assertEqual(Foo.history('Quz', 2), 2) 
        self.assertEqual(Foo.history('Baz', 1), 2)
        

class ModelHistory(unittest.TestCase):
    """Testing history of the whole model."""
    def test_history(self):
        """Test history of several variables and two treatments."""
        with model(treatments=['Bar', 'Baz']) as m:
            Foo = stock('Foo', PerTreatment({'Bar': 1, 'Baz': 2}), 0)
            Quz = variable('Quz', lambda x: x, 'Foo')
            Corge = accum('Corge', PerTreatment({'Bar': 1, 'Baz': 2}), 0)
            Grault = constant('Grault', PerTreatment({'Bar': 9, 'Baz':10}))
            Thud = variable('Thud', lambda x: x, 'Foo').no_history()

        self.assertEqual(
            m.history(),
            {
                'Foo': {'Bar': [0], 'Baz': [0]},
                'Quz': {'Bar': [0], 'Baz': [0]},
                'Corge': {'Bar': [0], 'Baz': [0]} 
            })

        m.step(10)

        self.assertEqual(
            m.history(),
            {
                'Foo': {
                    'Bar': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                    'Baz': [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
                    },
                'Quz': {
                    'Bar': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                    'Baz': [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
                    },
                'Corge': {
                    'Bar': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                    'Baz': [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
                    }
            })


    def test_derived_history(self):
        """Test history of several variables and two treatments."""
        with model(treatments=['Bar', 'Baz'], 
                   derived_treatments={
                        'Plugh': AmountBetter('Baz', 'Bar')}
            ) as m:
            Foo = stock('Foo', PerTreatment({'Bar': 1, 'Baz': 2}), 0
            ).derived()
            Quz = variable('Quz', lambda x: x, 'Foo'
            ).derived(scored_as='golf')
            Corge = accum('Corge', PerTreatment({'Bar': 1, 'Baz': 2}), 0
            ).derived()
            Grault = constant('Grault', PerTreatment({'Bar': 9, 'Baz':10})
            ).derived()
            Thud = variable('Thud', lambda x: x, 'Foo').no_history(
            ).derived()
            Fred = variable('Fred', 
                lambda foo, quz: {'foo': foo, 'quz': quz},
                'Foo', 'Quz'
            ).derived(scored_as='combo')

        self.assertEqual(
            m.history(),
            {
                'Foo': {'Plugh': [0]},
                'Quz': {'Plugh': [0]},
                'Corge': {'Plugh': [0]},
                'Fred': {'Plugh': [{'foo': 0, 'quz': 0}]}
            })

        m.step(10)

        self.assertEqual(
            m.history(),
            {
                'Foo': {'Plugh': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]},
                'Quz': {'Plugh': [0, -1, -2, -3, -4, -5, -6, -7, -8, -9, -10]},
                'Corge': {'Plugh': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]},
                'Fred': {'Plugh': [
                    {'foo': 0, 'quz': 0}, {'foo': 1, 'quz': -1}, 
                    {'foo': 2, 'quz': -2}, {'foo': 3, 'quz': -3}, 
                    {'foo': 4, 'quz': -4}, {'foo': 5, 'quz': -5}, 
                    {'foo': 6, 'quz': -6}, {'foo': 7, 'quz': -7}, 
                    {'foo': 8, 'quz': -8}, {'foo': 9, 'quz': -9}, 
                    {'foo': 10, 'quz': -10}]}
            })

        self.assertEqual(
            m.history(base=True)['Foo']['Bar'],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10] )
        self.assertEqual(
            m.history(base=True)['Quz']['Baz'],
            [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20] )
        self.assertEqual(
            m.history(base=True)['Corge']['Bar'],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10] )


class Unitary(unittest.TestCase):
    """For testing variables that do not differ by treatment."""

    def assert_unitary(self, variable):
        """Assert that this variable is the same for 'As is' and 'To be'."""
        self.assertEqual(
            variable.by_treatment('As is'), variable.by_treatment('To be'))

    def assert_not_unitary(self, variable):
        """Assert that this variable is not the same for 'As is' and 'To be'."""
        self.assertNotEqual(
            variable.by_treatment('As is'), variable.by_treatment('To be'))

    def test_simple_variable(self):
        """Test whether a simple variable can be unitary."""
        with model(treatments=['As is', 'To be']):
            Bar = variable('Bar', 12)

        self.assert_unitary(Bar)
        self.assertEqual(Bar['As is'], 12)
        self.assertEqual(Bar['To be'], 12)
        Bar['As is'] = 13
        self.assertEqual(Bar['As is'], 13)
        self.assertEqual(Bar['To be'], 13)

    def test_per_treatment(self):
        """Test whether a variable defined with PerTreatment is unitary."""
        with model(treatments=['As is', 'To be']):
            Foo = variable('Foo', PerTreatment({'As is': 12, 'To be': 13}))

        self.assert_not_unitary(Foo)
        self.assertEqual(Foo['As is'], 12)
        self.assertEqual(Foo['To be'], 13)
        Foo['As is'] = 14
        self.assertEqual(Foo['As is'], 14)
        self.assertEqual(Foo['To be'], 13)

    def test_variables_that_depend(self):
        """Test whether a variable that depends on a unitary is unitary."""
        with model(treatments=['As is', 'To be']):
            Unitary = variable('Unitary', 12)
            Fragmented = variable('Fragmented', 
                PerTreatment({'As is': 12, 'To be': 13}))
            DependsOnUnitary = variable('DependsOnUnitary', 
                lambda x: x, 'Unitary')
            DependsOnFragmented = variable('DependsOnFragmented', 
                lambda x: x, 'Fragmented')
        
        self.assert_unitary(DependsOnUnitary)
        self.assert_not_unitary(DependsOnFragmented)

    def test_unitary_stock(self):
        """Test whether a stock can be unitary."""
        with model(treatments=['As is', 'To be']):
            SimpleStock = stock('SimpleStock', 1, 1)
            UnitaryVar = variable('UnitaryVar', 2)
            FragmentedVar = variable('FragmentedVar', 
                PerTreatment({'As is': 2, 'To be': 3}))
            UnitaryStock = stock('UnitaryStock', 
                lambda x: x, ('UnitaryVar',), 0)
            FragmentedStock1 = stock('FragmentedStock1',
                lambda x: x, ('FragmentedVar',), 
                lambda x: x, ('UnitaryVar',))
            FragmentedStock2 = stock('FragmentedStock2',
                lambda x: x, ('UnitaryVar',),
                lambda x: x, ('FragmentedVar',))
            FragmentedStock3 = stock('FragmentedStock3',
                lambda x: x, ('UnitaryVar',),
                lambda x: x, ('FragmentedVar',))

        self.assert_unitary(SimpleStock)
        self.assert_unitary(UnitaryStock)
        self.assert_not_unitary(FragmentedStock1)
        self.assert_not_unitary(FragmentedStock2)
        self.assert_not_unitary(FragmentedStock3)

    def test_unitary_accum(self):
        """Test whether an accum can be unitary."""
        with model(treatments=['As is', 'To be']):
            SimpleAccum = accum('SimpleAccum', 1, 1)
            UnitaryVar = variable('UnitaryVar', 2)
            FragmentedVar = variable('FragmentedVar', 
                PerTreatment({'As is': 2, 'To be': 3}))
            UnitaryAccum = accum('UnitaryAccum', 
                lambda x: x, ('UnitaryVar',), 0)
            FragmentedAccum1 = accum('FragmentedAccum1',
                lambda x: x, ('FragmentedVar',), 
                lambda x: x, ('UnitaryVar',))
            FragmentedAccum2 = accum('FragmentedAccum2',
                lambda x: x, ('UnitaryVar',),
                lambda x: x, ('FragmentedVar',))
            FragmentedAccum3 = accum('FragmentedAccum3',
                lambda x: x, ('UnitaryVar',),
                lambda x: x, ('FragmentedVar',))

        self.assert_unitary(SimpleAccum)
        self.assert_unitary(UnitaryAccum)
        self.assert_not_unitary(FragmentedAccum1)
        self.assert_not_unitary(FragmentedAccum2)
        self.assert_not_unitary(FragmentedAccum3)

    def test_previous(self):
        """Test whether a previous can be unitary."""
        with model(treatments=['As is', 'To be']):
            UnitaryVar = variable('UnitaryVar', 2)
            FragmentedVar = variable('FragmentedVar', 
                PerTreatment({'As is': 2, 'To be': 3}))
            UnitaryPrevious = previous('UnitaryPrevious', 'UnitaryVar')
            FragmentedPrevious = previous('FragmentedPrevious', 'FragmentedVar')

        self.assert_unitary(UnitaryPrevious)
        self.assert_not_unitary(FragmentedPrevious)

    def test_causal_loop_unitary(self):
        """Test that a simple model with a causal loop is unitary."""
        with model(treatments=['As is', 'To be']) as m2:
            InterestRate = constant('InterestRate', 0.04)
            Interest = variable('Interest', 
                lambda s, ir: s * ir, 'Savings', 'InterestRate')
            Savings = stock('Savings', lambda i: i, ('Interest',), 1000)
        self.assert_unitary(InterestRate)
        self.assert_unitary(Interest)
        self.assert_unitary(Savings)

    def test_causal_loop_not_unitary(self):
        """Test that a simple model with a causal loop is not unitary."""
        with model(treatments=['As is', 'To be']) as m2:
            InterestRate = constant('InterestRate', 
                PerTreatment({'As is': 0.04, 'To be': 0.15}))
            Interest = variable('Interest', 
                lambda s, ir: s * ir, 'Savings', 'InterestRate')
            Savings = stock('Savings', lambda i: i, ('Interest',), 1000)
        self.assert_not_unitary(InterestRate)
        self.assert_not_unitary(Interest)
        self.assert_not_unitary(Savings)

    def test_unitary_set_warning(self):
        """Test that setting a unitary var in one treatment issues warning."""
        with model(treatments=['As is', 'To be']):
            InterestRate = constant('InterestRate', 0.03)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            InterestRate['To be'] = 0.05
            self.assertEqual(len(w), 1)
            self.assertEqual(w[-1].category, MinnetonkaWarning)
            self.assertEqual(str(w[-1].message), 
                'Setting amount of unitary variable InterestRate '+
                'in only one treatment')


class SafeDiv(unittest.TestCase):
    """For testing safe_div"""
    def test_safe_div(self):
        """Testing safe_div"""
        self.assertEqual(safe_div(5, 4), 1.25)
        self.assertEqual(safe_div(5, 0), 0)
        self.assertEqual(safe_div(5, 0, 1), 1)


class ArrayGraphXYandYX(unittest.TestCase):
    """For testing array_graph_xy"""
    # should test decreasing Xs and neither increasing nor decreasing errors
    def test_array_graph_xy(self):
        """Testing array_graph_xy"""
        XYs = (
            (1, 100), (1.5, 97.24), (2, 92.34), (2.5, 88.41), (3, 85.07), 
            (3.5, 80.42), (4, 75.39), (4.5, 66.52), (5, 57.80), (5.5, 47.95), 
            (6, 36.47), (6.5, 25.31), (7, 16.71), (7.5, 10.04), (8, 6.19), 
            (8.5, 3.35), (9, 2.10), (9.5, 1.01), (10, 0)
            )
        self.assertEqual(array_graph_xy(2, XYs), 92.34)
        self.assertAlmostEqual(array_graph_xy(7.4, XYs), 11.374)
        self.assertEqual(array_graph_xy(11, XYs), 0)
        self.assertEqual(array_graph_xy(1, XYs), 100)

    def test_array_graph_yx(self):
        """Testing array_graph_yx"""
        XYs = (
            (1, 100), (1.5, 97.24), (2, 92.34), (2.5, 88.41), (3, 85.07), 
            (3.5, 80.42), (4, 75.39), (4.5, 66.52), (5, 57.80), (5.5, 47.95), 
            (6, 36.47), (6.5, 25.31), (7, 16.71), (7.5, 10.04), (8, 6.19), 
            (8.5, 3.35), (9, 2.10), (9.5, 1.01), (10, 0)
            )
        self.assertEqual(array_graph_yx(92.34, XYs), 2)
        self.assertAlmostEqual(array_graph_yx(11.374, XYs), 7.4)
        self.assertEqual(array_graph_yx(0, XYs), 10)
        self.assertEqual(array_graph_yx(100, XYs), 1)

class AllAmounts(unittest.TestCase):
    def test_all_amounts(self):
        """Test the all_amounts() for a variety of variables"""

        with model(treatments=['As is', 'To be']) as m:
            Savings = stock(
                'Savings', lambda interest: interest, ('Interest',), 1000)
            Rate = variable(
                'Rate', PerTreatment({'As is': 0.05, 'To be': 0.06}))
            Interest = variable(
                'Interest', lambda savings, rate: savings * rate, 
                'Savings', 'Rate')
            PreviousInterest = previous('PreviousInterest', 'Interest', 0)
            AccumInterest = accum('AccumInterest', 
                lambda i: i, ('Interest',), 0)

        self.assertEqual(Savings.all(), {'As is': 1000, 'To be': 1000})
        self.assertEqual(Rate.all(), {'As is': 0.05, 'To be': 0.06})
        self.assertEqual(Interest.all(), {'As is': 50.0, 'To be': 60.0})
        self.assertEqual(PreviousInterest.all(), {'As is': 0, 'To be': 0})
        self.assertEqual(AccumInterest.all(), {'As is': 0, 'To be': 0})
        m.step()
        self.assertEqual(Savings.all(), {'As is': 1050, 'To be': 1060})
        self.assertEqual(Rate.all(), {'As is': 0.05, 'To be': 0.06})
        self.assertAlmostEqual(Interest.all()['As is'], 52.5)
        self.assertAlmostEqual(Interest.all()['To be'], 63.6)
        self.assertEqual(PreviousInterest.all(), {'As is': 50, 'To be': 60})
        self.assertAlmostEqual(AccumInterest.all()['As is'], 52.5)
        self.assertAlmostEqual(AccumInterest.all()['To be'], 63.6)
        m.step()
        self.assertAlmostEqual(Savings.all()['As is'], 1102.5)
        self.assertAlmostEqual(Savings.all()['To be'], 1123.6)
        self.assertEqual(Rate.all(), {'As is': 0.05, 'To be': 0.06})
        self.assertAlmostEqual(Interest.all()['As is'], 55.125)
        self.assertAlmostEqual(Interest.all()['To be'], 67.416)
        self.assertAlmostEqual(PreviousInterest.all()['As is'], 52.5)
        self.assertAlmostEqual(PreviousInterest.all()['To be'], 63.6)
        self.assertAlmostEqual(AccumInterest.all()['As is'], 107.625)
        self.assertAlmostEqual(AccumInterest.all()['To be'], 131.016)
        m.reset()
        self.assertEqual(Savings.all(), {'As is': 1000, 'To be': 1000})
        self.assertEqual(Rate.all(), {'As is': 0.05, 'To be': 0.06})
        self.assertEqual(Interest.all(), {'As is': 50.0, 'To be': 60.0})
        self.assertEqual(PreviousInterest.all(), {'As is': 0, 'To be': 0})
        self.assertEqual(AccumInterest.all(), {'As is': 0, 'To be': 0})
        m.step(2)
        self.assertAlmostEqual(Savings.all()['As is'], 1102.5)
        self.assertAlmostEqual(Savings.all()['To be'], 1123.6)
        self.assertEqual(Rate.all(), {'As is': 0.05, 'To be': 0.06})
        self.assertAlmostEqual(Interest.all()['As is'], 55.125)
        self.assertAlmostEqual(Interest.all()['To be'], 67.416)
        self.assertAlmostEqual(PreviousInterest.all()['As is'], 52.5)
        self.assertAlmostEqual(PreviousInterest.all()['To be'], 63.6)
        self.assertAlmostEqual(AccumInterest.all()['As is'], 107.625)
        self.assertAlmostEqual(AccumInterest.all()['To be'], 131.016)

class StrAndRepr(unittest.TestCase):
    def test_str(self):
        """Test str() for a variety of variables"""
        with model(treatments=['As is', 'To be']) as m:
            Savings = stock(
                'Savings', lambda interest: interest, ('Interest',), 1000)
            Rate = constant(
                'Rate', PerTreatment({'As is': 0.05, 'To be': 0.06}))
            Interest = variable(
                'Interest', lambda savings, rate: savings * rate, 
                'Savings', 'Rate')
            PreviousInterest = previous('PreviousInterest', 'Interest', 0)
            AccumInterest = accum('AccumInterest', 
                lambda i: i, ('Interest',), 0)

        self.assertEqual(str(Savings), "<Stock Savings>")
        self.assertEqual(str(Rate), "<Constant Rate>")
        self.assertEqual(str(Interest), "<Variable Interest>")
        self.assertEqual(str(PreviousInterest), "<Previous PreviousInterest>")
        self.assertEqual(str(AccumInterest), "<Accum AccumInterest>")

    def test_repr(self):
        """Test repr() for a variety of variables"""
        with model(treatments=['As is', 'To be']) as m:
            Savings = stock(
                'Savings', lambda interest: interest, ('Interest',), 1000)
            Rate = constant(
                'Rate', PerTreatment({'As is': 0.05, 'To be': 0.06}))
            Interest = variable(
                'Interest', lambda savings, rate: savings * rate, 
                'Savings', 'Rate')
            PreviousInterest = previous('PreviousInterest', 'Interest', 0)
            AccumInterest = accum('AccumInterest', 
                lambda i: i, ('Interest',), 0)

        self.assertEqual(repr(Savings), "stock('Savings')")
        self.assertEqual(repr(Rate), "constant('Rate')")
        self.assertEqual(repr(Interest), "variable('Interest')")
        self.assertEqual(repr(PreviousInterest), "previous('PreviousInterest')")
        self.assertEqual(repr(AccumInterest), "accum('AccumInterest')")

def bolded(string):
    """Return the string in bold."""
    return '\033[1m' + string + '\033[0m'

class ShowVariable(unittest.TestCase):
    """Test .show() on varaibles."""

    # adapted from https://bit.ly/2vXanhu
    @unittest.mock.patch('sys.stdout', new_callable=io.StringIO)
    def assert_show(self, variable, expected_output, expected_result, 
                    mock_stdout):
        """Test that var.show() matches expectations."""
        res = variable.show()
        self.assertEqual(mock_stdout.getvalue(), expected_output)
        self.assertEqual(res, expected_result)

    def test_show(self):
        """Test that show provides everything important about a variable"""

        with model(treatments=['As is', 'To be']) as m:
            Savings = stock(
                'Savings', 
                lambda interest: interest, 
                ('Interest',), 1000)
            Rate = variable(
                'Rate', PerTreatment({'As is': 0.05, 'To be': 0.06}))
            Interest = variable(
                'Interest', 
                lambda savings, rate: savings * rate, 
                'Savings', 'Rate')
            PreviousInterest = previous('PreviousInterest', 'Interest', 0)
            AccumInterest = accum('AccumInterest', 
                lambda i: i,
                ('Interest',), 0)
            NoFutureSavings = variable(
                'NoFutureSavings', lambda s: s, 'Savings'
            ).undefined_in('To be')

        self.assert_show(
            Savings, 
            bolded("Stock: Savings") + """

Amounts: {'As is': 1000, 'To be': 1000}

Initial definition: 1000
Initial depends on: []

Incremental definition: lambda interest: interest
Incremental depends on: ['Interest']
""",
            [Interest])

        self.assert_show(
            NoFutureSavings, 
            bolded("Variable: NoFutureSavings") + """

Amounts: {'As is': 1000}

Definition: 'NoFutureSavings', lambda s: s, 'Savings'
Depends on: ['Savings']
""",
            [Savings])

        self.assert_show(
            Rate, 
            bolded("Variable: Rate") + """

Amounts: {'As is': 0.05, 'To be': 0.06}

Definition: PerTreatment({"As is": 0.05, "To be": 0.06})
Depends on: []
""",
            [])

        self.assert_show(
            Interest,
            bolded("Variable: Interest") + """

Amounts: {'As is': 50.0, 'To be': 60.0}

Definition: lambda savings, rate: savings * rate
Depends on: ['Savings', 'Rate']
""",
            [Savings, Rate])

        self.assert_show(
            AccumInterest,
            bolded("Accum: AccumInterest") + """

Amounts: {'As is': 0, 'To be': 0}

Initial definition: 0
Initial depends on: []

Incremental definition: lambda i: i
Incremental depends on: ['Interest']
""",
            [Interest])

        self.assert_show(
            PreviousInterest,
            bolded("Previous: PreviousInterest") + """

Amounts: {'As is': 0, 'To be': 0}

Previous variable: Interest
""",
            [Interest])


class ValidateAndSetTest(unittest.TestCase):
    """Test Model.validate_and_set()."""
    def test_no_validator(self):
        """Test Model.validate_and_set() when no validator is defined."""
        with model() as m:
            InterestRate = constant('InterestRate', 0.04)
        self.assertEqual(
            m.validate_and_set('InterestRate', '', 0.05),
            {
                'success': True, 
                'variable': 'InterestRate',
                'treatment': '',
                'amount': 0.05
            })
        self.assertEqual(InterestRate[''], 0.05)

    def test_bad_variable(self):
        """Test Model.validate_and_set() with a bad variable."""
        with model() as m:
            constant('InterestRate', 0.04)
        self.assertEqual(
            m.validate_and_set('InterestRat', '', 0.05),
            {
                'success': False, 
                'variable': 'InterestRat',
                'treatment': '',
                'amount': 0.05,
                'error_code': 'UnknownVariable',
                'error_message': 'Variable InterestRat not known.',
            })

    def test_bad_treatment(self):
        """Test Model.validate_and_set() with a bad treatment."""
        with model(treatments=['foo']) as m:
            constant('InterestRate', 0.04)
        self.assertEqual(
            m.validate_and_set('InterestRate', 'bar', 0.05),
            {
                'success': False, 
                'variable': 'InterestRate',
                'treatment': 'bar',
                'amount': 0.05,
                'error_code': 'UnknownTreatment',
                'error_message': 'Treatment bar not known.',
            })

    def test_one_validator(self):
        """Test Model.validate_and_set() with a validator defined."""
        with model(treatments=['current', 'possible']) as m:
            constant('InterestRate', 0.04).constraint(
                lambda amt: amt > 0,
                "TooSmall",
                lambda amt, nm: f'{nm} is {amt}; must be greater than 0.',
                0.01)
        self.assertEqual(
            m.validate_and_set('InterestRate', '__all__', 0.05),
            {
                'success': True,
                'variable': 'InterestRate',
                'treatment': '__all__',
                'amount': 0.05
            })
        self.assertEqual(
            m.validate_and_set('InterestRate', 'possible', 0.0),
            {
                'success': False, 
                'variable': 'InterestRate',
                'treatment': 'possible',
                'amount': 0,
                'error_code': 'TooSmall',
                'error_message': 'InterestRate is 0.0; must be greater than 0.',
                'suggested_amount': 0.01
            })

    def test_two_validators(self):
        """Test Model.validate_and_set() with two validators defined."""
        with model(treatments=['current', 'possible']) as m:
            constant('InterestRate', 0.04).constraint(
                lambda amt: amt > 0,
                "TooSmall",
                lambda amt, nm: f'{nm} is {amt}; must be greater than 0.',
                0.01
            ).constraint(
                lambda amt: amt <= 1.0,
                "TooLarge",
                lambda amt, nm: f'{nm} is {amt}; should be less than 100%.'
            )
        self.assertEqual(
            m.validate_and_set('InterestRate', '__all__', 0.05),
            {
                'success': True,
                'variable': 'InterestRate',
                'treatment': '__all__',
                'amount': 0.05
            })
        self.assertEqual(
            m.validate_and_set('InterestRate', 'possible', 0.0),
            {
                'success': False, 
                'variable': 'InterestRate',
                'treatment': 'possible',
                'amount': 0,
                'error_code': 'TooSmall',
                'error_message': 'InterestRate is 0.0; must be greater than 0.',
                'suggested_amount': 0.01
            })
        self.assertEqual(
            m.validate_and_set('InterestRate', 'current', 2.5),
            {
                'success': False, 
                'variable': 'InterestRate',
                'treatment': 'current',
                'amount': 2.5,
                'error_code': 'TooLarge',
                'error_message': 'InterestRate is 2.5; should be less than 100%.'
            })

    def test_alternative_validator(self):
        """Test something that implements the validate()."""
        class _FakeValidator:
            @classmethod
            def validate(cls, amount, name):
                if amount > 0:
                    return True, None, None, None
                else:
                    return False, "Bad", "Really bad", 1 

        with model(treatments=['current', 'possible']) as m:
            constant('InterestRate', 0.04).constraint(_FakeValidator)

        self.assertEqual(
            m.validate_and_set('InterestRate', '__all__', 0.5),
            {
                'success': True,
                'variable': 'InterestRate',
                'treatment': '__all__',
                'amount': 0.5
            })

        self.assertEqual(
            m.validate_and_set('InterestRate', '__all__', -99),
            {
                'success': False,
                'variable': 'InterestRate',
                'treatment': '__all__',
                'amount': -99,
                'error_code': 'Bad',
                'error_message': 'Really bad',
                'suggested_amount': 1
            })

    def test_multiple_treatments(self):
        """Test Model.validate_and_set() when with multiple treatments."""
        with model(treatments=['current', 'imagined']) as m:
            InterestRate = constant('InterestRate', 
                PerTreatment({'current': 0.04, 'imagined': 0.04}))
        self.assertEqual(
            m.validate_and_set('InterestRate', 'current', 0.05),
            {
                'success': True, 
                'variable': 'InterestRate',
                'treatment': 'current',
                'amount': 0.05
            })
        self.assertEqual(InterestRate['current'], 0.05)
        self.assertEqual(InterestRate['imagined'], 0.04)

    def test_reset_model(self):
        """Test change value then, then Model.reset())"""
        with model() as m:
            InterestRate = constant('InterestRate', 0.04)

        self.assertEqual(InterestRate[''], 0.04)
        m.validate_and_set('InterestRate', '', 0.05)
        self.assertEqual(InterestRate[''], 0.05) 
        m.reset(reset_external_vars=False)
        self.assertEqual(InterestRate[''], 0.05)  
        m.reset()
        self.assertEqual(InterestRate[''], 0.04)

           

class ValidateAndSetAttributeTest(unittest.TestCase):
    """Test setting and validating attributes of variables."""
    def test_attribute_without_validator(self):
        """Test setting an atribute without a validator."""
        class _Size:
            def __init__(self, length, width, height):
                self.length = length
                self.width = width
                self.height = height

        with model() as m:
            Size = constant('Size', _Size(18, 16, 14))
        self.assertEqual(Size[''].length, 18)
        self.assertEqual(
            m.validate_and_set('Size', '', 17, excerpt='.length'),
            {
                'success': True,
                'variable': 'Size',
                'excerpt': '.length',
                'treatment': '',
                'amount': 17
            })
        self.assertEqual(Size[''].length, 17)

    def test_attribute_without_validator_multiple_treatments(self):
        """Test setting attribute without validaotr in multiple treatments."""
        class _Size:
            def __init__(self, length, width, height):
                self.length = length
                self.width = width
                self.height = height

        with model(treatments=['current', 'imagined']) as m:
            Size = constant('Size', 
                PerTreatment(
                    {'current': _Size(18, 16, 14),'imagined': _Size(18, 16, 14)}
                ))
        self.assertEqual(Size['current'].length, 18)
        self.assertEqual(Size['imagined'].length, 18)
        self.assertEqual(
            m.validate_and_set('Size', 'current', 17, excerpt='.length'),
            {
                'success': True,
                'variable': 'Size',
                'excerpt': '.length',
                'treatment': 'current',
                'amount': 17
            })
        self.assertEqual(Size['current'].length, 17)
        self.assertEqual(Size['imagined'].length, 18)
        with self.assertRaisesRegex(MinnetonkaError,
                'validate_and_set for Size on multiple treatments'):
            m.validate_and_set('Size', '__all__', 20, excerpt='.length')

        with model(treatments=['current', 'imagined']) as m:
            Size = constant('Size', _Size(18, 16, 14))

        self.assertEqual(
            m.validate_and_set('Size', '__all__', 17, excerpt='.length'),
            {
                'success': True,
                'variable': 'Size',
                'excerpt': '.length',
                'treatment': '__all__',
                'amount': 17
            })

        self.assertEqual(Size['current'].length, 17)
        self.assertEqual(Size['imagined'].length, 17)


    def test_valid_attribute(self):
        """Test setting an atribute with a validator."""
        class _Size:
            def __init__(self, length, width, height):
                self.length = length
                self.width = width
                self.height = height

            def validate(self, attr, amount):
                return True, '', '', None

        with model() as m:
            Size = constant('Size', _Size(18, 16, 14))
        self.assertEqual(Size[''].length, 18)
        self.assertEqual(
            m.validate_and_set('Size', '', 17, excerpt='.length'),
            {
                'success': True,
                'variable': 'Size',
                'excerpt': '.length',
                'treatment': '',
                'amount': 17
            })
        self.assertEqual(Size[''].length, 17)

    def test_invalid_attribute(self):
        """Test setting an attribute that does not pass validation."""
        class _Size:
            def __init__(self, length, width, height):
                self.length = length
                self.width = width
                self.height = height

            def validate(self, attr, amount):
                return False, 'Bad', 'Really quite bad', None

        with model() as m:
            Size = constant('Size', _Size(18, 16, 14))
        self.assertEqual(Size[''].length, 18)
        self.assertEqual(
            m.validate_and_set('Size', '', 17, excerpt='.length'),
            {
                'success': False,
                'variable': 'Size',
                'excerpt': '.length',
                'treatment': '',
                'error_code': 'Bad',
                'error_message': 'Really quite bad',
                'suggested_amount': None,
                'amount': 17

            })
        self.assertEqual(Size[''].length, 18)

    def test_unsettable_property(self):
        """Test setting a property that cannot be set."""
        class _Size:
            def __init__(self, length, width, height):
                self._length = length
                self._width = width
                self._height = height

            @property
            def length(self):
                return self._length

        with model() as m:
            Size = constant('Size', _Size(18, 16, 14))
        self.assertEqual(Size[''].length, 18)
        self.assertEqual(
            m.validate_and_set('Size', '', 17, excerpt='.length'),
            {
                'success': False,
                'variable': 'Size',
                'excerpt': '.length',
                'treatment': '',
                'error_code': 'Unsettable',
                'error_message': "Error can't set attribute raised when setting amount of _Size to 17", 
                'amount': 17

            })
        self.assertEqual(Size[''].length, 18)

    def test_attribute_chain_without_validator(self):
        """Test setting a chain of atributes without a validator."""
        class _Size:
            def __init__(self, length, width, height):
                self.length = length
                self.width = width
                self.height = height

        class _Measure:
            def __init__(self, metric, customary):
                self.metric = metric
                self.customary = customary

        class _Interval:
            def __init__(self, begin, end):
                self.begin = begin 
                self.end = end 

        with model() as m:
            Size = constant('Size', 
                _Size(_Measure(18, _Interval(1.0, 2.0)), 16, 14))
        self.assertEqual(Size[''].length.customary.begin, 1.0)
        self.assertEqual(
            m.validate_and_set(
                'Size', '', 1.3, excerpt='.length.customary.begin'),
            {
                'success': True,
                'variable': 'Size',
                'excerpt': '.length.customary.begin',
                'treatment': '',
                'amount': 1.3
            })
        self.assertEqual(Size[''].length.customary.begin, 1.3)

    def test_reset_model(self):
        """Test change attribute, then Model.reset())"""
        class _Size:
            def __init__(self, length, width, height):
                self.length = length
                self.width = width
                self.height = height

        with model() as m:
            Size = constant('Size', lambda: _Size(18, 16, 14))

        self.assertEqual(Size[''].length, 18)
        m.validate_and_set('Size', '', 19, excerpt='.length')
        self.assertEqual(Size[''].length, 19)
        m.recalculate()
        self.assertEqual(Size[''].length, 19)
        m.reset(reset_external_vars=False)
        self.assertEqual(Size[''].length, 19)
        m.reset()
        self.assertEqual(Size[''].length, 18)

class ValidateAllTest(unittest.TestCase):
    """Test validate_all()."""
    def test_nothing_to_validate(self):
        """Test validate_all() when no constraints are defined."""
        with model() as m:
            constant('X7Allowed', False)
            constant('X5Allowed', False)
            constant('X4Allowed', False)

        self.assertEqual(m.validate_all(), {'success': True})

    def test_simple(self):
        """Test validate_all() with a simple constraint."""
        with model() as m:
            constant('X7Allowed', False)
            constant('X5Allowed', False)
            X4 = constant('X4Allowed', False)

            constraint(
                ['X7Allowed', 'X5Allowed', 'X4Allowed'],
                lambda *machines: any(machines),
                "AtLeastOneTruthy",
                lambda names, amounts, trt: 
                    f'All machines are disabled: {", ".join(names)}')

        self.assertEqual(
            m.validate_all(),
            {
                'success': False,
                'errors': [
                    {
                        'error_code': 'AtLeastOneTruthy',
                        'inconsistent_variables': [
                            'X7Allowed', 'X5Allowed', 'X4Allowed'],
                        'error_message': 'All machines are disabled: X7Allowed, X5Allowed, X4Allowed',
                        'treatment': ''
                    }

                ]
            })

        X4[''] = True
        self.assertEqual(m.validate_all(), {'success': True})

    def test_one_treatment(self):
        """Test validate_all() that fails in one treatment only."""
        with model(treatments=['current', 'future']) as m:
            constant('X7Allowed', False)
            constant('X5Allowed', False)
            X4 = constant(
                'X4Allowed', PerTreatment({'current': True, 'future': True}))

            constraint(
                ['X7Allowed', 'X5Allowed', 'X4Allowed'],
                lambda *machines: any(machines),
                "AtLeastOneTruthy",
                lambda names, amounts, trt: 
                    f'All machines are disabled: {", ".join(names)}')


        self.assertEqual(m.validate_all(), {'success': True})
        X4['future'] = False
        self.assertEqual(
            m.validate_all(),
            {
                'success': False,
                'errors': [
                    {
                        'error_code': 'AtLeastOneTruthy',
                        'inconsistent_variables': [
                            'X7Allowed', 'X5Allowed', 'X4Allowed'],
                        'error_message': 'All machines are disabled: X7Allowed, X5Allowed, X4Allowed',
                        'treatment': 'future'
                    }

                ]
            })


    def test_two_constraints(self):
        """Test validate_all() with two different constraints."""
        with model() as m:
            constant('X7Allowed', False)
            X5 = constant('X5Allowed', False)
            constant('X4Allowed', False)

            constraint(
                ['X7Allowed', 'X5Allowed', 'X4Allowed'],
                lambda *machines: any(machines),
                "AtLeastOneTruthy",
                lambda names, amounts, trt: 
                    f'All machines are disabled: {", ".join(names)}')

            constant('Small', 0.4)
            constant('Medium', 0.5)
            Large = constant('Large', 0.05)

            constraint(
                ['Small', 'Medium', 'Large'],
                lambda *sizes: sum(sizes) == 1.0,
                'InvalidDistribution',
                lambda names, amounts, treatment: 
                    'Distribution of {} sums to {}, not 1.0, in {}'.format(
                        ", ".join(names), round(sum(amounts), 3), treatment))

        vresult = m.validate_all()
        self.assertEqual(vresult['success'], False)
        self.assertEqual(len(vresult['errors']), 2)
        self.assertIn(
            {
                'error_code': 'AtLeastOneTruthy',
                'inconsistent_variables': [
                    'X7Allowed', 'X5Allowed', 'X4Allowed'],
                'error_message': 'All machines are disabled: X7Allowed, X5Allowed, X4Allowed',
                'treatment': ''
            },
            vresult['errors'])
        self.assertIn(
            {
                'error_code': 'InvalidDistribution',
                'inconsistent_variables': ['Small', 'Medium', 'Large'],
                'error_message': 'Distribution of Small, Medium, Large sums to 0.95, not 1.0, in ',
                'treatment': ''
            },
            vresult['errors'])

        X5[''] = True
        Large[''] = 0.1
        self.assertEqual(m.validate_all(), {'success': True})

    def test_constraint_raising_error(self):
        """Test validate_all() with a broken constraint that raises error."""
        with model() as m:
            constant('X7Allowed', False)
            constraint(
                ['X7Allowed'],
                lambda machine: 1 / 0,
                "Whatever",
                lambda names, amounts, trt: 'whatever')

        vresult = m.validate_all()
        self.assertEqual(
            m.validate_all(),
            {
                'success': False,
                'errors': [
                    {
                        'error_code': 'Whatever',
                        'inconsistent_variables': ['X7Allowed'],
                        'error_message': 
                            'Constraint raised exception division by zero',
                        'treatment': ''
                    }

                ]
            })

        with model() as m:
            constant('X7Allowed', False)
            constraint(
                ['X7Allowed'],
                lambda x: False,
                "Whatever",
                lambda names, amounts, trt: 1 / 0)

        self.assertEqual(
            m.validate_all(),
            {
                'success': False,
                'errors': [
                    {
                        'error_code': 'Whatever',
                        'inconsistent_variables': ['X7Allowed'],
                        'error_message': 
                            'Constraint raised exception division by zero',
                        'treatment': ''
                    }

                ]
            })
 

class DerivedTreatmentTest(unittest.TestCase):
    """Test derived treatments."""
    def test_simple(self):
        """Test simple application of derived treatment."""
        with model(treatments=['current', 'possible'], 
                   derived_treatments={
                        'at-risk': AmountBetter('possible', 'current')}
        ) as m:
            Revenue = constant('Revenue', 
                PerTreatment({'current': 20, 'possible': 25})
            ).derived()
            Cost = constant('Cost',
                PerTreatment({'current': 19, 'possible': 18})
            ).derived(scored_as='golf')
            Earnings = variable('Earnings',
                lambda r, c: r-c,
                'Revenue', 'Cost'
            ).derived()

        self.assertEqual(Revenue['at-risk'], 5)
        self.assertEqual(Cost['at-risk'], 1)
        self.assertEqual(Earnings['at-risk'], 6)
        self.assertEqual(Earnings['current'], 1)

    def test_derived_treatment_name_dupes_treatment(self):
        """Test derived treatment with name that is already treatemnt."""
        with self.assertRaisesRegex(MinnetonkaError, 
                'Derived treatment possible is also a treatment'):
            model(treatments=['current', 'possible'], 
                  derived_treatments={
                    'possible': AmountBetter('possible', 'current')})

    def test_setting_derived_treatment(self):
        with model(treatments=['current', 'possible'], 
                   derived_treatments={
                        'at-risk': AmountBetter('possible', 'current')}
        ) as m:
            Revenue = constant('Revenue', 
                PerTreatment({'current': 20, 'possible': 25}))

        with self.assertRaisesRegex(MinnetonkaError, 
                'Cannot set Revenue in derived treatment at-risk.'):
            Revenue['at-risk'] = 2.7

    def test_mix(self):
        """Test a variable that is scored as a mix."""
        with model(treatments=['current', 'possible'], 
                   derived_treatments={
                        'at-risk': AmountBetter('possible', 'current')}
        ) as m:
            Revenue = constant('Revenue', 
                PerTreatment({'current': 20, 'possible': 25})
            ).derived()
            Cost = constant('Cost',
                PerTreatment({'current': 19, 'possible': 18})
            ).derived(scored_as='golf')
            Summary = variable('Summary',
                lambda r, c: {'revenue':r, 'cost':c},
                'Revenue', 'Cost'
            ).derived(scored_as='combo')

        self.assertEqual(Revenue['at-risk'], 5)
        self.assertEqual(Cost['at-risk'], 1)
        self.assertEqual(Summary['at-risk'], {'revenue': 5, 'cost': 1})

    def test_attempt_to_access_derived_of_nonderived(self):
        """Attempt to access derived treatment of non-derived variable."""
        with model(treatments=['current', 'possible'], 
                   derived_treatments={
                        'at-risk': AmountBetter('possible', 'current')}
        ) as m:
            Revenue = constant('Revenue', 
                PerTreatment({'current': 20, 'possible': 25})
            ) 
            Cost = constant('Cost',
                PerTreatment({'current': 19, 'possible': 18})
            ).derived(scored_as='golf')
            Earnings = variable('Earnings',
                lambda r, c: r-c,
                'Revenue', 'Cost'
            ).derived()

        with self.assertRaisesRegex(MinnetonkaError, 
                'Unknown treatment at-risk for variable Revenue'):
            Revenue['at-risk']

    def test_is_derived(self):
        """Test the function is_derived.""" 
        with model(treatments=['current', 'possible'], 
                   derived_treatments={
                        'at-risk': AmountBetter('possible', 'current')}
        ) as m:
            Revenue = constant('Revenue', 
                PerTreatment({'current': 20, 'possible': 25})
            ) 
            Cost = constant('Cost',
                PerTreatment({'current': 19, 'possible': 18})
            ).derived(scored_as='golf')
            Earnings = variable('Earnings',
                lambda r, c: r-c,
                'Revenue', 'Cost'
            ).derived()

        self.assertTrue(Cost.is_derived())
        self.assertFalse(Revenue.is_derived())


class ReplayTest(unittest.TestCase):
    """Test replaying a model."""
    def test_simple_replay(self):
        """Test Model.recording() and Model.replay() of simple variables."""
        def create_model():
            with model(treatments=['then', 'now']) as m:
                constant('Foo', PerTreatment({'then': 9, 'now': 10}))
                constant('Bar', 2)
                variable('Baz', lambda a, b: a+b, 'Foo', 'Bar')
                constant('Quz', 99)
            return m

        m = create_model() 
        m.validate_and_set('Foo', 'then', 99)
        m.validate_and_set('Bar', '__all__', 3)
        m.validate_and_set('Baz', 'now', 15) 
        m.validate_and_set('Foo', 'then', 11)
        m.recalculate()
        self.assertEqual(m['Foo']['then'], 11)
        self.assertEqual(m['Foo']['now'], 10)
        self.assertEqual(m['Bar']['then'], 3)
        self.assertEqual(m['Bar']['now'], 3)
        self.assertEqual(m['Baz']['then'], 14)
        self.assertEqual(m['Baz']['now'], 15) 

        recording = m.recording()
        m2 = create_model()
        m2.replay(recording) 
        self.assertEqual(m2['Foo']['then'], 11)
        self.assertEqual(m2['Foo']['now'], 10)
        self.assertEqual(m2['Bar']['then'], 3)
        self.assertEqual(m2['Bar']['now'], 3)
        self.assertEqual(m2['Baz']['then'], 14)
        self.assertEqual(m2['Baz']['now'], 15) 

    def test_replay_excerpts(self):
        """Test Model.recording() and Model.replay, including excerpts."""
        class _Size:
            def __init__(self, length, width, height):
                self.length = length
                self.width = width
                self.height = height

        class _Measure:
            def __init__(self, metric, customary):
                self.metric = metric
                self.customary = customary

        class _Interval:
            def __init__(self, begin, end):
                self.begin = begin 
                self.end = end 

        def create_model():
            with model() as m:
                constant('Size', 
                    _Size(_Measure(18, _Interval(1.0, 2.0)), 16, 14))
            return m

        m = create_model()
        m.validate_and_set('Size', '', 99, excerpt='.width')
        m.validate_and_set('Size', '', 19, excerpt='.length.metric')
        m.validate_and_set('Size', '', 15, excerpt='.length.customary.begin')
        m.validate_and_set('Size', '', 17, excerpt='.length.metric')
        self.assertEqual(m['Size'][''].width, 99)
        self.assertEqual(m['Size'][''].length.metric, 17)
        self.assertEqual(m['Size'][''].length.customary.begin, 15)
        self.assertEqual(m['Size'][''].height, 14)
        
        m2 = create_model()
        m2.replay(m.recording())
        self.assertEqual(m['Size'][''].width, 99)
        self.assertEqual(m['Size'][''].length.metric, 17)
        self.assertEqual(m['Size'][''].length.customary.begin, 15)
        self.assertEqual(m['Size'][''].height, 14)

    def test_complex_amount(self):
        """Test Model.recording() on amount that cannot be recorded."""
        class _Size:
            def __init__(self, length, width, height):
                self.length = length
                self.width = width
                self.height = height

        class _Measure:
            def __init__(self, metric, customary):
                self.metric = metric
                self.customary = customary

        class _Interval:
            def __init__(self, begin, end):
                self.begin = begin 
                self.end = end 

        def create_model():
            with model() as m:
                constant('Size', 
                    _Size(_Measure(18, _Interval(1.0, 2.0)), 16, 14))
            return m

        m = create_model()
        with self.assertRaisesRegex(MinnetonkaError,
                'Cannot save amount for later playback'):
            m.validate_and_set('Size', '', _Measure(10, 12), excerpt='.width')
        
        m.validate_and_set(
            'Size', '', _Measure(10, 12), excerpt='.width', record=False)
        m.validate_and_set('Size', '', 19, excerpt='.length.metric')

        m2 = create_model()
        m2.replay(m.recording())

        self.assertEqual(m2['Size'][''].length.metric, 19)
        self.assertEqual(m2['Size'][''].width, 16)

    def test_included_step(self):
        """Test Model.recording with a step or two involved."""
        def create_model():
            with model(end_time=10) as m:
                variable('X', 1)
                variable('Y', 22)
                S = stock('S',
                    """Start at 22 and increase by 1""",
                     lambda x: x, ('X',), lambda x: x, ('Y',))
            return m 

        m = create_model()
        self.assertEqual(m['S'][''], 22)
        m.step()
        self.assertEqual(m['S'][''], 23) 
        recording2 = m.recording()

        m.validate_and_set('X', '', 2)
        m.recalculate()
        m.step(2)
        recording3 = m.recording()

        m.step(to_end=True)
        recording4 = m.recording()

        m2 = create_model()
        m2.replay(recording2)
        self.assertEqual(m2['S'][''], 23)

        m3 = create_model()
        m3.replay(recording3)
        self.assertEqual(m3['S'][''], 27)

        m4 = create_model()
        m4.replay(recording4)
        self.assertEqual(m4['S'][''], 41)

    def test_reset(self):
        """Test model recording with a reset or two."""
        def create_model():
            with model() as m:
                variable('X', 1) 
            return m 

        m = create_model() 
        m.validate_and_set('X', '', 2)
        m.reset()
        recording2 = m.recording()

        m.validate_and_set('X', '', 2)
        m.reset(reset_external_vars=False)
        recording3 = m.recording()

        m2 = create_model()
        m2.replay(recording2)
        self.assertEqual(m2['X'][''], 1)

        m3 = create_model()
        m3.replay(recording3)
        self.assertEqual(m3['X'][''], 2)

    def test_multiple_resets(self):
        """Test that multiple resets and multiple steps are only done once."""
        def create_model():
            with model(end_time=10) as m:
                variable('X', 1)
                variable('Y', 22)
                S = stock('S',
                    """Start at 22 and increase by 1""",
                     lambda x: x, ('X',), lambda x: x, ('Y',))
            return m 


        m = create_model() 
        m.validate_and_set('X', '', 2)
        m.reset(reset_external_vars=False)
        m.recalculate()
        m.reset(reset_external_vars=False)
        m.step(3)
        m.reset(reset_external_vars=False)
        m.reset(reset_external_vars=False)
        recording = m.recording()

        m2 = create_model()
        m2.replay(recording)
        self.assertEqual(m2['X'][''], 2)
        self.assertEqual(
            m2._user_actions.thaw_recording(recording),
            # just two actions
            [
                {
                    'type': 'validate_and_set',
                    'treatment_name': '',
                    'excerpt': '',
                    'amount': 2,
                    'variable_name': 'X'
                },
                {'type': 'reset', 'reset_external_vars': False}
            ])

    def test_ignore_step(self):
        """Test replay while ignoring steps."""
        def create_model():
            with model(end_time=10) as m:
                variable('X', 1)
                variable('Y', 22)
                S = stock('S',
                    """Start at 22 and increase by 1""",
                     lambda x: x, ('X',), lambda x: x, ('Y',))
            return m 

        m = create_model()
        self.assertEqual(m['S'][''], 22)
        m.step()
        self.assertEqual(m['S'][''], 23) 
        recording2 = m.recording()

        m.validate_and_set('X', '', 2)
        m.recalculate()
        m.step(2)
        recording3 = m.recording()

        m.step(to_end=True)
        recording4 = m.recording()

        m2 = create_model()
        m2.replay(recording2, ignore_step=True)
        self.assertEqual(m2['S'][''], 22)

        m3 = create_model()
        m3.replay(recording3, ignore_step=True)
        self.assertEqual(m3['S'][''], 22)

        m4 = create_model()
        m4.replay(recording4, ignore_step=True)
        self.assertEqual(m4['S'][''], 22)


class CrossTreatmentTest(unittest.TestCase):
    """Test a cross."""
    def test_cross_constant(self):
        with model(treatments=['As is', 'To be']) as m: 
            Bar = constant('Bar', PerTreatment({'As is': 1, 'To be': 2}))
            Foo = cross('Foo', 'Bar', 'As is')

        self.assertEqual(Foo['As is'], 1)
        self.assertEqual(Foo['To be'], 1)

    def test_cross_variable(self):
        with model(treatments=['As is', 'To be']) as m: 
            Bar = constant('Bar', PerTreatment({'As is': 1, 'To be': 2}))
            Baz = variable('Baz', lambda x: x+2, 'Bar')
            Foo = cross('Foo', 'Baz', 'As is')

        self.assertEqual(Foo['As is'], 3)
        self.assertEqual(Foo['To be'], 3)

    def test_cross_model_variable(self):
        with model(treatments=['As is', 'To be']) as m: 
            S = stock('S', PerTreatment({'As is': 1, 'To be': 2}))
            Foo = cross('Foo', 'S', 'As is')

        m.step()
        self.assertEqual(Foo['As is'], 1)
        self.assertEqual(Foo['To be'], 1)
        m.step()
        self.assertEqual(Foo['As is'], 2)
        self.assertEqual(Foo['To be'], 2)

class UndefinedInTest(unittest.TestCase):
    """Test defining variable undefined in some treatments."""
    def test_undefined_value(self):
        """Test the value of something undefined."""
        with model(treatments=['conjecture', 'current', 'possible', 'design']):
            Foo = constant('Foo', 12 ).undefined_in('conjecture')
            Bar = variable('Bar',
                lambda x: x + 1,
                'Foo').undefined_in('conjecture', 'current')
        self.assertEqual(Foo['current'], 12)
        self.assertEqual(Foo['conjecture'], None)
        self.assertEqual(Bar['current'], None)
        self.assertEqual(Bar['conjecture'], None)
        self.assertEqual(Bar['possible'], 13)

    def test_undefined_stock(self):
        """Test a stock that is undefined for some treatments.""" 
        with model(treatments=['conjecture', 'current', 'possible', 'design']
               ) as m:
            variable('X', 1)
            variable('Y', 
                PerTreatment({'conjecture': 22, 'current': 22, 'possible': 22})
            ).undefined_in('design')
            S = stock('S',
                """Start at 22 and increase by 1""",
                 lambda x: x, ('X',), lambda x: x, ('Y',)
            ).undefined_in('design')
            P = variable('P', lambda x: x, 'S'
            ).undefined_in('possible', 'design')

        self.assertEqual(S['current'], 22)
        self.assertEqual(S['design'], None)
        self.assertEqual(P['conjecture'], 22)
        self.assertEqual(P['design'], None)
        self.assertEqual(P['possible'], None)
        m.step()        
        self.assertEqual(S['current'], 23)
        self.assertEqual(S['design'], None)
        self.assertEqual(P['conjecture'], 23)
        self.assertEqual(P['design'], None)
        self.assertEqual(P['possible'], None)
        m.step()
        self.assertEqual(S['current'], 24)
        self.assertEqual(S['design'], None)
        self.assertEqual(P['conjecture'], 24)
        self.assertEqual(P['design'], None)
        self.assertEqual(P['possible'], None)
        m.reset()
        self.assertEqual(S['current'], 22)
        self.assertEqual(S['design'], None)
        self.assertEqual(P['conjecture'], 22)
        self.assertEqual(P['design'], None)
        self.assertEqual(P['possible'], None)
        m.step()        
        self.assertEqual(S['current'], 23)
        self.assertEqual(S['design'], None)
        self.assertEqual(P['conjecture'], 23)
        self.assertEqual(P['design'], None)
        self.assertEqual(P['possible'], None)

    def test_undefined_accum(self):
        """Test an accum that is undefined for some treatments.""" 
        with model(treatments=['conjecture', 'current', 'possible', 'design']
               ) as m:
            variable('X', 1)
            variable('Y', 22)
            A = stock('A',
                """Start at 23 and increase by 1""",
                 lambda x: x, ('X',), lambda x: x, ('Y',)
            ).undefined_in('design')
            P = variable('P', lambda x: x, 'A'
            ).undefined_in('possible', 'design')

        self.assertEqual(A['current'], 22)
        self.assertEqual(A['design'], None)
        self.assertEqual(P['conjecture'], 22)
        self.assertEqual(P['design'], None)
        self.assertEqual(P['possible'], None)
        m.step()        
        self.assertEqual(A['current'], 23)
        self.assertEqual(A['design'], None)
        self.assertEqual(P['conjecture'], 23)
        self.assertEqual(P['design'], None)
        self.assertEqual(P['possible'], None)
        m.step()
        self.assertEqual(A['current'], 24)
        self.assertEqual(A['design'], None)
        self.assertEqual(P['conjecture'], 24)
        self.assertEqual(P['design'], None)
        self.assertEqual(P['possible'], None)
        m.reset()
        self.assertEqual(A['current'], 22)
        self.assertEqual(A['design'], None)
        self.assertEqual(P['conjecture'], 22)
        self.assertEqual(P['design'], None)
        self.assertEqual(P['possible'], None)
        m.step()        
        self.assertEqual(A['current'], 23)
        self.assertEqual(A['design'], None)
        self.assertEqual(P['conjecture'], 23)
        self.assertEqual(P['design'], None)
        self.assertEqual(P['possible'], None)

    def test_partial_per_treatment(self):
        """Test PerTreatment that is not defined for undefined treatments."""
        with model(treatments=['conjecture', 'current', 'possible', 'design']):
            Foo = constant('Foo', 12 ).undefined_in('conjecture')
            Bar = variable('Bar',
                PerTreatment({
                    'current': lambda x: x+1,
                    'possible': lambda x: x+2
                    }),
                'Foo'
            ).undefined_in('conjecture', 'design') 
        self.assertEqual(Bar['current'], 13)
        self.assertEqual(Bar['conjecture'], None)
        self.assertEqual(Bar['possible'], 14)

    def test_recalculate(self):
        """Test recalculating with undefined."""
        with model(treatments=['conjecture', 'current', 'possible', 'design']
                ) as m:
            Foo = constant('Foo', 12 ).undefined_in('conjecture')
            Bar = variable('Bar',
                lambda x: x + 1,
                'Foo').undefined_in('conjecture', 'current')
        self.assertEqual(Foo['current'], 12)
        self.assertEqual(Foo['conjecture'], None)
        self.assertEqual(Bar['current'], None)
        self.assertEqual(Bar['conjecture'], None)
        self.assertEqual(Bar['possible'], 13)
        m.step()
        Foo['__all__'] = 19
        m.recalculate()
        self.assertEqual(Foo['current'], 19)
        self.assertEqual(Foo['conjecture'], None)
        self.assertEqual(Bar['current'], None)
        self.assertEqual(Bar['conjecture'], None)
        self.assertEqual(Bar['possible'], 20)

    def test_previous(self):
        """Test a previous that is undefined for some treatments.""" 
        with model(treatments=['conjecture', 'current', 'possible', 'design']
               ) as m:
            variable('X', 1)
            variable('Y', 22).undefined_in('design')
            S = stock('S',
                """Start at 22 and increase by 1""",
                 lambda x: x, ('X',), lambda x: x, ('Y',)
            ).undefined_in('design')
            P = previous('P', 'S').undefined_in('possible', 'design')

        self.assertEqual(S['current'], 22)
        self.assertEqual(S['design'], None)
        self.assertEqual(P['conjecture'], 22)
        self.assertEqual(P['design'], None)
        self.assertEqual(P['possible'], None)
        m.step()        
        self.assertEqual(S['current'], 23)
        self.assertEqual(S['design'], None)
        self.assertEqual(P['conjecture'], 22)
        self.assertEqual(P['design'], None)
        self.assertEqual(P['possible'], None)
        m.step()        
        self.assertEqual(S['current'], 24)
        self.assertEqual(S['design'], None)
        self.assertEqual(P['conjecture'], 23)
        self.assertEqual(P['design'], None)
        self.assertEqual(P['possible'], None)

    def test_all_amounts(self):
        """Test all amounts with undefined."""
        with model(treatments=['conjecture', 'current', 'possible', 'design']
               ) as m:
            X = variable('X', 1)
            Y = variable('Y', 22).undefined_in('design')
            S = stock('S',
                """Start at 22 and increase by 1""",
                 lambda x: x, ('X',), lambda x: x, ('Y',)
            ).undefined_in('design', 'possible')

        self.assertEqual(
            X.all(), 
            {'conjecture': 1, 'current': 1, 'possible': 1, 'design': 1})
        self.assertEqual(
            Y.all(),
            {'conjecture': 22, 'current': 22, 'possible': 22})
        self.assertEqual(S.all(), {'conjecture': 22, 'current': 22} )
        m.step()
        self.assertEqual(
            X.all(), 
            {'conjecture': 1, 'current': 1, 'possible': 1, 'design': 1})
        self.assertEqual(
            Y.all(),
            {'conjecture': 22, 'current': 22, 'possible': 22})
        self.assertEqual(S.all(), {'conjecture': 23, 'current': 23} )

    def test_dispatch_function(self):
        """Test dispatch function with undefined."""
        with model(treatments=['conjecture', 'current', 'possible', 'design']
               ) as m:
            X = variable('X', 1).undefined_in('conjecture', 'current')
            Y = variable('Y', 22).undefined_in('possible', 'design')
            XorY = variable('XorY',
                PerTreatment({
                    'conjecture': lambda _, y: y,
                    'current': lambda _, y: y,
                    'possible': lambda x, _: x,
                    'design': lambda x, _: x
                    }),
                'X',
                'Y')
        self.assertEqual(
            XorY.all(),
            {'conjecture': 22, 'current': 22, 'possible': 1, 'design': 1})

    def test_history(self):
        """Test history with some variables undefined."""

        with model(treatments=['Bar', 'Baz']) as m:
            Foo = stock('Foo', PerTreatment({'Bar': 1, 'Baz': 2}), 0)
            Quz = variable('Quz', lambda x: x, 'Foo').undefined_in('Baz')
            Corge = accum('Corge', PerTreatment({'Bar': 1, 'Baz': 2}), 0) 

        self.assertEqual(
            m.history(),
            {
                'Foo': {'Bar': [0], 'Baz': [0]},
                'Quz': {'Bar': [0]},
                'Corge': {'Bar': [0], 'Baz': [0]} 
            })

        m.step(10)

        self.assertEqual(
            m.history(),
            {
                'Foo': {
                    'Bar': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                    'Baz': [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
                    },
                'Quz': {
                    'Bar': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10] 
                    },
                'Corge': {
                    'Bar': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                    'Baz': [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
                    }
            })

    def test_validate_all(self):
        """Test validate_all(), with a variable in one treatment."""
        with model(treatments=['all good', 'one bad']) as m:
            constant('Small', 0.4)
            constant('Medium', 0.5).undefined_in('one bad')
            Large = constant('Large', 0.05)

            constraint(
                ['Small', 'Medium', 'Large'],
                lambda *sizes: sum(sizes) == 1.0,
                'InvalidDistribution',
                lambda names, amounts, treatment: 
                    'Distribution of {} sums to {}, not 1.0, in {}'.format(
                        ", ".join(names), round(sum(amounts), 3), treatment))


        vresult = m.validate_all()
        self.assertEqual(vresult['success'], False)
        Large['__all__'] = 0.1
        self.assertEqual(m.validate_all(), {'success': True})

    # The existing error handling is OK, better than trying to catch this
    # explicitly and not allowing dispatch functions.
    
    # def test_undefined_rainy_day(self):
    #     """Test error handling when a variable uses an undefined variable."""
    #     with self.assertRaisesRegex(MinnetonkaError,
    #             "Bar uses undefined conjecture amount of Foo"):
    #         with model(treatments=['conjecture', 'current']):
    #             Foo = constant('Foo', 12 ).undefined_in('conjecture')
    #             Bar = variable('Bar', lambda x: x + 1, 'Foo')

class OnInitTest(unittest.TestCase):
    """Test on_init and on_reset."""
    def test_simple(self):
        """Test on_init and on_reset."""
        def set_seed(md):
            random.seed(99)

        with model(on_init=set_seed, on_reset=set_seed) as m:
            foo = variable('Foo', lambda: random.randint(0, 999))

        foo1a = foo['']
        m.step()
        foo2a = foo['']
        m.reset()
        foo1b = foo['']
        m.step()
        foo2b = foo['']

        self.assertEqual(foo1a, foo1b)
        self.assertEqual(foo2a, foo2b)


class DetailsTest(unittest.TestCase):
    """Test the details function."""
    def assertDictAlmostEqual(self, x, y):
        if isinstance(x, (int, float, complex)):
            self.assertAlmostEqual(x, y)
        elif isinstance(x, dict):
            for k in x.keys():
                self.assertDictAlmostEqual(x[k], y[k])
            # in case some keys in y are not in x
            for k in y.keys():
                self.assertDictAlmostEqual(x[k], y[k])
        else:
            self.assertEqual(x, y)

    def test_constant(self):
        """Test details for a constant."""

        with model() as m:
            foo = constant('Foo', 12)
            bar = constant('Bar', 99
            ).substitute_description_for_amount("Bar is a pathetic constant")

        self.assertEqual(
            foo.details(),
            {
                'name': 'Foo',
                'varies over time': False,
                'amount': {"": 12},
                'caucus': {"": 12}
            })

        self.assertEqual(
            bar.details(),
            {
                'name': 'Bar',
                'varies over time': False,
                'summary description': 'Bar is a pathetic constant',
                'caucus': 'Bar is a pathetic constant'
            })

        with model(
            treatments=['As is', 'To be'], 
            derived_treatments={'Improvement': AmountBetter('To be', 'As is')}
        ) as m:

            foo = constant('Foo', PerTreatment({'As is': 2, 'To be': 2.6}))
            baz = constant('Baz', lambda x: x+x, 'Foo', 
            ).derived()
            bar = constant('Bar', PerTreatment({'As is': 20})
            ).undefined_in('To be'
            ).summarizer(
                "Number as English",
                lambda x, _: "Twenty" if x == 20 else "Not Twenty")

        foo_deets = foo.details() 
        self.assertEqual(
            foo_deets,
            {
                'name': 'Foo',
                'varies over time': False,
                'amount':{'To be': 2.6, 'As is': 2},
                'caucus': {'To be': 2.6, 'As is': 2}
            })

        self.assertEqual(
            bar.details(),
            {
                'name': 'Bar',
                'varies over time': False,
                'summary description': "Number as English",
                'summary':{'As is': "Twenty"},
                'caucus': {'As is': "Twenty"}
            })

        self.assertDictAlmostEqual(
            baz.details(),
            {
                'name': 'Baz',
                'varies over time': False, 
                'amount':{'To be': 5.2, 'As is': 4, 'Improvement': 1.2},
                'caucus': {'To be': 5.2, 'As is': 4, 'Improvement': 1.2}
            })

    def test_normal_variable(self):
        """Test details for a normal variable."""
        def _convert_to_english(num, _):
            num2words = {
                1: 'One', 2: 'Two', 3: 'Three', 4: 'Four', 5: 'Five', 
                6: 'Six', 7: 'Seven', 8: 'Eight', 9: 'Nine', 10: 'Ten', 
                11: 'Eleven', 12: 'Twelve', 13: 'Thirteen', 14: 'Fourteen', 
                15: 'Fifteen', 16: 'Sixteen', 17: 'Seventeen', 18: 'Eighteen', 
                19: 'Nineteen', 20: 'Twenty', 0: 'Zero'}
            try:
                return num2words[num]
            except KeyError:
                return "Many"

        with model(treatments=['As is', 'To be']) as m:
            week = variable('Week', lambda md: md.TIME, '__model__'
            ).summarizer("As English", _convert_to_english
            ).caucuser(lambda amts: 'complex')
            next_week = variable('NextWeek', lambda x: x+1, 'Week'
            ).caucuser(sum)
            week_after = variable('WeekAfter', lambda x: x+2, 'Week'
            ).substitute_description_for_amount(
                "Sometime in the distant future")

        m.step(4)
        self.assertEqual(
            week.details(),
            {
                'name': 'Week',
                'varies over time': True,
                'summary description': 'As English',
                'summary': {'As is': ['Zero', 'One', 'Two', 'Three', 'Four'],
                            'To be': ['Zero', 'One', 'Two', 'Three', 'Four']},
                'caucus': {'As is': 'complex', 'To be': 'complex'}
            })

        self.assertEqual(
            next_week.details(),
            {
                'name': 'NextWeek',
                'varies over time': True,
                'amounts': {'As is': [1, 2, 3, 4, 5],
                            'To be': [1, 2, 3, 4, 5]},
                'caucus': {'As is': 15.0, 'To be': 15.0}
            })

        self.assertEqual(
            week_after.details(),
            {
                'name': 'WeekAfter',
                'varies over time': True,
                'summary description': 'Sometime in the distant future',
                'caucus': 'Sometime in the distant future'
            })

    def test_use_treatment(self):
        """Test details with a summarizer that uses the treatment."""
        def _convert_to_english(num, trt):
            num2words = {
                1: 'One', 2: 'Two', 3: 'Three', 4: 'Four', 5: 'Five', 
                6: 'Six', 7: 'Seven', 8: 'Eight', 9: 'Nine', 10: 'Ten', 
                11: 'Eleven', 12: 'Twelve', 13: 'Thirteen', 14: 'Fourteen', 
                15: 'Fifteen', 16: 'Sixteen', 17: 'Seventeen', 18: 'Eighteen', 
                19: 'Nineteen', 20: 'Twenty', 0: 'Zero'}
            try:
                return num2words[num] + ' ' + trt 
            except KeyError:
                return "Many"

        with model(treatments=['As is', 'To be']) as m:
            week = variable('Week', lambda md: md.TIME, '__model__'
            ).summarizer("As English", _convert_to_english
            ).caucuser(lambda amts: 'complex')
            next_week = variable('NextWeek', lambda x: x+1, 'Week')
            week_after = variable('WeekAfter', lambda x: x+2, 'Week'
            ).substitute_description_for_amount(
                "Sometime in the distant future")

        m.step(4)
        self.assertEqual(
            week.details(),
            {
                'name': 'Week',
                'varies over time': True,
                'summary description': 'As English',
                'summary': {'As is': ['Zero As is', 'One As is', 'Two As is', 
                                      'Three As is', 'Four As is'],
                            'To be': ['Zero To be', 'One To be', 'Two To be', 
                                      'Three To be', 'Four To be']},
                'caucus': {'As is': 'complex', 'To be': 'complex'}
            })

    def test_normal_derived(self):
        """Test details with a variable that has a derived treatment."""

        with model(
            treatments=['As is', 'To be'], 
            derived_treatments={'Improvement': AmountBetter('To be', 'As is')}
        ) as m:
            foo = stock('Foo', PerTreatment({'As is': 1, 'To be': 2})
            ).derived().caucuser(sum)
            bar = variable('Bar', lambda x: x, 'Foo'
            ).derived(scored_as='golf')

        m.step(3)
        self.assertEqual(
            foo.details(),
            {
                'name': 'Foo',
                'varies over time': True,
                'amounts': {
                    'As is': [0, 1, 2, 3],
                    'To be': [0, 2, 4, 6],
                    'Improvement': [0, 1, 2, 3]},
                'caucus': {'As is': 6.0, 'To be': 12.0, 'Improvement': 6.0}
            })

        self.assertEqual(
            bar.details(),
            {
                'name': 'Bar',
                'varies over time': True,
                'amounts': {
                    'As is': [0, 1, 2, 3],
                    'To be': [0, 2, 4, 6],
                    'Improvement': [0, -1, -2, -3]
                },
                'caucus': {'As is': 1.5, 'To be': 3.0, 'Improvement': -1.5}
            })


    def test_stocks(self):
        """Test details for a stock."""
        def _convert_to_english(num, _):
            num2words = {
                1: 'One', 2: 'Two', 3: 'Three', 4: 'Four', 5: 'Five', 
                6: 'Six', 7: 'Seven', 8: 'Eight', 9: 'Nine', 10: 'Ten', 
                11: 'Eleven', 12: 'Twelve', 13: 'Thirteen', 14: 'Fourteen', 
                15: 'Fifteen', 16: 'Sixteen', 17: 'Seventeen', 18: 'Eighteen', 
                19: 'Nineteen', 20: 'Twenty', 0: 'Zero'}
            try:
                return num2words[num]
            except KeyError:
                return "Many"

        with model(treatments=['As is', 'To be']) as m:
            first = stock('First', PerTreatment({'As is': 1, 'To be': 2})
            ).summarizer('As English', _convert_to_english
            ).caucuser(lambda amts: 'complex')
            second = stock('Second', lambda f: f, ('First',), 0
            ).caucuser(sum)
            third = stock('Third', lambda f, s: f + s, ('First', 'Second'), 0
            ).substitute_description_for_amount('A lot')

        m.step(4)
        self.assertEqual(
            first.details(),
            {
                'name': 'First',
                'varies over time': True,
                'summary description': 'As English',
                'summary': {'As is': ['Zero', 'One', 'Two', 'Three', 'Four'],
                            'To be': ['Zero', 'Two', 'Four', 'Six', 'Eight']},
                'caucus': {'As is': 'complex', 'To be': 'complex'}
            })

        self.assertEqual(
            second.details(),
            {
                'name': 'Second',
                'varies over time': True,
                'amounts': {'As is': [0, 0, 1, 3, 6],
                            'To be': [0, 0, 2, 6, 12]},
                'caucus': {'As is': 10.0, 'To be': 20.0}
            })

        self.assertEqual(
            third.details(),
            {
                'name': 'Third',
                'varies over time': True,
                'summary description': 'A lot',
                'caucus': 'A lot'
            })

    def test_accums(self):
        """Test details for an accum."""
        def _convert_to_english(num, _):
            num2words = {
                1: 'One', 2: 'Two', 3: 'Three', 4: 'Four', 5: 'Five', 
                6: 'Six', 7: 'Seven', 8: 'Eight', 9: 'Nine', 10: 'Ten', 
                11: 'Eleven', 12: 'Twelve', 13: 'Thirteen', 14: 'Fourteen', 
                15: 'Fifteen', 16: 'Sixteen', 17: 'Seventeen', 18: 'Eighteen', 
                19: 'Nineteen', 20: 'Twenty', 0: 'Zero'}
            try:
                return num2words[num]
            except KeyError:
                return "Many"

        with model(treatments=['As is', 'To be']) as m:
            first = accum('First', PerTreatment({'As is': 1, 'To be': 2})
            ).summarizer('As English', _convert_to_english
            ).caucuser(lambda amts: 'complex')
            second = accum('Second', lambda f: f, ('First',), 0)
            third = accum('Third', lambda f, s: f + s, ('First', 'Second'), 0
            ).substitute_description_for_amount('A lot')

        m.step(4)
        self.assertEqual(
            first.details(),
            {
                'name': 'First',
                'varies over time': True,
                'summary description': 'As English',
                'summary': {'As is': ['Zero', 'One', 'Two', 'Three', 'Four'],
                            'To be': ['Zero', 'Two', 'Four', 'Six', 'Eight']},
                'caucus': {'As is': 'complex', 'To be': 'complex'}
            })

        self.assertEqual(
            second.details(),
            {
                'name': 'Second',
                'varies over time': True,
                'amounts': {'As is': [0, 1, 3, 6, 10],
                            'To be': [0, 2, 6, 12, 20]},
                'caucus': {'As is': 4.0, 'To be': 8.0}
            })

        self.assertEqual(
            third.details(),
            {
                'name': 'Third',
                'varies over time': True,
                'caucus': 'A lot',
                'summary description': 'A lot'
            })

    def test_previous(self):
        """Test details for a normal variable."""
        def _convert_to_english(num, _):
            num2words = {
                1: 'One', 2: 'Two', 3: 'Three', 4: 'Four', 5: 'Five', 
                6: 'Six', 7: 'Seven', 8: 'Eight', 9: 'Nine', 10: 'Ten', 
                11: 'Eleven', 12: 'Twelve', 13: 'Thirteen', 14: 'Fourteen', 
                15: 'Fifteen', 16: 'Sixteen', 17: 'Seventeen', 18: 'Eighteen', 
                19: 'Nineteen', 20: 'Twenty', 0: 'Zero'}
            try:
                return num2words[num]
            except KeyError:
                return "Many"

        with model() as m:
            stock('Foo', 1, 0)
            lf = previous('LastFoo', 'Foo'
            ).summarizer('As English', _convert_to_english
            ).caucuser(lambda amts: 'complex')
            lf2 = previous('LastFoo2', 'Foo').caucuser(sum)
            lf3 = previous('LastFoo3', 'Foo'
            ).substitute_description_for_amount('A lot')

        m.step(4)
        self.assertEqual(
            lf.details(),
            {
                'name': 'LastFoo',
                'varies over time': True,
                'summary description': 'As English',
                'summary': {'': ['Zero', 'Zero', 'One', 'Two', 'Three']},
                'caucus': {'': 'complex'}
            })

        self.assertEqual(
            lf2.details(),
            {
                'name': 'LastFoo2',
                'varies over time': True,
                'amounts': {'': [0, 0, 1, 2, 3]},
                'caucus': {'': 6.0}
            })

        self.assertEqual(
            lf3.details(),
            {
                'name': 'LastFoo3',
                'varies over time': True,
                'summary description': 'A lot',
                'caucus': 'A lot'
            })

    def test_cross(self):
        """Test details for a cross variable."""
        def _convert_to_english(num, _):
            num2words = {
                1: 'One', 2: 'Two', 3: 'Three', 4: 'Four', 5: 'Five', 
                6: 'Six', 7: 'Seven', 8: 'Eight', 9: 'Nine', 10: 'Ten', 
                11: 'Eleven', 12: 'Twelve', 13: 'Thirteen', 14: 'Fourteen', 
                15: 'Fifteen', 16: 'Sixteen', 17: 'Seventeen', 18: 'Eighteen', 
                19: 'Nineteen', 20: 'Twenty', 0: 'Zero'}
            try:
                return num2words[num]
            except KeyError:
                return "Many"

        with model(treatments=['As is', 'To be']) as m:
            s = stock('S', PerTreatment({'As is': 1, 'To be': 2}))
            foo = cross('Foo', 'S', 'As is'
            ).summarizer('As English', _convert_to_english
            ).caucuser(lambda amts: 'complex') 

        m.step(4)
        self.assertEqual(
            foo.details(),
            {
                'name': 'Foo',
                'varies over time': True,
                'summary description': 'As English',
                'summary': {'As is': ['Zero', 'One', 'Two', 'Three', 'Four'],
                            'To be': ['Zero', 'One', 'Two', 'Three', 'Four']},
                'caucus': {'As is': 'complex', 'To be': 'complex'}
            })
 

class ModifiedTest(unittest.TestCase):
    """Test the is_modified() function on a variable instance."""
    def test_constant(self):
        """Test is_modified on a constant instance."""
        with model() as m:
            foo = constant('Foo', 12)

        self.assertFalse(m.is_modified('Foo', ''))
        foo[''] = 13
        self.assertTrue(m.is_modified('Foo', ''))

        with model(treatments=['As is', 'To be']) as m:
            bar = constant('Bar', PerTreatment({'As is': 12, 'To be': 13}))

        self.assertFalse(m.is_modified('Bar', 'As is'))
        self.assertFalse(m.is_modified('Bar', 'To be'))
        bar['To be'] = 14
        self.assertFalse(m.is_modified('Bar', 'As is'))
        self.assertTrue(m.is_modified('Bar', 'To be'))
        bar['As is'] = 13
        self.assertTrue(m.is_modified('Bar', 'As is'))
        self.assertTrue(m.is_modified('Bar', 'To be'))

    def test_constant_reset(self):
        """Test is_modified on a constant after being reset."""
        with model() as m:
            foo = constant('Foo', 12) 

        foo[''] = 13
        self.assertTrue(m.is_modified('Foo', ''))
        m.reset()
        self.assertFalse(m.is_modified('Foo', ''))
        foo[''] = 14
        self.assertTrue(m.is_modified('Foo', ''))
        m.reset(reset_external_vars=False)
        self.assertTrue(m.is_modified('Foo', ''))

    def test_variable(self):
        """Test is_modified on a variable instance."""
        with model() as m:
            foo = constant('Foo', 12)
            bar = variable('Bar', lambda x: x + 2, 'Foo')

        self.assertFalse(m.is_modified('Bar', ''))
        foo[''] = 13        
        self.assertFalse(m.is_modified('Bar', ''))
        bar[''] = 99       
        self.assertTrue(m.is_modified('Bar', ''))


class VelocityTest(unittest.TestCase):
    """Testing velocity()"""
    def test_simple(self):
        """Test a simple use of velocity work, with a stock"""
        with model() as m:
            stock('Foo', 1, 0)
            FooVelocity = velocity('FooVelocity', 'Foo')

        self.assertEqual(FooVelocity[''], 0)
        m.step()
        self.assertEqual(FooVelocity[''], 1)
        m.step()
        self.assertEqual(FooVelocity[''], 1)
        m.reset()
        self.assertEqual(FooVelocity[''], 0)

    def test_timestep(self):
        """Test a simple use of velocity work, with a timestep that is not 1"""
        with model(timestep=0.5) as m:
            stock('Foo', 1, 0)
            FooVelocity = velocity('FooVelocity', 'Foo')

        self.assertEqual(FooVelocity[''], 0)
        m.step()
        self.assertEqual(FooVelocity[''], 1)
        m.step()
        self.assertEqual(FooVelocity[''], 1)
        m.reset()
        self.assertEqual(FooVelocity[''], 0)

    def test_treatments(self):
        """Test velocity works across treatments"""
        with model(treatments=['as is', 'to be']) as m:
            Bar = stock('Bar', 1, 0)
            Foo = variable('Foo', 
                PerTreatment({
                    'as is': lambda x: x * x,
                    'to be': lambda x: x * x * x
                    }),
                'Bar')
            FooVelocity = velocity('FooVelocity', 'Foo')

        self.assertEqual(FooVelocity['as is'], 0)
        self.assertEqual(FooVelocity['to be'], 0)
        m.step()        
        self.assertEqual(FooVelocity['as is'], 1)
        self.assertEqual(FooVelocity['to be'], 1) 
        m.step()
        self.assertEqual(FooVelocity['as is'], 3)
        self.assertEqual(FooVelocity['to be'], 7)  
        m.step()
        self.assertEqual(FooVelocity['as is'], 5)
        self.assertEqual(FooVelocity['to be'], 19) 
        m.reset()
        self.assertEqual(FooVelocity['as is'], 0)
        self.assertEqual(FooVelocity['to be'], 0)
        m.step()        
        self.assertEqual(FooVelocity['as is'], 1)
        self.assertEqual(FooVelocity['to be'], 1)  

    def test_cycle(self):
        """Test that a cycle is caught."""
        with self.assertRaises(MinnetonkaError) as me:
            with model() as m:
                Foo = variable('Foo', lambda x: x+1, 'FooVelocity')
                FooVelocity = velocity('FooVelocity', 'Foo')
        self.assertEqual(me.exception.message,
            'Circularity among variables: Foo <- FooVelocity <- Foo')

    def test_array(self):
        """Test velocity with numpy arrays."""
        with model() as m:
            stock('Savings', 
                lambda x: x, 'Interest', np.array([1000, 1000, 1000]))
            variable('Interest',
                lambda savings, rate: savings * rate,
                'Savings',
                'Rate')
            constant('Rate', np.array([0.08, 0.09, 0.1]))
            SavingsVelocity = velocity('SavingsVelocity', 'Savings')

        assert_array_equal(SavingsVelocity[''], np.array([0, 0, 0]))
        m.step()
        assert_array_equal(SavingsVelocity[''], np.array([80, 90, 100]))
        m.step()
        assert_array_almost_equal(
            SavingsVelocity[''], np.array([86.4, 98.1, 110]))
        m.step()
        assert_array_almost_equal(
            SavingsVelocity[''], np.array([93.312, 106.929, 121]))
        m.reset()
        assert_array_equal(SavingsVelocity[''], np.array([0, 0, 0]))
        m.step()
        assert_array_equal(SavingsVelocity[''], np.array([80, 90, 100]))

    def test_array_timestep(self):
        """Test velocity with numpy arrays and nonzero timestep."""
        with model(timestep=0.5) as m:
            stock('Savings', 
                lambda x: x, 'Interest', np.array([1000, 1000, 1000]))
            variable('Interest',
                lambda savings, rate: savings * rate,
                'Savings',
                'Rate')
            constant('Rate', np.array([0.08, 0.09, 0.1]))
            SavingsVelocity = velocity('SavingsVelocity', 'Savings')

        assert_array_equal(SavingsVelocity[''], np.array([0, 0, 0]))
        m.step()
        assert_array_equal(SavingsVelocity[''], np.array([80, 90, 100]))
        m.step()
        assert_array_almost_equal(
            SavingsVelocity[''], np.array([83.2, 94.05, 105]))
        m.step()
        assert_array_almost_equal(
            SavingsVelocity[''], np.array([86.528, 98.28225, 110.25]))
        m.reset()
        assert_array_equal(SavingsVelocity[''], np.array([0, 0, 0]))
        m.step()
        assert_array_equal(SavingsVelocity[''], np.array([80, 90, 100]))

    def test_tuple(self):
        """Test velocity with tuple values."""
        with model() as m:
            stock('Savings', 
                foreach(lambda x: x), 'Interest', (1000, 1000, 1000))
            variable('Interest',
                foreach(lambda savings, rate: savings * rate),
                'Savings',
                'Rate')
            constant('Rate', (0.08, 0.09, 0.1))
            SavingsVelocity = velocity('SavingsVelocity', 'Savings')

        self.assertEqual(SavingsVelocity[''], (0, 0, 0))
        m.step()
        self.assertEqual(SavingsVelocity[''], (80, 90, 100))
        m.step()
        self.assertAlmostEqual(SavingsVelocity[''][0], 86.4)
        self.assertAlmostEqual(SavingsVelocity[''][1], 98.1)
        self.assertAlmostEqual(SavingsVelocity[''][2], 110)
        m.step()
        self.assertAlmostEqual(SavingsVelocity[''][0], 93.312)
        self.assertAlmostEqual(SavingsVelocity[''][1], 106.929)
        self.assertAlmostEqual(SavingsVelocity[''][2], 121)
        m.reset()
        self.assertEqual(SavingsVelocity[''], (0, 0, 0))
        m.step()
        self.assertEqual(SavingsVelocity[''], (80, 90, 100))

    def test_tuple_timestep(self):
        """Test velocity with tuple values and nonzero timestep."""
        with model(timestep=0.5) as m:
            stock('Savings', 
                foreach(lambda x: x), 'Interest', (1000, 1000, 1000))
            variable('Interest',
                foreach(lambda savings, rate: savings * rate),
                'Savings',
                'Rate')
            constant('Rate', (0.08, 0.09, 0.1))
            SavingsVelocity = velocity('SavingsVelocity', 'Savings')

        self.assertEqual(SavingsVelocity[''], (0, 0, 0))
        m.step()
        self.assertEqual(SavingsVelocity[''], (80, 90, 100))
        m.step()
        self.assertAlmostEqual(SavingsVelocity[''][0], 83.2)
        self.assertAlmostEqual(SavingsVelocity[''][1], 94.05)
        self.assertAlmostEqual(SavingsVelocity[''][2], 105)
        m.step()
        self.assertAlmostEqual(SavingsVelocity[''][0], 86.528)
        self.assertAlmostEqual(SavingsVelocity[''][1], 98.28225)
        self.assertAlmostEqual(SavingsVelocity[''][2], 110.25)
        m.reset()
        self.assertEqual(SavingsVelocity[''], (0, 0, 0))
        m.step()
        self.assertEqual(SavingsVelocity[''], (80, 90, 100))

    def test_dict(self):
        """Test velocity with dict values."""
        with model() as m:
            stock('Savings', 
                foreach(lambda x: x), 'Interest', 
                {'foo': 1000, 'bar': 900, 'baz': 800})
            variable('Interest',
                foreach(lambda savings, rate: savings * rate),
                'Savings',
                'Rate')
            constant('Rate', {'foo': 0.08, 'bar': 0.09, 'baz': 0.1})
            SavingsVelocity = velocity('SavingsVelocity', 'Savings')

        self.assertEqual(SavingsVelocity[''], {'foo': 0, 'bar': 0, 'baz': 0})
        m.step()
        self.assertEqual(SavingsVelocity[''], {'foo': 80, 'bar': 81, 'baz': 80})
        m.step()
        self.assertAlmostEqual(SavingsVelocity['']['foo'], 86.4)
        self.assertAlmostEqual(SavingsVelocity['']['bar'], 88.29)
        self.assertAlmostEqual(SavingsVelocity['']['baz'], 88.0)
        m.step()
        self.assertAlmostEqual(SavingsVelocity['']['foo'], 93.312)
        self.assertAlmostEqual(SavingsVelocity['']['bar'], 96.2361)
        self.assertAlmostEqual(SavingsVelocity['']['baz'], 96.8)
        m.reset()
        self.assertEqual(SavingsVelocity[''], {'foo': 0, 'bar': 0, 'baz': 0})
        m.step()
        self.assertEqual(SavingsVelocity[''], {'foo': 80, 'bar': 81, 'baz': 80})

    def test_dict_timestep(self):
        """Test velocity with dict values and nonzero timestep."""
        with model(timestep=0.5) as m:
            stock('Savings', 
                foreach(lambda x: x), 'Interest', 
                {'foo': 1000, 'bar': 900, 'baz': 800})
            variable('Interest',
                foreach(lambda savings, rate: savings * rate),
                'Savings',
                'Rate')
            constant('Rate', {'foo': 0.08, 'bar': 0.09, 'baz': 0.1})
            SavingsVelocity = velocity('SavingsVelocity', 'Savings')

        self.assertEqual(SavingsVelocity[''], {'foo': 0, 'bar': 0, 'baz': 0})
        m.step(2)
        self.assertAlmostEqual(SavingsVelocity['']['foo'], 83.2)
        self.assertAlmostEqual(SavingsVelocity['']['bar'], 84.645)
        self.assertAlmostEqual(SavingsVelocity['']['baz'], 84.0)
        m.step(2)
        self.assertAlmostEqual(SavingsVelocity['']['foo'], 89.98912)
        self.assertAlmostEqual(SavingsVelocity['']['bar'], 92.4344561)
        self.assertAlmostEqual(SavingsVelocity['']['baz'], 92.61)
        m.step()
        self.assertAlmostEqual(SavingsVelocity['']['foo'], 93.5886848)
        self.assertAlmostEqual(SavingsVelocity['']['bar'], 96.5940067)
        self.assertAlmostEqual(SavingsVelocity['']['baz'], 97.2405)
        m.reset()
        self.assertEqual(SavingsVelocity[''], {'foo': 0, 'bar': 0, 'baz': 0})
        m.step()
        self.assertEqual(SavingsVelocity[''], {'foo': 80, 'bar': 81, 'baz': 80})

    def test_named_tuple(self):
        """Test velocity with named tuple values."""
        FBB = collections.namedtuple('FBB', ['foo', 'bar', 'baz'])
        with model() as m:
            stock('Savings', 
                foreach(lambda x: x), 'Interest', 
                FBB(foo=1000, bar=900, baz=800))
            variable('Interest',
                foreach(lambda savings, rate: savings * rate),
                'Savings',
                'Rate')
            constant('Rate', FBB(foo=0.08, bar=0.09, baz=0.1))
            SavingsVelocity = velocity('SavingsVelocity', 'Savings')

        self.assertEqual(SavingsVelocity[''], FBB(foo=0, bar=0, baz=0))
        m.step()
        self.assertEqual(SavingsVelocity[''], FBB(foo=80, bar=81, baz=80))
        m.step()
        self.assertAlmostEqual(SavingsVelocity[''].foo, 86.4)
        self.assertAlmostEqual(SavingsVelocity[''].bar, 88.29)
        self.assertAlmostEqual(SavingsVelocity[''].baz, 88.0)
        m.step()
        self.assertAlmostEqual(SavingsVelocity[''].foo, 93.312)
        self.assertAlmostEqual(SavingsVelocity[''].bar, 96.2361)
        self.assertAlmostEqual(SavingsVelocity[''].baz, 96.8)
        m.reset()
        self.assertEqual(SavingsVelocity[''], FBB(foo=0, bar=0, baz=0))
        m.step()
        self.assertEqual(SavingsVelocity[''], FBB(foo=80, bar=81, baz=80))

    def test_named_tuple_timestep(self):
        """Test velocity with named tuple values and nonzero timestep."""
        FBB = collections.namedtuple('FBB', ['foo', 'bar', 'baz'])
        with model(timestep=0.5) as m:
            stock('Savings', 
                foreach(lambda x: x), 'Interest', 
                FBB(foo=1000, bar=900, baz=800))
            variable('Interest',
                foreach(lambda savings, rate: savings * rate),
                'Savings',
                'Rate')
            constant('Rate', FBB(foo=0.08, bar=0.09, baz=0.1))
            SavingsVelocity = velocity('SavingsVelocity', 'Savings')

        self.assertEqual(SavingsVelocity[''], FBB(foo=0, bar=0, baz=0))
        m.step(2)
        self.assertAlmostEqual(SavingsVelocity[''].foo, 83.2)
        self.assertAlmostEqual(SavingsVelocity[''].bar, 84.645)
        self.assertAlmostEqual(SavingsVelocity[''].baz, 84.0)
        m.step(2)
        self.assertAlmostEqual(SavingsVelocity[''].foo, 89.98912)
        self.assertAlmostEqual(SavingsVelocity[''].bar, 92.4344561)
        self.assertAlmostEqual(SavingsVelocity[''].baz, 92.61)
        m.step()
        self.assertAlmostEqual(SavingsVelocity[''].foo, 93.5886848)
        self.assertAlmostEqual(SavingsVelocity[''].bar, 96.5940067)
        self.assertAlmostEqual(SavingsVelocity[''].baz, 97.2405)
        m.reset()
        self.assertEqual(SavingsVelocity[''], FBB(foo=0, bar=0, baz=0))
        m.step()
        self.assertEqual(SavingsVelocity[''], FBB(foo=80, bar=81, baz=80))  

    def test_mn_named_tuple(self):
        """Test velocity with mn_namedtuple values."""
        FBB = mn_namedtuple('FBB', ['foo', 'bar', 'baz'])
        with model() as m:
            stock('Savings', 
                lambda x: x, 'Interest', 
                FBB(foo=1000, bar=900, baz=800))
            variable('Interest',
                lambda savings, rate: savings * rate,
                'Savings',
                'Rate')
            constant('Rate', FBB(foo=0.08, bar=0.09, baz=0.1))
            SavingsVelocity = velocity('SavingsVelocity', 'Savings')

        self.assertEqual(SavingsVelocity[''], FBB(foo=0, bar=0, baz=0))
        m.step()
        self.assertEqual(SavingsVelocity[''], FBB(foo=80, bar=81, baz=80))
        m.step()
        self.assertAlmostEqual(SavingsVelocity[''].foo, 86.4)
        self.assertAlmostEqual(SavingsVelocity[''].bar, 88.29)
        self.assertAlmostEqual(SavingsVelocity[''].baz, 88.0)
        m.step()
        self.assertAlmostEqual(SavingsVelocity[''].foo, 93.312)
        self.assertAlmostEqual(SavingsVelocity[''].bar, 96.2361)
        self.assertAlmostEqual(SavingsVelocity[''].baz, 96.8)
        m.reset()
        self.assertEqual(SavingsVelocity[''], FBB(foo=0, bar=0, baz=0))
        m.step()
        self.assertEqual(SavingsVelocity[''], FBB(foo=80, bar=81, baz=80))

    def test_mn_named_tuple_timestep(self):
        """Test velocity with mn_namedtuple values and nonzero timestep."""
        FBB = mn_namedtuple('FBB', ['foo', 'bar', 'baz'])
        with model(timestep=0.5) as m:
            stock('Savings', 
                lambda x: x, 'Interest', 
                FBB(foo=1000, bar=900, baz=800))
            variable('Interest',
                lambda savings, rate: savings * rate,
                'Savings',
                'Rate')
            constant('Rate', FBB(foo=0.08, bar=0.09, baz=0.1))
            SavingsVelocity = velocity('SavingsVelocity', 'Savings')

        self.assertEqual(SavingsVelocity[''], FBB(foo=0, bar=0, baz=0))
        m.step(2)
        self.assertAlmostEqual(SavingsVelocity[''].foo, 83.2)
        self.assertAlmostEqual(SavingsVelocity[''].bar, 84.645)
        self.assertAlmostEqual(SavingsVelocity[''].baz, 84.0)
        m.step(2)
        self.assertAlmostEqual(SavingsVelocity[''].foo, 89.98912)
        self.assertAlmostEqual(SavingsVelocity[''].bar, 92.4344561)
        self.assertAlmostEqual(SavingsVelocity[''].baz, 92.61)
        m.step()
        self.assertAlmostEqual(SavingsVelocity[''].foo, 93.5886848)
        self.assertAlmostEqual(SavingsVelocity[''].bar, 96.5940067)
        self.assertAlmostEqual(SavingsVelocity[''].baz, 97.2405)
        m.reset()
        self.assertEqual(SavingsVelocity[''], FBB(foo=0, bar=0, baz=0))
        m.step()
        self.assertEqual(SavingsVelocity[''], FBB(foo=80, bar=81, baz=80))  

    def test_dict_tuple(self):
        """Test velocity with dicts of tuples."""
        with model() as m:
            stock('Savings', 
                foreach(foreach(lambda x: x)), 'Interest', 
                {'foo': (1000, 1050), 'bar': (900, 950), 'baz': (800, 850)})
            variable('Interest',
                foreach(lambda savings, rate: tuple(s * rate for s in savings)),
                'Savings',
                'Rate')
            constant('Rate', {'foo': 0.08, 'bar': 0.09, 'baz': 0.1})
            SavingsVelocity = velocity('SavingsVelocity', 'Savings')

        self.assertEqual(
            SavingsVelocity[''], 
            {'foo': (0, 0), 'bar': (0, 0), 'baz': (0, 0)})
        m.step()
        self.assertEqual(
            SavingsVelocity[''], 
            {'foo': (80.0, 84.0), 'bar': (81.0, 85.5), 'baz': (80.0, 85.0)})
        m.step()
        self.assertAlmostEqual(SavingsVelocity['']['foo'][0], 86.4)
        self.assertAlmostEqual(SavingsVelocity['']['bar'][0], 88.29)
        self.assertAlmostEqual(SavingsVelocity['']['baz'][0], 88.0)        
        self.assertAlmostEqual(SavingsVelocity['']['foo'][1], 90.72)
        self.assertAlmostEqual(SavingsVelocity['']['bar'][1], 93.195)
        self.assertAlmostEqual(SavingsVelocity['']['baz'][1], 93.5) 
        m.reset()
        self.assertEqual(
            SavingsVelocity[''], 
            {'foo': (0, 0), 'bar': (0, 0), 'baz': (0, 0)})
        m.step()
        self.assertEqual(
            SavingsVelocity[''], 
            {'foo': (80.0, 84.0), 'bar': (81.0, 85.5), 'baz': (80.0, 85.0)}) 

    def test_dict_tuple_timestep(self):
        """Test velocity with dict of tuples and nonzero timestep."""
        with model(timestep=0.5) as m:
            stock('Savings', 
                foreach(foreach(lambda x: x)), 'Interest', 
                {'foo': (1000, 1050), 'bar': (900, 950), 'baz': (800, 850)})
            variable('Interest',
                foreach(lambda savings, rate: tuple(s * rate for s in savings)),
                'Savings',
                'Rate')
            constant('Rate', {'foo': 0.08, 'bar': 0.09, 'baz': 0.1})
            SavingsVelocity = velocity('SavingsVelocity', 'Savings') 

        self.assertEqual(
            SavingsVelocity[''], 
            {'foo': (0, 0), 'bar': (0, 0), 'baz': (0, 0)})
        m.step(2)
        self.assertAlmostEqual(SavingsVelocity['']['foo'][0], 83.2)       
        self.assertAlmostEqual(SavingsVelocity['']['foo'][1], 87.36)
        self.assertAlmostEqual(SavingsVelocity['']['bar'][0], 84.645)
        self.assertAlmostEqual(SavingsVelocity['']['bar'][1], 89.3475)
        self.assertAlmostEqual(SavingsVelocity['']['baz'][0], 84.0) 
        self.assertAlmostEqual(SavingsVelocity['']['baz'][1], 89.25)  
        m.step(2)
        self.assertAlmostEqual(SavingsVelocity['']['foo'][0], 89.98912)      
        self.assertAlmostEqual(SavingsVelocity['']['foo'][1], 94.488576)
        self.assertAlmostEqual(SavingsVelocity['']['bar'][0], 92.4344561)
        self.assertAlmostEqual(SavingsVelocity['']['bar'][1], 97.5697037)
        self.assertAlmostEqual(SavingsVelocity['']['baz'][0], 92.61)  
        self.assertAlmostEqual(SavingsVelocity['']['baz'][1], 98.398125) 
        m.reset()
        self.assertEqual(
            SavingsVelocity[''], 
            {'foo': (0, 0), 'bar': (0, 0), 'baz': (0, 0)})
        m.step(2)
        self.assertAlmostEqual(SavingsVelocity['']['foo'][0], 83.2)       
        self.assertAlmostEqual(SavingsVelocity['']['foo'][1], 87.36)
        self.assertAlmostEqual(SavingsVelocity['']['bar'][0], 84.645)
        self.assertAlmostEqual(SavingsVelocity['']['bar'][1], 89.3475)
        self.assertAlmostEqual(SavingsVelocity['']['baz'][0], 84.0) 
        self.assertAlmostEqual(SavingsVelocity['']['baz'][1], 89.25)  

class PerTreatmentAlternative(unittest.TestCase):
    """Test alternative syntax for PerTreatment."""
    def test_alt_syntax(self):
        """Test per_treatment."""
        with model(treatments=['as_is', 'to_be']) as m:
            Foo = variable('Foo', per_treatment(as_is=12, to_be=9))
            Bar = stock('Bar', 
                per_treatment(as_is=lambda x: x, to_be=lambda x: -x),
                'Foo',
                100)
        self.assertEqual(Foo['as_is'], 12)
        self.assertEqual(Foo['to_be'], 9)
        m.step()
        self.assertEqual(Bar['as_is'], 112)
        self.assertEqual(Bar['to_be'], 91)
        m.step()
        self.assertEqual(Bar['as_is'], 124)
        self.assertEqual(Bar['to_be'], 82)

