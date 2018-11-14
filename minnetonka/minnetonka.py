"""minnetonka.py: defining a language for value modeling."""

__author__ = "Dave Bridgeland"
__copyright__ = "Copyright 2017-2018, Hanging Steel Productions LLC"
__credits__ = ["Dave Bridgeland"]
__version__ = "1"
__maintainer__ = "Dave Bridgeland"
__email__ = "dave@hangingsteel.com"
__status__ = "Prototype"

import warnings
import copy
import pdb
import collections 
import itertools
import logging
import json
import time
import inspect
import re

from scipy.stats import norm

import numpy as np

"""
Variables:


    # a varaible can be defined to be the previous value of another
    OldFoo = previous('OldFoo', 'Foo', 1)


    # a variable can show the values of all its treatments as a dict
    Step.all()

Models:

    # A variable can be accessed by name from a model
    m['Earnings']

Treatments:
    # a treatment can be found by name
    as_is = m.treatment('As is')

    # a treatment knows about its name and description
    as_is.name
    as_is.description

    # all the treatments can be found as a list
    m.treatments()

    # a model created without an explicit treatment has a null treatment---a
    # single treatment with an empty string name
    m = model()
    m.treatment('')

Variable values:
    # a variable value can be accessed, by treatment name
    DischargeBegins = variable('DischargeBegins', 12)
    m = model([DischargeBegins])
    DischargeBegins['']
    #    --> 12

    # a variable has a value in each treatment of the model
    DischargeEnds = variable('DischargeEnds',
        PerTreatment{'As is': 20, 'To be': 18}))
    m = model([DischargeEnds], treatments=['As is', 'To be'])
    DischargeEnds['As is']
    #    --> 20
    DischargeEnds['To be']
    #    -- 18

    # a variable value can be set, overriding any other value
    DischargeBegins[''] = 22

    Earnings = variable('Earnings', lambda r, c: r - c, 'Revenue', 'Cost')
    Earnings[''] == 1_000_000

    # a variable value can be set in all treatments at once
    DischargeBegins = variable('DischargeBegins',
            PerTreatment({'As is': 10, 'To be': 8, 'Might be': 6}))
    DischargeBegins['__all__'] = 9



Model behavior:

    # a model can be reset
    m = model([stock('Year', 1, 2017)])
    m.step(10)
    m.reset()
    m['Year']['']
    #   --> 2017

    # resetting resets any overridden variables
    DischargeProgress = variable('DischargeProgress', lambda: 0.5)
    m = model([DischargeProgress])
    DischargeProgress[''] = 0.75
    DischargeProgress['']
    #   --> 0.75
    m.reset()
    DischargeProgress['']
    #   --> 0.5

    # resetting resets any overridden variables
    DischargeProgress = variable('DischargeProgress', lambda: 0.5)
    m = model([DischargeProgress])
    DischargeProgress[''] = 0.75
    DischargeProgress['']
    #   --> 0.75
    m.reset(reset_external_vars=False)
    DischargeProgress['']
    #   --> 0.75

    # setting a variable's values can create a need for explicit recalculation
    Foo = variable('Foo', 9)
    Bar = variable('Bar', lambda f: f, 'Foo')
    m = model([Foo, Bar])
    Foo[''] = 2.4
    Bar['']
    #   --> 9
    m.recalculate()
    Bar['']
    #   --> 2.4

"""


class Model:
    """
    A collection of variables, that can be simulated.

    A model is a self-contained collection of variables and treatments. 
    A model can be simulated, perhaps running one step at a time, perhaps
    multiple steps, perhaps until the end.

    Typically a model is defined as a context using :func:`model`, 
    with variables and stocks within the model context. See example below.

    Parameters
    ----------
    treatments : list of :class:`Treatment`
        The treatments defined for the model. Each treatment is a different
        simulated scenario, run in parallel.

    timestep : int or float, optional
        The simulated duration of each call to :meth:`step`. The default is
        1.

    start_time : int or float, optional
        The first time period, before the first call to :meth:`step`. Default: 
        0

    end_time : int or float, optional
        The last time period, after a call to :meth:`step` 
        with ``to_end=True``. Default: None, meaning never end

    See Also
    --------
    :func:`model` : the typical way to create a model

    Examples
    --------
    Create a model with two treatments and three variables:

    >>> with model(treatments=['As is', 'To be']) as m:
    ...  variable('Revenue', np.array([30.1, 15, 20]))
    ...  variable('Cost', 
    ...     PerTreatment({'As is': np.array([10, 10, 10]),
    ...                  {'To be': np.array([5, 5, 20])})
    ...  variable('Earnings', lambda r, c: r - c, 'Revenue', 'Cost')
    """

    # is a model being defined in a context manager? which one?
    _model_context = None

    def __init__(self, treatments, timestep=1, start_time=0, end_time=None):
        """Initialize the model, with treatments and optional timestep."""
        self._treatments = {t.name: t for t in treatments}
        # prior to m.initialize(), this is a regular dict. It is
        # converted to an OrderedDict on initialization, ordered with
        # dependent variables prior to independent variables
        self._variables = ModelVariables()
        self._pseudo_variable = ModelPseudoVariable(self)
        self._timestep = timestep
        self._start_time = start_time 
        self._end_time = end_time

    def __getitem__(self, variable_name):
        """Return the named variable, supporting [] notation."""
        return self._variables.variable(variable_name)

    def __enter__(self):
        """Enter the model context; accumulate variables to add to model."""
        self._variables_not_yet_added = []
        Model._model_context = self
        self._uninitialize()
        return self

    def __exit__(self, exception_type, exception_value, exception_traceback):
        """Exit the model context; add variables to model."""
        logging.info('enter')
        if exception_type is None:
            self._add_variables_and_initialize(*self._variables_not_yet_added)
        self._variables_not_yet_added = []
        Model._model_context = None
        logging.info('exit')

    def step(self, n=1, to_end=False):
        """
        Simulate the model **n** steps.

        Simulate the model, either one step (default), or ``n`` steps,
        or until the model's end.

        Parameters
        ----------
        n : int, optional
            Number of steps to advance. The default is 1, one step at a time.
        to_end : bool, optional
            If ``True``, simulate the model until its end time

        Returns
        -------
        None

        Raises
        ------
        MinnetonkaError
            If ``to_end`` is ``True`` but the model has no end time.

        Examples
        --------
        A model can simulate one step at a time:

        >>> m = model([stock('Year', 1, 2019)])
        >>> m.step()
        >>> m['Year']['']
        2020
        >>> m.step()
        >>> m['Year']['']
        2021

        A model can simulate several steps at a time:

        >>> m2 = model([stock('Year', 1, 2019)])
        >>> m2.step(n=10)
        >>> m2['Year']['']
        2029

        A model can simulate until the end:
        
        >>> m3 = model([stock('Year', 1, 2019)], end_time=20)
        >>> m3.step(to_end=True)
        >>> m3['Year']['']
        2039
        """
        if self._end_time is None or self.TIME < self._end_time:
            if to_end:
                n = int((self._end_time - self.TIME) / self._timestep)
            for i in range(n):
                self._step_one()
        else:
            raise MinnetonkaError(
                'Attempted to simulation beyond end_time: {}'.format(
                    self._end_time))

    def reset(self, reset_external_vars=True):
        """
        Reset simulation, back to the begining.

        Reset simulation time back to the beginning time, and reset the
        amounts of all variables back to their initial amounts. 

        Parameters
        ----------
        reset_external_vars : bool, optional
            Sometimes variables are set to amounts outside the model logic.
            (See example below, and more examples with :func:`constant`, 
            :func:`variable`, and :func:`stock`.) 
            Should these externally-defined variables be reset to their initial amounts
            when the model as a whole is reset? Default: True, reset those
            externally-defined variables.

        Returns
        -------
            None

        Examples
        --------
        Create a simple model.

        >>> m = model([stock('Year', 1, 2019)])
        >>> m['Year']['']
        2019

        Step the model.

        >>> m.step()
        >>> m['Year']['']
        2020
        >>> m.step()
        >>> m['Year']['']
        2021

        Reset the model.

        >>> m.reset()
        >>> m['Year']['']
        2019

        Change the amount of year. **Year** is now externally defined.

        >>> m['Year'][''] = 1955
        >>> m['Year']['']
        1955

        Reset the model again.

        >>> m.reset(reset_external_vars=False)
        >>> m['Year']['']
        1955

        Reset one more time.

        >>> m.reset()
        >>> m['Year']['']
        2019
        """
        self._initialize_time()
        self._variables.reset(reset_external_vars)

    def initialize(self):
        """Initialize simulation."""
        logging.info('enter')
        self._initialize_time()
        self._variables.initialize(self)

    def _step_one(self):
        """Advance the simulation a single step."""
        self._increment_time()
        self._variables.step(self._timestep)

    def _increment_time(self):
        """Advance time variables one time step."""
        self.TIME = self.TIME + self._timestep
        self.STEP = self.STEP + 1

    def treatments(self):
        """Return an iterators of the treatments.

        rtype: collections.Iterable[Treatment]
        """
        return self._treatments.values()

    def treatment(self, treatment_name):
        """
        Return a particular treatment from the model.



        :param str treatment_name: name of the treatment to be returned
        :return: the treatment named
        :rtype: Treatment
        :raises MinnetonkaError: if the model has no treatment of that name
        """
        try:
            return self._treatments[treatment_name]
        except KeyError:
            raise MinnetonkaError('Model has no treatment {}'.format(
                treatment_name))

    def variable(self, variable_name):
        """
        Return a single variable from the model, by name.

        A variable is typically accessed from a model by subscription, like a 
        dictionary value from a dictionary, e.g. ``modl['var']``. The
        subscription syntax is syntactic sugar for :meth:`variable`.

        Note that :meth:`variable` returns a variable object, not the current
        amount of the variable. To find the variable's current amount
        in a particular treatment, use a further subscription with the
        treatment name, e.g. ``modl['var']['']``. See examples below.

        Parameters
        ----------
        variable_name : str
            The name of the variable. The variable might be a plain variable,
            a stock, an accum, a constant, or any of the variable-like objects
            known by the model.

        Returns
        -------
        Variable : newly-defined variable with name ``variable_name``

        Raises
        ------
        MinnetonkaError
            If no variable named ``variable_name`` exists in the model

        Examples
        --------
        Create a model **m** with three variables, and only the default
        treatment.

        >>> with model() as m:
        ...     variable('Earnings', lambda r, c: r - c, 'Revenue', 'Cost')
        ...     variable('Cost', 10)
        ...     variable('Revenue', 12)

        Find the variable **Cost** ...

        >>> m.variable('Cost')
        __main__.Cost

        ... or use subscription syntax to do the same thing

        >>> m['Cost']
        __main__.Cost
        >>> m.variable('Cost') == m['Cost']
        True

        Find the current amount of **Cost** in the default treatment.

        >>> m['Cost']['']
        10
        """
        return self._variables.variable(variable_name)

    def add_variables(self, *variables):
        """Add the variables and everything they depend on.
        """
        logging.info('enter on variables {}'.format(variables))
        self._variables.add_variables(self, *variables)

    @classmethod
    def add_variable_to_current_context(cls, var_object):
        """If a context is currently open, add this variable object to it.

        :param Variable var_object: the variable object being added to the context
        """
        if cls._model_context is not None:
            cls._model_context._variables_not_yet_added.append(var_object)

    def _add_variables_and_initialize(self, *variables):
        """Add variables and initialize. The model may already be inited."""
        logging.info('enter on variables {}'.format(variables))
        self.add_variables(*variables)
        self.initialize()

    def _uninitialize(self):
        """Remove the effects of initializtion."""
        self._variables.uninitialize()
        self._initialize_time()

    def _initialize_time(self):
        """Set time variables to the beginning."""
        self.TIME = self._start_time
        self.STEP = 0

    

    def previous_step(self):
        """Return the prior value of STEP."""
        return self.STEP - 1

    def recalculate(self):
        """Recalculate all variables, without changing the step."""
        self._variables.recalculate()

    def variable_instance(self, variable_name, treatment_name):
        """Find or create right instance for this variable and treatment."""
        # A more pythonic approach than checking for this known string?
        if variable_name == '__model__':
            return self._pseudo_variable
        else:
            return self.variable(variable_name).by_treatment(treatment_name)


def model(variables=[], treatments=[''], initialize=True, timestep=1, 
          start_time=0, end_time=None):
    """
    Create and initialize a model, an instance of :class:`Model`

    A model is a collection of variables, with one or more treatments. A
    model can be simulated, changing the value of variables with each simulated
    step. 

    A model can be created via :meth:`Model`, after treatment objects have 
    been explicitly created. But typically this function
    is used instead, as it is simpler.

    A model sets a context, so variables can be defined for 
    the newly created model, as in the example below.

    Parameters
    ----------
    variables : list of :class:`Variable`, optional
        List of variables that are part of the model. If not specified,
        the default is [], no variables. An alternative to 
        creating the variables first, then the model, is to define the 
        variables within the model context, as in the example below. 
    treatments : list of str, or list of tuple of (str, str), optional
        List of treatment specs. Each treatment specs is a simulation scenario,
        simulated in parallel. Typical treatments might include 'As is', 
        'To be', 'At risk', 'Currently', With minor intervention', 
        etc. A treatment can be either a string---the name of the 
        treatment---or a tuple of two strings---the name and a short 
        description. See examples below.
        If not specified, the default is ``['']``, a single
        treatment named by the empty string. 
    initialize : bool, optional
        After the variables are added to the model, should all the variables
        be given their initial values? If more variables need to be added to 
        the model, wait to initialize. Default: True
    timestep : int, optional
        How much simulated time should elapse between each step? Default: 1 
        time unit
    start_time : int, optional
        At what time should the simulated clock start? Default: start at 0
    end_time : int, optional
        At what simulated time should the simulatation end? Default: None, 
        never end

    Returns
    -------
    Model
        the newly created model

    See Also
    --------
    :class:`Model` : a model, once created
    
    Examples
    --------
    Create a model with no variables and only the null treatment:

    >>> m = model()

    A model that defines two treatments:

    >>> model(treatments=['As is', 'To be'])

    One of the treatments has a description:

    >>> model(treatments=[('As is', 'The current situation'), 'To be'])

    A model with two variables:

    >>> m = model([DischargeBegins, DischargeEnds])

    Variables can be defined when the model is created:

    >>> m = model([
    ...         variable('Revenue', np.array([30.1, 15, 20])),
    ...         variable('Cost', np.array([10, 10, 10])),
    ...         variable('Earnings', lambda r, c: r - c, 'Revenue', 'Cost')
    ...    ])

    A model is a context, supporting variable addition:

    >>> with model() as m:
    ...  variable('Revenue', np.array([30.1, 15, 20]))
    ...  variable('Cost', np.array([10, 10, 10]))
    ...  variable('Earnings', lambda r, c: r - c, 'Revenue', 'Cost')
    """
    def _create_treatment_from_spec(spec):
        """Create treatment.

        Spec is either a name or a tuple of name and description.
        """
        try:
            name, description = spec
            return Treatment(name, description)
        except ValueError:
            return Treatment(spec)

    if end_time is not None and end_time < start_time:
        raise MinnetonkaError('End time {} is before start time {}'.format(
            end_time, start_time))
    m = Model(
        [_create_treatment_from_spec(spec) for spec in treatments], timestep,
        start_time, end_time)
    m.add_variables(*variables)
    if initialize and variables:
        m.initialize()
    return m


class ModelVariables:
    """Manage the ordered list of variables of a model."""

    def __init__(self):
        """Initialize the model variables."""
        self._variables = {}
        self._is_ordered = False

    def _variable_iterator(self):
        """Return an iterator over variables."""
        return self._variables.values()

    def _varirable_name_iterator(self):
        """Return an iterator over variable names."""
        return self._variables.keys()

    def add_variables(self, model, *variables):
        """Add the list of variables."""
        logging.info('enter with variables {}'.format(variables))
        assert not self._is_ordered, (
            'Cannot add variables {} after the variables are ordered').format(
                variables)
        for var in variables:
            self._add_single_variable(model, var)

    def _add_single_variable(self, model, var):
        """Add a variable to the model variables."""
        logging.info('enter with variable {}'.format(var))
        if var.name() in self._variables:
            warnings.warn(
                'Variable {} redefined'.format(var.name()), MinnetonkaWarning)
        self._variables[var.name()] = var
        var.note_model(model)

    def variable(self, variable_name):
        """Return the variable with variable_name, if it exists."""
        try:
            return self._variables[variable_name]
        except AttributeError:
            try:
                return self._variables_ordered_for_init[variable_name]
            except KeyError:
                raise MinnetonkaError(
                    'Unknown variable {}'.format(variable_name))
        except KeyError:
            raise MinnetonkaError('Unknown variable {}'.format(variable_name))

    def initialize(self, model):
        """Initialize the variables of the simulation."""
        logging.info('enter')
        self._check_for_cycles(model)
        self._label_taries() 
        self._create_all_variable_instances()
        self._wire_variable_instances(model)
        self._sort_variables()
        self._set_initial_amounts()
        logging.info('exit')
        
    def _check_for_cycles(self, model):
        """Check for any cycle among variables, raising error if necessary."""
        logging.info('enter')
        variables_seen = []
        for variable in self._variable_iterator(): 
            if variable not in variables_seen:
                variable.check_for_cycle(variables_seen)

    def _label_taries(self):
        """Label every model variable as either unitary or multitary."""
        self._label_tary_initial()
        self._label_multitary_succedents()
        self._label_unknowns_unitary()

    def _label_tary_initial(self):
        """Label the tary of model variables, with some unknown."""
        for var in self._variable_iterator():
            if not var.has_unitary_definition():
                var.set_tary('multitary')
            elif var.antecedents(ignore_pseudo=True) == []:
                var.set_tary('unitary')
            else: 
                var.set_tary('unknown')

    def _label_multitary_succedents(self):
        """Label all succedents of multitary variables as multitary."""
        succedents = self._collect_succedents()
        multitaries = [v for v in self._variable_iterator() 
                       if v.tary() == 'multitary']
        for var in multitaries:
            self._label_all_succedents_multitary(var, succedents)

    def _collect_succedents(self):
        """Return dict of succedents of each variable."""
        succedents = {v: set([]) for v in self._variable_iterator()}
        for var in self._variable_iterator():
            for ante in var.antecedents(ignore_pseudo=True):
                succedents[ante].add(var)
        return succedents

    def _label_all_succedents_multitary(self, var, succedents):
        """Label all succedents (and their succedents) or var as multitary."""
        var.set_tary('multitary')
        for succ in succedents[var]:
            if succ.tary() == 'unknown':
                self._label_all_succedents_multitary(succ, succedents)

    def _label_unknowns_unitary(self):
        """Label every unknown variable as unitary."""
        for v in self._variable_iterator():
            if v._tary == 'unknown':
                v.set_tary('unitary')

    def _create_all_variable_instances(self):
        """Create all variable instances."""
        logging.info('enter')
        for variable in self._variable_iterator():
            variable.create_variable_instances()

    def _wire_variable_instances(self, model):
        """Provide each of the var instances with its antecedent instances."""
        logging.info('enter') 
        for variable in self._variable_iterator():
            variable.wire_instances()

    def _sort_variables(self):
        """Sort the variables from dependent to independent, twice.

        Create two sorted lists, one for init and the other for step.
        They are identical, except for the effect of accums and stock and
        previous.
        """
        logging.info('enter')
        self._variables_ordered_for_init = self._sort_variables_for(
            for_init=True)
        self._variables_ordered_for_step = self._sort_variables_for(
            for_init=False)
        self._is_ordered = True

    def _sort_variables_for(self, for_init=False):
        """Sort the variables from dependent to independent."""
        ordered_variables = collections.OrderedDict()

        def _maybe_insert_variable_and_antes(variable_name, already_seen):
            """Insert the variable and its antecedents if they do exist."""
            if variable_name in already_seen:
                pass
            elif (variable_name not in ordered_variables):
                var = self.variable(variable_name)
                for ante in var.depends_on(
                        for_init=for_init, for_sort=True, ignore_pseudo=True):
                    _maybe_insert_variable_and_antes(
                        ante, [variable_name] + already_seen)
                ordered_variables[variable_name] = var

        for variable in self._variable_iterator():
            _maybe_insert_variable_and_antes(variable.name(), list())
        return ordered_variables

    def _set_initial_amounts(self):
        """Set initial amounts for all the variables."""
        logging.info('enter')
        for var in self._variables_ordered_for_init.values():
            var.set_all_initial_amounts()
        logging.info('exit')

    def uninitialize(self):
        """Undo the initialization, typically to add more variables."""
        self._is_ordered = False
        self._delete_existing_variable_instances()

    def _delete_existing_variable_instances(self):
        """Delete any variable instances that were previouslsy created."""
        for variable in self._variable_iterator():
            variable.delete_all_variable_instances()

    def reset(self, reset_external_vars):
        """Reset variables.

        If reset_external_vars is false, don't reset the external variables,
        those whose value has been set outside the model itself.
        """
        for var in self._variables_ordered_for_init.values():
            var.reset_all(reset_external_vars)

    def step(self, timestep):
        """Advance all the variables one step in the simulation."""
        for var in self._variables_ordered_for_step.values():
            var.calculate_all_increments(timestep)
        for var in self._variables_ordered_for_step.values():
            var.step_all()

    def recalculate(self):
        """Recalculate all the variables without advancing step."""
        for var in self._variables_ordered_for_step.values():
            var.recalculate_all()


class Treatment:
    """A treatment applied to a model."""

    def __init__(self, name, description=None):
        """Initialize this treatment."""
        self.name = name
        self.description = description
        self._variables = {}

    def __repr__(self):
        """Print text representation of this treatment."""
        if self.description is None:
            return "Treatment('{}')".format(self.name)
        else:
            return "Treatment('{}', '{}')".format(self.name, self.description)

    def addVariable(self, newvar):
        """Add the variable to this list of variables."""
        self._variables[newvar.name()] = newvar

    def remove_variable(self, var_to_remove):
        """Remove this variable."""
        del self._variables[var_to_remove.name()]

    def __getitem__(self, key):
        """Return the variable with name of key."""
        try:
            return self._variables[key]
        except KeyError as ke:
            raise MinnetonkaError('{} has no variable {}'.format(self, key))


def treatment(spec):
    """Create a new treatment, with the specification."""
    return Treatment(spec)


def treatments(*treatment_names):
    """Create a bunch of treatments, and return them as a tuple."""
    return tuple(
        Treatment(treatment_name) for treatment_name in treatment_names)

#
# Variable classes
#


class CommonVariable(type):
    """The common superclass for all Minnetonka variables and variable-like things."""

    def __getitem__(self, treatment_name):
        """Get [] for this variable."""
        return self.by_treatment(treatment_name).amount()

    def __setitem__(self, treatment_name, amount):
        """Set [] for this variable."""
        if treatment_name == '__all__':
            return self.set_amount_all(amount)
        elif self.is_unitary():
            warnings.warn(
                'Setting amount of unitary variable {} '.format(self.name()) +
                'in only one treatment',
                MinnetonkaWarning)
            self.set_amount_all(amount)
        else:
            return self.by_treatment(treatment_name).set_amount(amount)

    def __repr__(self):
        return "{}('{}')".format(self._kind().lower(), self.name())

    def __str__(self):
        return "<{} {}>".format(self._kind(), self.name())

    def _kind(self):
        """'Variable' or 'Stock' or 'Accum' or whatever."""
        return type(self).__name__

    def create_variable_instances(self):
        """Create variable instances for this variable."""
        if self.is_unitary():
            v = self()
            for treatment in self._model.treatments():
                v._initialize_treatment(treatment)
        else:
            for treatment in self._model.treatments():
                self(treatment)

    # To do: use property
    
    def tary(self):
        """Is the variable unitary or not (i.e. multitary)?"""
        return self._tary 

    def set_tary(self, tary_value):
        """Set the taryness of the variable."""
        self._tary = tary_value 

    def is_unitary(self):
        """Returns whether variable is unitary: same across all treatments.

        Some variables always take the same value across all treatments.
        Is this variable one of those unitary variables?

        :return: whether the variable is unitary
        :rtype: boolean
        """
        return self.tary() == 'unitary'

    def note_model(self, model):
        """Keep track of the model, for future reference."""
        self._model = model

    def by_treatment(self, treatment_name):
        """Return the variable instance associated with this treatment."""
        try:
            return self._by_treatment[treatment_name]
        except (KeyError, AttributeError):
            raise MinnetonkaError(
                'Variable {} not initialized with treatment {}'.format(
                    self.name(), treatment_name))

    def all_instances(self):
        """Return all the instances of this variable."""
        if self.is_unitary():
            return itertools.islice(self._by_treatment.values(), 0, 1) 
        else:
            return self._by_treatment.values()

    def set_all_initial_amounts(self):
        """Set the initial amounts of all the variable instances."""
        if self.is_unitary():
            treatment_name, var = list(self._by_treatment.items())[0]
            var._set_initial_amount(treatment_name)
        else:
            for treatment_name, var in self._by_treatment.items():
                var._set_initial_amount(treatment_name)

    def reset_all(self, reset_external_vars):
        """Reset this variable to its initial amount.

        Reset all variable instances of this variable class to their initial
        amounts. But maybe don't reset the variables set externally, depending
        on the value of reset_external_vars
        """
        for var in self.all_instances():
            var._clear_history()
            var._reset(reset_external_vars)

    def step_all(self):
        """Advance all the variable instances one step."""
        for var in self.all_instances():
            var._record_current_amount()
            var._step()

    def recalculate_all(self):
        """Recalculate all the variable instances, without changing step."""
        for var in self._by_treatment.values():
            var._recalculate()

    def calculate_all_increments(self, ignore):
        """Ignore this in general. Only meaningful for stocks."""
        pass

    def set_amount_all(self, amount):
        """Set the amount for all treatments."""
        for var in self.all_instances():
            var.set_amount(amount)

    def delete_all_variable_instances(self):
        """Delete all variables instances."""
        for v in self.all_instances():
            v._treatment.remove_variable(v)
        self._by_treatment = {}

    def history(self, treatment_name, step):
        """Return the amount at a particular timestep."""
        return self.by_treatment(treatment_name)._history(step)

    def wire_instances(self):
        """For each instance of this variable, set the vars it depends on."""
        for treatment in self._model.treatments():
            self.by_treatment(treatment.name).wire_instance(
                self._model, treatment.name)

    def check_for_cycle(self, checked_already, dependents=None):
        """Check for cycles involving this variable."""
        if self in checked_already:
            return
        elif dependents is None:
            dependents = []
        elif self in dependents:
            varnames = [d.name() for d in dependents] + [self.name()]
            raise MinnetonkaError('Circularity among variables: {}'.format(
                ' <- '.join(varnames)))

        dependents = dependents + [self]
        self._check_for_cycle_in_depends_on(checked_already, dependents)
        checked_already.append(self)

    def all(self):
        """Return a dict of all amounts, one for each treatment."""
        return {tmt: inst.amount() for tmt, inst in self._by_treatment.items()}

    def show(self):
        """Show everything important about the variable."""
        self._show_name()
        self._show_doc()
        self._show_amounts()
        self._show_definition_and_dependencies()
        return self.antecedents()

    def _show_name(self):
        """Print the variable type and name."""
        bold = '\033[1m'; endbold = '\033[0m'
        print('{}{}: {}{}\n'.format(bold, self._kind(), self.name(), endbold))

    def _show_doc(self):
        """Show the documentation of the variable, if any."""
        try:
            if self.__doc__:
                print(self.__doc__)
                print()
        except:
            pass

    def _show_amounts(self):
        """Show the amounts for all the instances of the variable."""
        print('Amounts: {}\n'.format(self.all()))

class CommonVariableInstance(object, metaclass=CommonVariable):
    """
    Any of the variety of variable types.
    """

    def __init__(self, treatment=None):
        """Initialize this variable."""
        self._extra_model_amount = None
        self._clear_history()
        if treatment is not None:
            self._initialize_treatment(treatment)

    def _initialize_treatment(self, treatment):
        """Do all initialization regarding the treatment."""
        self._treatment = treatment
        treatment.addVariable(self)
        if hasattr(type(self), '_by_treatment'):
            type(self)._by_treatment[treatment.name] = self
        else:
            type(self)._by_treatment = {treatment.name: self}

    @classmethod
    def name(cls):
        """Return the name of the variable."""
        return cls.__name__

    def amount(self):
        """Return the current value of amount."""
        if self._extra_model_amount is None:
            return self._amount
        else:
            return self._extra_model_amount

    def treatment(self):
        """Return the treatment of the variable instance."""
        return self._treatment

    def _clear_history(self):
        """Clear the stepwise history of amounts from the variable."""
        self._old_amounts = []

    def _record_current_amount(self):
        """Record the current amount of the variable, prior to a step."""
        self._old_amounts.append(self._amount)

    def _history(self, step):
        """Return the amount at timestep step."""
        if step == len(self._old_amounts):
            return self.amount()
        elif step == -1:
            return None
        try:
            return self._old_amounts[step]
        except IndexError:
            raise MinnetonkaError("{}['{}'] has no value for step {}".format(
                self.name(), self.treatment().name, step))

    def previous_amount(self):
        """Return the amount in the previous step."""
        previous = self._model.previous_step()
        try:
            return self._history(previous)
        except:
            return None


class SimpleVariableInstance(CommonVariableInstance):
    """A variable that is not an incrementer."""

    def _reset(self, reset_external_vars):
        """Reset to beginning of simulation."""
        if reset_external_vars or self._extra_model_amount is None:
            self._set_initial_amount()

    def _step(self):
        """Advance this simple variable one time step."""
        self._amount = self._calculate_amount()

    def _recalculate(self):
        """Recalculate this simple variagble."""
        self._amount = self._calculate_amount()

    def _set_initial_amount(self, treatment=None):
        """Set the step 0 amount for this simple variable."""
        logging.info('setting initial amount for simple variable {}'.format(
            self))
        self._amount = self._calculate_amount()
        self._extra_model_amount = None

    def set_amount(self, amount):
        """Set an amount for the variable, outside the logic of the model."""
        self._extra_model_amount = amount

    @classmethod
    def depends_on(cls, for_init=False, for_sort=False, ignore_pseudo=False):
        """Return variable names this one depends on.

        :param for_init: return only the variables used in initialization
        :param for_sort: return only the variables relevant for sorting vars
        :param ignore_pseudo: do not return names of pseudo-variables
        :return: list of all variable names this variable depends on
        """
        # ignore for_init and for_sort since behavior is the same for simple variable
        return cls._calculator.depends_on(ignore_pseudo)


class Variable(CommonVariable):
    """A variable whose amount is calculated from amounts of other variables."""

    def _check_for_cycle_in_depends_on(self, checked_already, dependents):
        """Check for cycles among the depends on for this plain variable."""
        for dname in self.depends_on(ignore_pseudo=True):
            d = self._model.variable(dname)
            d.check_for_cycle(checked_already, dependents=dependents)

    def _show_definition_and_dependencies(self):
        """Print the definition and the variables it depends on."""
        print('Definition: {}'.format(self._calculator.serialize_definition()))
        print('Depends on: {}'.format(self.depends_on()))

    def antecedents(self, ignore_pseudo=False):
        """Return all the depends_on variables."""
        m = self._model
        return [m[v] for v in self.depends_on(ignore_pseudo=ignore_pseudo)]

    def has_unitary_definition(self):
        """Returns whether the variable has a unitary definition."""
        return self._calculator.has_unitary_definition()


class VariableInstance(SimpleVariableInstance, metaclass=Variable):
    """
    A variable whose amount is calculated from the amounts of other variables.

    """

    def _calculate_amount(self):
        """Calculate the current amount of this plain variable."""
        try:
            calculator = self._calculator
        except AttributeError:
            raise MinnetonkaError(
                'Variable {} needs to define how to calculate'.format(
                    type(self).name()))
        try:
            treatment = self._treatment
        except AttributeError:
            raise MinnetonkaError(
                'Variable {} needs to define its treatment'.format(
                    type(self).name()))
        try:
            depends_on = self._depends_on_instances
        except AttributeError:
            raise MinnetonkaError(
                'Variable {} needs to define what it depends on'.format(
                    type(self).name()))
        try:
            return calculator.calculate(
                treatment.name, [v.amount() for v in depends_on])
        except KeyError:
            raise MinnetonkaError('Treatment {} not defined for {}'.format(
                self._treatment.name, type(self).name()))
        except:
            print('Error in calculating amount of {}'.format(self))
            raise

    def wire_instance(self, model, treatment_name):
        """Set the variables this instance depends on."""
        self._depends_on_instances = [
            model.variable_instance(v, treatment_name)
            for v in self.depends_on()]




class Calculator:
    """Calculate amounts based on either lambdas or treatments or constants."""

    def __init__(self, definition, depends_on_var_names):
        """Initialize calculator."""
        self._definition = definition
        self._depends_on_var_names = depends_on_var_names

    def calculate(self, treatment_name, depends_on_amounts):
        """Calculate amount of thing."""
        # could optimize by doing this all only once
        try:
            defn = self._definition.by_treatment(treatment_name)
        except (KeyError, TypeError, AttributeError):
            defn = self._definition

        # must use callable() because foo() raises a TypeError exception
        # under two circumstances: both if foo is called with the wrong
        # number of arguments and if foo is not a callable
        if callable(defn):
            return defn(*depends_on_amounts)
        else:
            return defn

    def depends_on(self, ignore_pseudo=False):
        """Return variables this calculator depends on.

        :param ignore_pseudo: do not return names of pseudo-variables
        :return: list of all variable names this calculator depends on
        """
        return [v for v in self._depends_on_var_names 
                if v != '__model__' or ignore_pseudo is False ]

    def serialize_definition(self):
        """Return the serialization of the the definition of this calculator"""
        try:
            return self._definition.serialize_definition()
        except:
            try:
                return self.source()
            except:
                return self._definition 

    def source(self):
        """Return source of how this is calculated."""
        src = inspect.getsource(self._definition)
        src = src.strip()
        return self._remove_trailing_comma(src)

    def _remove_trailing_comma(self, src):
        """Remove trailing comma, if present"""
        return re.sub(',\s*\Z', '', src)
        # src = re.sub("\A'.*',\s*lambda", 'lambda', src, count=1)
        # src = re.sub('\A".*",\s*lambda', 'lambda', src, count=1)
        return src 

    def has_unitary_definition(self):
        """Is the definition of this calculator unitary?"""
        try:
            self._definition.treatments_and_values()
            return False
        except AttributeError:
            return True 


class ModelPseudoVariable():
    """Special variable for capturing the model."""

    def __init__(self, m):
        """Initialize pseudo-variable."""
        self._model = m

    def amount(self):
        """Return current amount of the pseudo-variable."""
        return self._model

    @classmethod
    def check_for_cycle(cls, checked_already, dependents=None):
        """Check for cycles involving this variable."""
        pass


#
# Defining variables
#


def variable(variable_name, *args):
    """
    Create a plain variable.

    A variable has a value---called an 'amount'---that changes over simulated 
    time. A single
    variable can take a different amount in each model treatment. The amount 
    of a variable can be any Python object, except a string or a Python 
    callable. A variable can be defined in terms of other variables, via a 
    Python callable.

    A simple variable differs form other variable-like objects (e.g.
    stocks) in that it keeps no state. Its amount depends entirely on its 
    definition, and the amounts of other variables used in the definition.

    Parameters
    ----------
    variable_name : str
        Name of the variable. The name is unique within a single model.

    args
        Either a list of a callable and the names of variables, or a
        Python object that is neither a string nor a callable. In either case,
        the first element of args is an optional docstring-like description.
        See examples below.

    Returns
    -------
    Variable
        the newly-created variable

    See Also
    --------
    constant : Create a variable whose amount does not change

    stock : Create a system dynamics stock

    :class:`Variable` : a simple variable, once created

    :class:`PerTreatment` : for defining how a variable has a different amount
        for each treatment

    Examples
    --------
    A variable with a constant amount. (An alternate approach
    is to define this with :func:`constant`.)

    >>> variable('DischargeBegins', 12)

    The amount can be any non-callable, non-string Python object.

    >>> Revenue = variable('Revenue', np.array([30.1, 15, 20]))
    
    >>> Cost = variable('Cost', {'drg001': 1000, 'drg003': 1005})

    A variable can have an optional docstring-like description.

    >>> Revenue = variable('Revenue',
    ...     '''The revenue for each business unit, as a 3 element array''',
    ...     np.array([30.1, 15, 20]))

    A variable can have a different constant amount for each treatment.

    >>> DischargeEnds = variable('DischargeEnds',
    ...     PerTreatment({'As is': 20, 'To be': 18}))

    A variable can take a different value every timestep, via an 
    expression wrapped in a lambda ...

    >>> RandomValue = variable('RandomValue', lambda: random.random() + 1)

    ... or via any Python callable.

    >>> RandomValue = variable('RandomValue', random.random)

    The expression can depend on another variable in the model ...

    >>> DischargeProgress = variable(
    ...     'DischargeProgress', lambda db: (current_step - db) / 4,
    ...     'DischargeBegins')

    ... or depend on multiple variables.

    >>> Earnings = variable('Earnings', lambda r, c: r - c, 'Revenue', 'Cost')

    A variable can use different expressions in different treatments ...

    >>> DischargeEnds = variable('DischargeEnds',
    ...     PerTreatment(
    ...         {'As is': lambda db: db + 10, 'To be': lambda db: db + 5}),
    ...         'DischargeBegins')

    ... or an expression in one treatment and a constant in another.

    >>> MortalityImprovement = variable('MortailityImprovement',
    ...     PerTreatment({'Value at risk': lambda x: x, 'Value expected': 0}),
    ...     'MortalityImprovementViaRRC')

    An expression can use something in the model itself.

    >>> Step = variable('Step', lambda md: md.TIME, '__model__')
    """
    logging.info('Creating variable %s', variable_name)
    return _parse_and_create(variable_name, VariableInstance, 'Variable', args)


def _parse_and_create(name, variable_class, create_what, args):
    """Parse args and create variable."""
    if len(args) == 0:
        raise MinnetonkaError('{} {} has no definition'.format(
            create_what, name))
    if len(args) == 1:
        return _create_variable(name, variable_class, args[0])
    elif isinstance(args[0], str):
        docstring, definition, *depends_on_variables = args
        return _create_variable(
            name, variable_class, definition, docstring=docstring,
            depends_on_variables=depends_on_variables)
    else:
        definition, *depends_on_variables = args
        return _create_variable(name, variable_class, definition,
                                depends_on_variables=depends_on_variables)


def _create_variable(
        variable_name, variable_class, definition, docstring='',
        depends_on_variables=()):
    """Create a new variable of this name and with this definition."""
    calc = create_calculator(definition, depends_on_variables)
    newvar = type(variable_name, (variable_class,),
                  {'__doc__': docstring, '_calculator': calc})
    Model.add_variable_to_current_context(newvar)
    return newvar


def create_calculator(definition, variables):
    """Create a new calculator from this definition and variables.

    definition = either a constant or a callable or constants that
                 vary by treatment or callables that vary by treatment
    variables = list of strings, each one a variable name
    """
    if variables is None:
        variables = []
    assert is_list_of_strings(variables), \
        ('Attempted to create calculator with something other than a' +
         ' list of strings: {}'.format(variables))
    return Calculator(definition, variables)


def is_list_of_strings(arg):
    """Return whethr arg is list of tuple of strings."""
    return is_sequence(arg) and is_all_strings(arg)


def is_sequence(arg):
    """Return whether arg is list or tuple or some other sequence."""
    return (not hasattr(arg, "strip") and
            hasattr(arg, "__getitem__") or
            hasattr(arg, "__iter__"))


def is_all_strings(arg):
    """Return whether arg is iterable, are all the elements strings."""
    # unpythonic
    return all(isinstance(elt, str) for elt in arg)


class PerTreatment:
    """Indicate different values in each treatment.

    Initialized with dict of treatments and values, e.g.
    PerTreatment({'As is': 12, 'To be': 9})
    """

    def __init__(self, treatments_and_values):
        """Initialize PerTreatment."""
        self._treatments_and_values = treatments_and_values

    def treatments_and_values(self):
        """Return all treatments and values, as a dict."""
        return self._treatments_and_values

    def by_treatment(self, treatment_name):
        """Return definition associate with treatment name."""
        try:
            return self._treatments_and_values[treatment_name]
        except KeyError:
            raise MinnetonkaError("Treatment '{}' not defined".format(
                treatment_name))

    def serialize_definition(self):
        """Return the serialization of the definition of this calculator."""
        return 'PerTreatment({{{}}})'.format(', '.join(map(
            lambda k, v: self._serialize_treatment(k, v), 
            self._treatments_and_values.keys(), 
            self._treatments_and_values.values())))

    def _serialize_treatment(self, k, v):
        """Serialize the key snd value so it can be an item in a larger dict."""
        try:
            return '"{}": {}'.format(k, v.serialize_definition())
        except:
            return '"{}": {}'.format(k, v)

    def any_callable(self):
        """Is the value callable in any treatment?"""


#
# Defining constants
#


def constant(constant_name, *args):
    """
    Create a variable whose amount does not vary.

    A constant is a variable that does not vary. Its amount is set on
    initialization, and then does not change over the course of the 
    simulation. When the simulation is reset, the constant can be given a
    new amount. 

    A single constant can take a different amount in each
    model treatment. The amount of a constant in a particular treatment
    can be found using the subscription brackets, e.g. **Interest['to be']**.
    See examples below.

    The amount of a constant can be any Python object, except a string or
    a Python callable. It can be defined in terms of other variables, 
    using a callable in the definition. See examples below.

    The amount of a constant can be changed explicitly, outside the model
    logic, e.g. **Interest['to be'] = 0.07**. Once changed, the amount of
    the constant remains the same, at least until the simulation is reset
    or the amount is again changed explicitly. See examples below.

    Parameters
    ----------
    constant_name : str
        Name of the constant. The name is unique within a single model.

    args 
        Either a list of a single Python object (that is neither a string 
        nor a callable),
        or a list of a callable and the names of variables. In either case,
        the first element of args is an optional docstring-like description.

    Returns
    -------
    Variable
        the newly-created constant

    See Also
    --------
    variable : Create a plain variable whose amount changes

    stock : Create a system dynamics stock

    :class:`PerTreatment` : for defining how a constant takes different
        (constant) amounts for each treatment 

    Examples
    --------
    Create a constant without a description.

    >>> constant('KitchenCloses', 22)

    Create constant with a description.

    >>> constant('KitchenCloses', 
    ...     '''What time is the kitchen scheduled to close?''',
    ...     22.0)

    Create a constant whose amount is a Python dictionary.

    >>> constant('KitchenCloses', 
    ...     {'Friday': 23.5, 'Saturday': 23.5, 'Sunday': 21.0, 
    ...      'Monday': 22.0, 'Tuesday': 22.0, 'Wednesday': 22.0, 
    ...      'Thursday': 22.5})

    Create a constant whose amount differs by treatment.

    >>> constant('KitchenOpens', 
    ...     PerTreatment({'As is': 17.0, 'Open early': 15.5}))

    Create a constant whose (non-varying) amount is calculated from other 
    variables.

    >>> constant('KitchenDuration',
    ...     '''How many hours is the kitchen open each day?''',
    ...     lambda c, o: c - o, 'KitchenCloses', 'KitchenOpens')
    
    Create a model with one variable and two constants.

    >>> import random
    >>> with model() as m:
    ...     C1 = constant('C1', lambda: random.randint(0, 9)
    ...     V = variable('V', lambda: random.randint(0, 9))
    ...     C2 = constant('C2', lambda v: v, 'V')

    **C2** and **V** have the same amount, a random integer between 0 and 9.
    **C1** has a different amount.

    >>> V['']
    2
    >>> C2['']
    2
    >>> C1['']
    0

    The simulation is advanced by one step. The variable **V** has a new 
    amount, but the constants **C1** and **C2** remain the same.

    >>> m.step()
    >>> V['']
    7
    >>> C2['']
    2
    >>> C1['']
    0

    The simulation is reset. Now **C2** and **V** have the same value, again.

    >>> m.reset()
    >>> V['']
    6
    >>> C2['']
    6
    >>> C1['']
    8

    The amount of **C2** is changed, outside of the model logic.

    >>> C2[''] = 99
    >>> C2[''] 
    99

    **C2** still stays constant, even after a step.

    >>> m.step()
    >>> C2['']
    99

    But on reset, **C2**'s amount is once again changed to **V**'s amount.

    >>> m.reset()
    >>> V['']
    1
    >>> C2['']
    1

    Show the details of **C2**

    >>> C2.show()
    Constant: C1
    Amounts: {'': 6}
    Definition: C1 = constant('C1', lambda: random.randint(0, 9))
    Depends on: []

    """
    logging.info('Creating constant %s', constant_name)
    return _parse_and_create(constant_name, ConstantInstance, 'Constant', args)

class Constant(Variable):
    pass

class ConstantInstance(VariableInstance, metaclass=Constant):
    """A variable that does not vary."""

    def _step(self):
        pass

    def _recalculate(self):
        pass

    def _history(self, _):
        """No history for a constant. Everything is the current value."""
        return self.amount()

    def _clear_history(self):
        """No history for a constant. Everything is the current value."""
        pass

    def _record_current_amount(self):
        """No history for a constant. Everything is the current value."""
        pass

    def previous_amount(self):
        """No history for a constant. Everything is the current value."""
        return self.amount()




#
# Stock classes
#

class Incrementer(Variable):
    """A variable with internal state, that increments every step."""
    def _show_definition_and_dependencies(self):
        """Print the definitions and variables it depends on."""
        print('Initial definition: {}'.format(
            self._initial.serialize_definition()))
        print('Initial depends on: {}\n'.format(self._initial.depends_on()))
        print('Incremental definition: {}'.format(
            self._incremental.serialize_definition()))
        print('Incremental depends on: {}'.format(
            self._incremental.depends_on()))

    def antecedents(self, ignore_pseudo=False):
        """Return all the depends_on variables."""
        all_depends = list(dict.fromkeys(
            list(self._initial.depends_on(ignore_pseudo=ignore_pseudo)) + 
            list(self._incremental.depends_on(ignore_pseudo=ignore_pseudo))))
        return [self._model[v] for v in all_depends]

    def has_unitary_definition(self):
        """Returns whether the variable has a unitary definition."""
        return (self._initial.has_unitary_definition() and
                self._incremental.has_unitary_definition())

class IncrementerInstance(CommonVariableInstance, metaclass=Incrementer):
    """A variable instance with internal state, that increments every step."""

    def _reset(self, external_vars):
        """Reset to beginning of simulation."""
        self._set_initial_amount(self._treatment.name)

    def _set_initial_amount(self, treatment_name):
        """Set the initial amount of the incrementer."""
        msg = 'setting initial amount for incrementer {}, treatment {}'.format(
                self, treatment_name)
        logging.info(msg)
        try:
            self._amount = copy.deepcopy(self._initial.calculate(
                treatment_name,
                [v.amount() for v in self._initial_depends_on_instances]))
        except:
            print('Error while {}'.format(msg))
            raise

    def set_amount(self, new_amount):
        """Set a new amount, outside the logic of the model."""
        self._amount = new_amount

    def _recalculate(self):
        """Recalculate without advancing a step."""
        # For incrementer recalcs only happen on increment time
        pass

    def wire_instance(self, model, treatment_name):
        """Set the variables this instance depends on."""
        self._initial_depends_on_instances = [
            model.variable_instance(v, treatment_name)
            for v in self.depends_on(for_init=True)]
        self._increment_depends_on_instances = [
            model.variable_instance(v, treatment_name)
            for v in self._incremental.depends_on()]

class Stock(Incrementer):
    """A system dynamics stock."""

    def calculate_all_increments(self, timestep):
        """Compute the increment for all stock variable instances."""
        for var in self.all_instances():
            var._calculate_increment(timestep)

    def _check_for_cycle_in_depends_on(self, checked_already, dependents=None):
        """Check for cycles involving this stock."""
        # Note stocks are fine with cycles involving the incr calculator"""
        for dname in self.depends_on(for_init=True):
            d = self._model.variable(dname)
            d.check_for_cycle(checked_already, dependents=dependents)

class StockInstance(IncrementerInstance, metaclass=Stock):
    """A instance of a system dynamics stock for a particular treatment."""

    def _calculate_increment(self, timestep):
        """Compute the increment."""
        full_step_incr = self._incremental.calculate(
            self._treatment.name,
            [v.amount() for v in self._increment_depends_on_instances])
        self._increment_amount = full_step_incr * timestep

    def _step(self):
        """Advance the stock by one step."""
        self._amount = self._amount + self._increment_amount

    @classmethod
    def depends_on(cls, for_init=False, for_sort=False, ignore_pseudo=False):
        """Return the variables this stock depends on.

        :param for_init: return only the variables used in initialization
        :param for_sort: return only the variables relevant for sorting vars
        :param ignore_pseudo: do not return names of pseudo-variables
        :return: list of all variable names this variable depends on
        """
        if for_init:
            return cls._initial.depends_on(ignore_pseudo)
        elif for_sort:
            return []
        else:
            return cls._incremental.depends_on(ignore_pseudo)


#
# Defining stocks
#


def stock(stock_name, *args):
    """
    Create a system dynamics stock.

    In `system dynamics <https://en.wikipedia.org/wiki/System_dynamics>`_,
    a stock is used to model something that accumulates or depletes over
    time. The stock defines both an initial amount and an increment.

    At any simulated period, the stock has an amount. The amount changes
    over time, incrementing or decrementing at each period. The amount
    can be a simple numeric like a Python integer or a Python float. 
    Or it might be some more complex Python object: a list,
    a tuple, a numpy array, or an instance of a user-defined class. In 
    any case, the stock's amount must support addition.

    The stock may have several amounts, one for each treatment in the
    model of which it lives. 

    A stock definition has two parts: an initial amount and an increment.
    The initial amount is defined by either a Python object, or by
    a callable and a list of dependency variables. If defined by a Python 
    object, any Python object can be used, except for a string or a 
    callable. 

    If the initial amount is defined by a callable and list of variables,
    that callable is called a single time, at stock initialization, with the 
    initial amounts of each of the dependency variables. The names of
    the dependency variables are provided, i.e. the list is a list of strings.
    Each dependency
    variable named can either be a plain variable or a stock or a constant
    or any of the other variable elements of a model. The result of the 
    execution of the callable becomes the initial amount of the stock.

    The stock increment is also defined by either a Python object, or by
    a callable and a list of dependency variables. If defined by a Python
    object, any Python object can be used, except for a string or a 
    callable. 

    If the increment is a callable, the callable is called once every
    period, with the amounts of each of the dependency variables. Each 
    dependency variable
    can be a plain variable or a stock or a constant or any of the 
    elements of a model. The callable is given the amounts of the 
    variables at the previous period, not the current period, to 
    determine the increment of the stock for this period. 

    The increment is how much the stock's amount changes in each unit of
    time. If the timestep of the model is 1.0, the stock's amount will
    change by increment. If the timestep is not 1.0, the stock's amount
    will change by a different quantity. For example, if the timestep 
    is 0.5, the stock's amount will change by half the increment, at
    every step.

    The initial amount and the increment may vary by treatment, either
    because one or more of the the dependency variables vary by treatment,
    or because of an explicit :class:`PerTreatment` expression. 

    Parameters
    ----------
    stock_name : str
        Name of the stock. The name must be unique within a single model.

    args
        The args might include an optional docstring-like description, at
        the beginning. The interpretation of the remaining args depends on
        their count. 

        If there is only a single argument (aside from the
        optional description), it is the non-callable
        definition of the increment. The stock is assumed to be initialized
        at zero.

        If there are only two arguments (aside from the description),
        the first is the non-callable definition of the increment, and the
        second is the non-callable definiton of the initialization.

        If there are three arguments (aside from description):

        #. the first argument is a callable definition of the increment, executed every period

        #. the second argument is a tuple of names of dependency variables, whose amounts will be used as arguments of the increment callable. The tuple may be empty (i.e. **()**), if the increment callable takes no arguments.
        
        #. the third argument is a Python object, the non-callable definition of the initialization

        If there are four arguments (sans descriptions):

        #. the first argument is a callable definition of the increment, executed every period

        #. the second argument is a tuple of names of dependency variables, whose amounts will be used as arguments of the increment callable. The tuple may be empty (i.e. **()**), if the increment callable takes no arguments.
        
        #. the third argument is a callable definition of the initialization, executed once, at at initialization, and again each time the simulation is reset

        #. the fourth argument is a tuple of names of dependency variables, whose amounts are used as arguments of the initialization callable. The tuple may be empty (i.e. **()**), if the initialization callable takes no arguments.

        See usage examples below.

    Returns
    -------
    Stock
        the newly-created stock

    See Also
    --------
    variable : Create a non-stock plain variable

    constant : Create a plain variable whose amount does not change

    accum : Create an accum, much like a stock except that it uses the 
        amounts of the variables in the current period, instead of the 
        previous period.

    :class:`PerTreatment` : for defining how an increment or initialization
        varies from treatment to treatment

    Examples
    --------
    A stock that starts with the amount 2018, and increments the amount
    by 1 at each period.

    >>> stock('Year', 1, 2019)

    The initial amount defaults to zero.

    >>> stock('Age', 1)

    A stock can take a docstring-like description.

    >>> stock('Year', '''the current year''', 1, 2019)

    The initial amount can be different in each treatment.

    >>> stock('MenuItemCount', 1, PerTreatment({'As is': 20, 'To be': 22}))

    The increment can be different for each treatment.

    >>> stock('MenuItemCount', PerTreatment({'As is': 1, 'To be': 2}), 20)

    The increment can be a callable that uses no variables. Note the empty
    tuple of variables.

    >>> stock('MenuItemCount', lambda: random.randint(0,2), (), 20)

    The initial amount can be a callable. If the initial amount is a 
    callable, the increment must also be a callable. Note the empty tuples.

    >>> stock('MenuItemCount', 
    .,,     lambda: random.randint(15, 18), (),
    ...     lambda: random.randint(20, 22), ())

    Variables can be provided for the increment callable.

    >>> stock('Savings', lambda interest: interest, ('Interest',), 0)

    Variables can be provided for the initial callable.

    >>> stock('Savings', 
    ...     lambda interest, additions: interest + additions, 
    ...     ('Interest', 'AdditionsToSavings'), 
    ...     lambda initial: initial, 
    ...     ('InitialSavings',))

    Feedback is supported.

    >>> stock('Savings', lambda interest: interest, ('Interest',), 1000)
    ... variable('Rate', 0.05)
    ... variable('Interest', 
    ...     lambda savings, rate: savings * rate, 'Savings', 'Rate')

    The amounts can be numpy arrays, or other Python objects.

    >>> stock('Revenue', np.array([5, 5, 10]), np.array([0, 0, 0]))
    ... variable('Cost', np.array([10, 10, 10]))
    ... variable('Earnings', lambda r, c: r - c, 'Revenue', 'Cost')
    """
    logging.info('Creating stock %s', stock_name)
    incr_def, incr_vars, init_def, init_vars, docstring = _parse_stock(
        stock_name, args)
    return _create_stock(
        stock_name, docstring, incr_def, incr_vars, init_def, init_vars)


def _parse_stock(name, args):
    """Parse the arguments in stock_args, and return them properly sorted."""
    assert len(args) > 0, '{} has no definition'.format(name)
    if isinstance(args[0], str):
        docstring, incr_def, *args = args
    else:
        incr_def, *args = args
        docstring = ''

    if not args:
        return incr_def, None, 0, None, docstring
    elif len(args) == 1:
        return incr_def, None, args[0], None, docstring
    elif len(args) == 2:
        return incr_def, args[0], args[1], None, docstring
    else:
        return incr_def, args[0], args[1], args[2], docstring


def _create_stock(stock_name, docstring,
                  increment_definition, increment_variables,
                  initial_definition, initial_variables):
    """Create a new stock."""
    initial = create_calculator(initial_definition, initial_variables)
    incr = create_calculator(increment_definition, increment_variables)
    newstock = type(stock_name, (StockInstance,),
                    {'__doc__': docstring, '_initial': initial,
                     '_incremental': incr})
    Model.add_variable_to_current_context(newstock)
    return newstock


#
# Accum class
#

class Accum(Incrementer):
    """Like ACCUM in SimLang."""
    def _check_for_cycle_in_depends_on(cls, checked_already, dependents=None):
        """Check for cycles involving this accum."""
        for dname in cls.depends_on(for_init=True):
            d = cls._model.variable(dname)
            d.check_for_cycle(checked_already, dependents=dependents)
        for dname in cls.depends_on(for_init=False):
            d = cls._model.variable(dname)
            d.check_for_cycle(checked_already, dependents=dependents)


class AccumInstance(IncrementerInstance, metaclass=Accum):
    """Like ACCUM in SimLang, for a particular treatment instance."""

    @classmethod
    def depends_on(cls, for_init=False, for_sort=False, ignore_pseudo=False):
        """Return the variables this accum depends on.

        :param for_init: return only the variables used in initialization
        :param for_sort: return only the variables relevant for sorting vars
        :param ignore_pseudo: do not return names of pseudo-variables
        :return: list of all variable names this variable depends on
        """
        if for_init:
            return cls._initial.depends_on(ignore_pseudo)
        else:
            return cls._incremental.depends_on(ignore_pseudo)

    def _step(self):
        """Advance the accum by one step."""
        increment = self._incremental.calculate(
            self._treatment.name,
            [v.amount() for v in self._increment_depends_on_instances]
            )
        self._amount += increment


def accum(accum_name, *args):
    """
    Create a system dynamics accum. 

    An accum is much like a :class:`Stock`, modeling something that
    accumulates or depletes over time. Like a stock, an accum defines
    both an initial amount and an increment.

    There is one key difference between a stock and an accum: an accum 
    is incremented with the current amounts
    of its dependency variables, not the amounts in the last period. 
    This seemingly minor difference has a big impact: a circular dependency
    can be created with a stock, but not with an accum. The stock
    **Savings** can depend on **Interest**, which depends in turn on
    **Savings**. But this only works if **Savings** is a stock. If 
    **Savings** is an accum, the same circular dependency is a model error.

    At any simulated period, the accum has an amount. The amount changes
    over time, incrementing or decrementing at each period. The amount
    can be a simple numeric like a Python integer or a Python float. 
    Or it might be some more complex Python object: a list,
    a tuple, a numpy array, or an instance of a user-defined class. In 
    any case, the accum's amount must support addition.

    The accum may have several amounts, one for each treatment in the model.

    An accum definition has two parts: an initial amount and an increment.
    The initial amount is defined by either a Python object, or by a
    a callable and a list of dependency variables. If it is defined by a Python 
    object, any Python object can be used, except for a string or a 
    callable.

    If the initial amount is defined by a callable and list of variables,
    that callable is called a single time, at accum initialization, with the 
    initial amounts of each of the dependency variables. The names of
    the dependency variables are provided, i.e. the list is a list of strings.
    Each dependency
    variable named can either be a plain variable or a stock or a constant
    or any of the other variable elements of a model. The result of the 
    execution of the callable becomes the initial amount of the accum.

    The accum increment is also defined by either a Python object, or by
    a callable and a list of dependency variables. If defined by a Python
    object, any Python object can be used, except for a string or a 
    callable. 

    If the increment is a callable, the callable is called once every
    period, with the amounts of each of the dependency variables. As with
    the initial amount, names of dependency variables are given, and the
    list is a list of strings. Each dependency variable
    can be a plain variable or a stock or a constant or any of the 
    elements of a model. The callable is given the amounts of the 
    variables at the previous period, not the current period, to 
    determine the increment of the stock for this period. 

    The increment is how much the stock's amount changes in each period.
    Note that this is another difference between an accum and a stock: 
    for a stock the amount incremented depends on the timestep; for an 
    accum it does not. For example, if both a stock **S** and an accum **A**
    have an increment of 10, and the timestep is 0.5, **S** will increase
    by 5 every period but **A** will increase by 10.

    The initial amount and the increment may vary by treatment, either
    because one or more of the the dependency variables vary by treatment,
    or because of an explicit :class:`PerTreatment` expression. 

    Parameters
    ----------
    accum_name : str
        Name of the accum. The name must be unique within a model.

    args
        The args might include an optional docstring-like description, at
        the beginning. The interpretation of the remaining args depends on
        their count. 

        If there is only a single argument (aside from the
        optional description), it is the non-callable
        definition of the increment. The accum is assumed to be initialized
        at zero.

        If there are only two arguments (aside from the description),
        the first is the non-callable definition of the increment, and the
        second is the non-callable definiton of the initialization.

        If there are three arguments (aside from description):

        #. the first argument is a callable definition of the increment, executed every period

        #. the second argument is a tuple of names of dependency variables, whose amounts will be used as arguments of the increment callable. The tuple may be empty (i.e. **()**), if the increment callable takes no arguments.
        
        #. the third argument is a Python object, the non-callable definition of the initialization

        If there are four arguments (sans descriptions):

        #. the first argument is a callable definition of the increment, executed every period

        #. the second argument is a tuple of names of dependency variables, whose amounts will be used as arguments of the increment callable. The tuple may be empty (i.e. **()**), if the increment callable takes no arguments.
        
        #. the third argument is a callable definition of the initialization, executed once, at at initialization, and again each time the simulation is reset

        #. the fourth argument is a tuple of names of dependency variables, whose amounts are used as arguments of the initialization callable. The tuple may be empty (i.e. **()**), if the initialization callable takes no arguments.

        See usage examples below.

    Returns
    -------
    Accum
        the newly-created accum

    See Also
    --------
    variable : Create a non-stock plain variable

    constant : Create a plain variable whose amount does not change

    stock : Create an stock, much like a stock except that it uses the 
        amounts of the variables in the prior period, instead of the current
        period

    :class:`PerTreatment` : for defining how an increment or initialization
        varies from treatment to treatment

    Examples
    --------
    An accum that starts with the amount 2018, and increments the amount
    by 1 at each timestep.

    >>> accum('Year', 1, 2019)

    The initial amount defaults to zero.

    >>> accum('Age', 1)

    An accum can take a docstring-like description.

    >>> accum('Year', '''the current year''', 1, 2019)

    The initial amount can be different in each treatment.

    >>> accum('MenuItemCount', 1, PerTreatment({'As is': 20, 'To be': 22}))

    The increment can be different for each treatment.

    >>> accum('MenuItemCount', PerTreatment({'As is': 1, 'To be': 2}), 20)

    The increment can be a callable that uses no variables. Note the empty
    tuple of variables.

    >>> accum('MenuItemCount', lambda: random.randint(0,2), (), 20)

    The initial amount can be a callable. If the initial amount is a 
    callable, the increment must also be a callable. Note the empty tuples.

    >>> accum('MenuItemCount', 
    ...     lambda: random.randint(15, 18), (),
    ...     lambda: random.randint(20, 22), ())

    Variables can be provided for the increment callable.

    >>> accum('Savings', lambda interest: interest, ('Interest',), 0)

    Variables can be provided for the initial callable.

    >>> accum('Savings', 
    ...     lambda interest, additions: interest + additions, 
    ...     ('Interest', 'AdditionsToSavings'), 
    ...     lambda initial: initial, 
    ...     ('InitialSavings',))
    """
    logging.info('Creating accume %s', accum_name)
    incr_def, incr_vars, init_def, init_vars, docstring = _parse_stock(
        accum_name, args)
    return _create_accum(
        accum_name, docstring, incr_def, incr_vars, init_def, init_vars)


def _create_accum(accum_name, docstring,
                  increment_definition=0, increment_variables=None,
                  initial_definition=0, initial_variables=None):
    """Create a new accum."""
    initial = create_calculator(initial_definition, initial_variables)
    increment = create_calculator(increment_definition, increment_variables)
    new_accum = type(accum_name, (AccumInstance,),
                     {'__doc__': docstring, '_initial': initial,
                      '_incremental': increment})
    Model.add_variable_to_current_context(new_accum)
    return new_accum


#
# previous: a variable that accesses previous value of another variable
#

class Previous(CommonVariable):
    def _check_for_cycle_in_depends_on(self, checked_already, dependents):
        """Check for cycles among the depends on for this simpler variable."""
        pass

    def _show_definition_and_dependencies(self):
        """Print the definition and variables it depends on."""
        print('Previous variable: {}'.format(self._earlier))

    def antecedents(self, ignore_pseudo=False):
        """Return all the depends_on variables."""
        if ignore_pseudo and self._earlier == '__model__':
            return []
        else:
            return [self._model[self._earlier]]

    def has_unitary_definition(self):
        """Returns whether the previous has a unitary definition."""
        return True


class PreviousInstance(SimpleVariableInstance, metaclass=Previous):
    """A variable that takes the previous amount of another variable."""

    def wire_instance(self, model, treatment_name):
        """Set the variable this instance depends on."""
        self._previous_instance = model.variable_instance(
            self._earlier, treatment_name)

    def _calculate_amount(self):
        """Calculate the current amount of this previous."""
        previous_amount = self._previous_instance.previous_amount()
        if previous_amount is not None:
            return previous_amount
        elif self._init_amount == '_prior_var':
            # no previous olds, use current value
            current_amount = self._previous_instance.amount()
            return current_amount
        else:
            return self._init_amount

    @classmethod
    def depends_on(cls, for_init=False, for_sort=False, ignore_pseudo=False):
        """Return the variables this variable depends on.

        :param for_init: return only the variables used in initialization
        :param for_sort: return only the variables relevant for sorting vars
        :param ignore_pseudo: do not return names of pseudo-variables
        :return: list of all variable names this variable depends on
        """
        if ignore_pseudo and cls._earlier == '__model__':
            return []
        if not for_sort:
            return [cls._earlier]
        elif for_init and cls._init_amount == '_prior_var':
            return [cls._earlier]
        else:
            return []




def previous(variable_name, *args):
    """
    Create a new previous, a variable whose amount is the amount of another
    variable in the previous timestep.

    Parameters
    ----------
    variable_name : str
        Name of the previous. The name must be unique within a single model.

    args 
        The args might include an optional docstring-like description, at
        the beginning. The interpretation of the remaining args depends on
        their count.

        If there is only a single argument (aside from the optional 
        description), it is the name of the prior variable, as a string.
        The prior the variable is the variable 
        whose previous amount becomes the current amount of this preious. 
        The initial amount of the newly defined previous is the initial 
        amount of the prior variable.

        If there are two arguments (aside from the optional description),
        the first is the name of the prior variable, and the second is a
        Python object that is the initial amount of the previous. Any 
        Python object is allowed, except a string or a callable.

        As with any other variable, a previous may take different amounts
        in each treatment of the model.

    Returns
    -------
    Previous
        the newly created previous

    See Also
    --------
    variable : Create a non-stock plain variable

    constant : Create a plain variable whose amount does not change

    Examples
    --------
    Finding yesterday's sales, when the timestep is one day.

    >>> previous('YesterdaySales', 'Sales')

    A previous might have a description.

    >>> previous('YesterdaySales', 
        '''Total sales in the prior day''', 
        'Sales')

    A previous might have an initial amount, if that amount needs to be 
    different from the initial amount of the prior.

    >>> previous('YesterdaySales', 
        '''Total sales in the prior day''', 
        'Sales',
        3000)
    """
    if len(args) == 1:
        earlier = args[0]
        docstring = ''
        init_amount = '_prior_var'
    elif len(args) == 2 and isinstance(args[1], str):
        docstring, earlier = args
        init_amount = '_prior_var'
    elif len(args) == 2:
        earlier, init_amount = args
        docstring = ''
    elif len(args) == 3:
        docstring, earlier, init_amount = args
    elif len(args) == 0:
        raise MinnetonkaError(
            'Previous {} names no variable for prior value'.format(
                variable_name))
    else:
        raise MinnetonkaError('Too many arguments for previous {}: {}'.format(
            variable_name, args))

    return _create_previous(variable_name, docstring, earlier, init_amount)


def _create_previous(
        latter_var_name, docstring, earlier_var_name,
        init_amount='_prior_var'):
    """Create a new previous.

    Create a new previous, a variable that accesses previous value of another
    variable.
    """
    newvar = type(latter_var_name, (PreviousInstance,),
                  {'__doc__': docstring, '_earlier': earlier_var_name,
                   '_init_amount': init_amount})
    Model.add_variable_to_current_context(newvar)
    return newvar


#
# foreach: for iterating across a dict within a variable
#


def foreach(by_item_callable):
    """Return a new callable that will iterate across dict or tuple.

    The new callable---when provided with a dict or tuple---will iterate
    across it, calling the original callable, and creating a new dict or tuple.
    If multiple dicts or tuples are provided, the original callable is called
    on each set of them, in turn.
    """
    def _across(item1, *rest_items):
        return _foreach_fn(item1)(item1, *rest_items)

    def _foreach_fn(item):
        """Return the appropriate foreach function for the argument."""
        if isinstance(item, dict):
            return _across_dicts
        elif isinstance(item, MinnetonkaNamedTuple):
            return _across_namedtuples
        elif isinstance(item, tuple):
            return _across_tuples
        else:
            raise MinnetonkaError(
                'First arg of foreach {} must be dictionary or tuple'.format(
                    item))

    def _across_dicts(dict1, *rest_dicts):
        """Execute by_item_callable on every item across dict."""
        try:
            return {k: by_item_callable(dict1[k], *[_maybe_element(r, k)
                                                    for r in rest_dicts])
                    for k in dict1.keys()}
        except KeyError:
            raise MinnetonkaError('Foreach encountered mismatched dicts')

    def _maybe_element(maybe_dict, k):
        """Return maybe_dict[k], or just maybe_dict, if not a dict."""
        # It's kind of stupid that it tries maybe_dict[k] repeatedly
        try:
            return maybe_dict[k]
        except TypeError:
            return maybe_dict

    def _across_namedtuples(*tuples):
        """Execute by_item_callable across tuples and scalars."""
        if _is_all_same_type_or_nontuple(*tuples):
            tuples = [_repeat_if_necessary(elt) for elt in tuples]
            return type(tuples[0])(*(by_item_callable(*tupes)
                                     for tupes in zip(*tuples)))
        else:
            raise MinnetonkaError('Foreach encountered mismatched namedtuples')

    def _is_all_same_type_or_nontuple(first_thing, *rest_things):
        """Return whether everything is either the same type, or a scalar."""
        first_type = type(first_thing)
        return all(type(thing) == first_type or not isinstance(thing, tuple)
                   for thing in rest_things)

    def _across_tuples(*tuples):
        """Execute by_item_callable across tuples."""
        tuples = (_repeat_if_necessary(elt) for elt in tuples)
        return tuple(by_item_callable(*tupes) for tupes in zip(*tuples))

    def _repeat_if_necessary(elt):
        """Make an infinite iter from a scalar."""
        return elt if isinstance(elt, tuple) else itertools.repeat(elt)

    return _across

#
# mn_namedtuple: a variant of namedtuple in which the named tuples support
# some basic operations
#
# Add new operations as needed
#


class MinnetonkaNamedTuple():
    """A mixin class for std namedtuple, so operators can be overridden."""

    def __add__(self, other):
        """Add something to a mn_named tuple.

        Either add another mn_namedtuple to this one, element by element, or
        add a scalar to each element.
        """
        if isinstance(other, tuple):
            try:
                return type(self)(*(x + y for x, y in zip(self, other)))
            except:
                return NotImplemented
        else:
            try:
                return type(self)(*(x + other for x in self))
            except:
                return NotImplemented

    def __radd__(self, other):
        """Add a mn_named tuple to a scalar.

        Add every element of a mn_namedtuple to a scalar.
        """
        if isinstance(other, tuple):
            return NotImplemented
        else:
            try:
                return type(self)(*(x + other for x in self))
            except:
                return NotImplemented

    def __sub__(self, other):
        """Subtract something from the mn_namedtuple.

        Either subtract another mn_namedtuple from it, element by element,
        or subtract a scalar from each element.
        """
        if isinstance(other, tuple):
            try:
                return type(self)(*(x - y for x, y in zip(self, other)))
            except:
                return NotImplemented
        else:
            try:
                return type(self)(*(x - other for x in self))
            except:
                return NotImplemented

    def __rsub__(self, other):
        """Subtract a mn_namedtuple from a scalar.

        Subtract every element of a mn_namedtuple from a scalar.
        """
        if isinstance(other, tuple):
            return NotImplemented
        else:
            try:
                return type(self)(*(other - x for x in self))
            except:
                NotImplemented

    def __mul__(self, other):
        """Multiply every element in the mn_namedtuple.

        Either multiply it by another mn_namedtuple, element by element,
        or multiple every element by a scalar.
        """
        if isinstance(other, tuple):
            try:
                return type(self)(*(x * y for x, y in zip(self, other)))
            except:
                return NotImplemented
        else:
            try:
                return type(self)(*(x * other for x in self))
            except:
                return NotImplemented

    @classmethod
    def _create(cls, val):
        """Create a new namedtuple with a value of val for every field."""
        return cls._make([val for _ in range(len(cls._fields))])


def mn_namedtuple(typename, *args, **kwargs):
    """Create a namedtuple class that supports operator overriding."""
    assert type(typename) == str, "Namedtuple name must be a string"
    inner_typename = '_' + typename
    inner_type = collections.namedtuple(inner_typename, *args, **kwargs)
    return type(typename, (MinnetonkaNamedTuple, inner_type,), {})


#
# Utility functions
#


def safe_div(dividend, divisor, divide_by_zero_value=0):
    """Return the result of division, allowing the divisor to be zero."""
    return dividend / divisor if divisor != 0 else divide_by_zero_value


def norm_cdf(x, mu, sigma):
    """Find the normal CDF of x given mean and standard deviation."""
    return norm(loc=mu, scale=sigma).cdf(x)


def array_graph_xy(x, XYs):
    """Find linear interpolation of f(x) given a tuple of Xs and Ys.

    Like ARRAYGRAPHXY in SimLang.
    """
    Xs, Ys = map(list, zip(*XYs))
    return _inner_array_graph(x, Xs, Ys)


def _inner_array_graph(x, Xs, Ys):
    if np.all(np.diff(Xs) > 0):
        return np.interp(x, Xs, Ys)
    elif np.all(np.diff(Xs[::-1]) > 0):
        return np.interp(x, Xs[::-1], Ys[::-1])
    else:
        raise MinnetonkaError(
            'Xs {} are neither increasing nor descreasing', Xs)


def array_graph_yx(y, XYs):
    """Find x such that f(x) is approproximately y via linear interpolation.

    Like ARRAYGRAPHYX in SimLang.
    """
    Xs, Ys = map(list, zip(*XYs))
    return _inner_array_graph(y, Ys, Xs)

#
# Errors and warnings
#


class MinnetonkaError(Exception):
    """An error for some problem in Minnetonka."""

    def __init__(self, message):
        """Initialize the MinnetonkaError."""
        self.message = message


class MinnetonkaWarning(Warning):
    """A warning for some problem in Minnetonka."""

    def __init__(self, message):
        """Initialize the MinnetonkaWarning."""
        self.message = message

#
# Logging
#


class JsonSafeFormatter(logging.Formatter):
    """An alternative formatter, based on bit.ly/2ruBlL5."""

    def __init__(self, *args, **kwargs):
        """Initialize the json formatter."""
        super().__init__(*args, **kwargs)

    def format(self, record):
        """Format the record for json."""
        record.msg = json.dumps(record.msg)[1:-1]
        return super().format(record)


def initialize_logging(logging_level, logfile):
    """Initialize the logging system, both to file and to console for errs."""
    root_logger = logging.getLogger()
    root_logger.setLevel(logging_level)
    initialize_logging_to_file(logging_level, logfile, root_logger)
    initialize_logging_errors_to_console(root_logger)


def initialize_logging_to_file(logging_level, logfile, logger):
    """Initialize the logging system, using a json format for log files."""
    jsonhandler = logging.FileHandler(logfile, mode='w')
    jsonhandler.setLevel(logging_level)
    formatter = JsonSafeFormatter("""{
        "asctime": "%(asctime)s",
        "levelname": "%(levelname)s",
        "thread": "%(thread)d",
        "filename": "%(filename)s",
        "funcName": "%(funcName)s",
        "message": "%(message)s"
        }""")
    formatter.converter = time.gmtime
    jsonhandler.setFormatter(formatter)
    logger.addHandler(jsonhandler)


def initialize_logging_errors_to_console(logger):
    """Log errors to the console, in a simple single-line format."""
    ch = logging.StreamHandler()
    ch.setLevel(logging.ERROR)
    ch.setFormatter(logging.Formatter('Error: %(asctime)s - %(message)s'))
    logger.addHandler(ch)
