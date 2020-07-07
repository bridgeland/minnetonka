"""minnetonka.py: value modeling in python"""

__author__ = "Dave Bridgeland"
__copyright__ = "Copyright 2017-2019, Hanging Steel Productions LLC"
__credits__ = ["Dave Bridgeland"]
__version__ = "0.0.2"
__maintainer__ = "Dave Bridgeland"
__email__ = "dave@hangingsteel.com"
__status__ = "Prototype"

# Unless explicitly stated otherwise all files in this repository are licensed 
# under the Apache Software License 2.0. 

import warnings
import copy
import collections 
import itertools
import logging
import json
import time
import inspect
import re

from scipy.stats import norm

import numpy as np

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
    ...     variable('Revenue', np.array([30.1, 15, 20]))
    ...     variable('Cost', 
    ...         PerTreatment({'As is': np.array([10, 10, 10]),
    ...                      {'To be': np.array([5, 5, 20])})
    ...     variable('Earnings', lambda r, c: r - c, 'Revenue', 'Cost')
    """

    # is a model being defined in a context manager? which one?
    _model_context = None

    def __init__(self, treatments, derived_treatments, timestep=1, 
                 start_time=0, end_time=None, on_init=None, on_reset=None):
        """Initialize the model, with treatments and optional timestep."""
        self._treatments = treatments 
        self._derived_treatments = derived_treatments
        # prior to m.initialize(), this is a regular dict. It is
        # converted to an OrderedDict on initialization, ordered with
        # dependent variables prior to independent variables
        self._variables = ModelVariables()
        self._pseudo_variable = ModelPseudoVariable(self)
        self._user_actions = UserActions()
        self._timestep = timestep
        self._start_time = start_time 
        self._end_time = end_time
        self._constraints = []
        self._on_init = on_init 
        self._on_reset = on_reset

        #: Current time in the model, accessible in a specifier. See
        #: example detailed in :func:`variable`
        self.TIME = start_time

    @property
    def STARTTIME(self):
        return self._start_time
    @property
    def ENDTIME(self):
        return self._end_time
    
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
        if to_end and self._end_time:
            for i in range(int((self._end_time - self.TIME) / self._timestep)):
                self._step_one()
            self._user_actions.append_step(n, to_end)

        elif self._end_time is None or self.TIME < self._end_time:
            for i in range(n):
                self._step_one()
            self._user_actions.append_step(n, to_end)

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
        if self._on_reset:
            self._on_reset(self)
        self._initialize_time()
        self._variables.reset(reset_external_vars)
        self._user_actions.append_reset(reset_external_vars)

    def initialize(self):
        """Initialize simulation."""
        logging.info('enter')
        if self._on_init:
            self._on_init(self)
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
        """Return an iterator of the treatments."""
        return self._treatments.values()

    def _is_valid_treatment(self, treatment):
        """Is the treatment valid?"""
        return treatment == '__all__' or treatment in self._treatments

    def treatment(self, treatment_name):
        """Return a particular treatment from the model."""
        try:
            return self._treatments[treatment_name]
        except KeyError:
            raise MinnetonkaError('Model has no treatment {}'.format(
                treatment_name))

    def derived_treatment_exists(self, treatment_name):
        """Does the derived treatment exist on the model?"""
        return treatment_name in self._derived_treatments

    def derived_treatment(self, treatment_name):
        """Return a particular derived treatment from the model."""
        try:
            return self._derived_treatments[treatment_name]
        except KeyError:
            raise MinnetonkaError('Model has no derived treatment {}'.format(
                treatment_name))

    def derived_treatments(self):
        """Iterator over names of all derived treatments."""
        return self._derived_treatments.keys()


    def variable(self, variable_name):
        """
        Return a single variable from the model, by name.

        Return a single variable---or stock or constant or accum or previous---
        from the model, by providing the variable's name. 

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
        variable('Cost')

        ... or use subscription syntax to do the same thing

        >>> m['Cost']
        variable('Cost')
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


    @classmethod
    def add_constraint_to_current_context(cls, constraint):
        """If context is currently open, add this constraint."""
        if cls._model_context is not None:
            cls._model_context._constraints.append(constraint)

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
        """
        Recalculate all variables, without advancing the step.

        Recalculation is only necessary when the amount of a variable (or 
        constant or stock) is changed
        explicitly, outside of the model logic. The variables that depend on
        that changed variable will take amounts that do not reflect the changes,
        at least until the model is stepped. If that is not appropriate, a
        call to **recalculate()** will calculate new updated amounts for all
        those dependent variables.

        Example
        -------
        >>> with model() as m:
        ...    Foo = constant('Foo', 9)
        ...    Bar = variable('Bar', lambda x: x+2, 'Foo')
        >>> Bar['']
        11

        >>> Foo[''] = 7

        **Bar** still takes the amount based on the previous amount of **Foo**.

        >>> Bar['']
        11

        Recalculating updates the amounts.

        >>> m.recalculate()
        >>> Bar['']
        9
        """
        if self.STEP==0:
            self._variables.recalculate(at_start=True)
        else:
            self._variables.recalculate(at_start=False)
        self._user_actions.append_recalculate()

    def variable_instance(self, variable_name, treatment_name):
        """Find or create right instance for this variable and treatment."""
        # A more pythonic approach than checking for this known string?
        if variable_name == '__model__':
            return self._pseudo_variable
        else:
            return self.variable(variable_name).by_treatment(treatment_name)

    def validate_and_set(self, variable_name, treatment_name, new_amount,
                         excerpt='', record=True):
        """Validate the new_amount and if valid set the variable to it."""
        res = _Result(
            variable=variable_name, 
            amount=new_amount, 
            treatment=treatment_name,
            excerpt=excerpt)
        try: 
            var = self.variable(variable_name)
        except MinnetonkaError:
            return res.fail(
                'UnknownVariable', f'Variable {variable_name} not known.')
        if self._is_valid_treatment(treatment_name):
            res = var.validate_and_set(treatment_name, new_amount, res, excerpt)
            if res['success'] and record:
                self._user_actions.append_set_variable(
                    variable_name, treatment_name, new_amount, excerpt)
            return res
        else:
            return res.fail(
                'UnknownTreatment', f'Treatment {treatment_name} not known.')

    def validate_all(self):
        """Validate against all cross-variable constraints. Return results."""
        errors = self._validate_errors()
        if len(errors) == 0:
            return {'success': True}
        else:
            return {'success': False, 'errors': errors}

    def _validate_errors(self):
        """Return all validation errors from all the constraints."""
        errors = (constraint.fails(self) for constraint in self._constraints)
        return [err for err in errors if err]

    def recording(self):
        """Return a string of all the user actions, for persistance."""
        return self._user_actions.recording()

    def replay(self, recording, rewind_actions_first=True, ignore_step=False):
        """Replay a bunch of previous actions."""
        self._user_actions.replay(
            recording, self, rewind_first=rewind_actions_first, 
            ignore_step=ignore_step)

    def history(self, base=False):
        """Return history of all amounts of all variables in all treatments."""
        return self._variables.history(base=base)

    def is_modified(self, varname, treatment_name):
        """Has variable named varname been modified in treatment?"""
        return self.variable_instance(varname, treatment_name).is_modified()


def model(variables=[], treatments=[''], derived_treatments=None,
          initialize=True, timestep=1, start_time=0, end_time=None,
          on_init=None, on_reset=None):
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

    variable : Create a :class:`Variable` to put in a model

    constant : Create a :class:`Constant` to put in a model

    previous : Create a :class:`Previous` to put in a model

    stock : Create a system dynamics :class:`Stock`, to put in a model

    accum : Create an :class:`Accum`, to put in a model
    
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

    derived_treatments={} if derived_treatments is None else derived_treatments
    for dt in derived_treatments.keys():
        if dt in treatments:
            raise MinnetonkaError(f'Derived treatment {dt} is also a treatment')

    if end_time is not None and end_time < start_time:
        raise MinnetonkaError('End time {} is before start time {}'.format(
            end_time, start_time))

    m = Model(
        {t.name: t for t in [
            _create_treatment_from_spec(spec) for spec in treatments]},
        derived_treatments=derived_treatments,
        timestep=timestep,
        start_time=start_time, 
        end_time=end_time,
        on_init=on_init,
        on_reset=on_reset)
    m.add_variables(*variables)
    if initialize and variables:
        m.initialize()
    return m


class UserActions:
    """Manage the list of user actions."""
    def __init__(self):
        self._actions = [] 

    def append_set_variable(self, varname, treatment_name, new_amount, excerpt):
        """Add a single user action (e.g. set variable) to record.""" 
        self._append_action(ValidateAndSetAction(
            varname, treatment_name, excerpt, new_amount))

    def _append_action(self, new_action):
        """Add the new action to the lsit of actions."""
        if any(new_action.supercedes(action) for action in self._actions):
            self._actions = [action for action in self._actions 
                             if not new_action.supercedes(action)]
        self._actions.append(new_action)

    def append_step(self, n, to_end):
        """Add a single user step action to record."""
        self._append_action(StepAction(n, to_end))

    def append_recalculate(self):
        """Append a single recalculate action to records."""
        self._append_action(RecalculateAction())

    def append_reset(self, reset_external_vars):
        """Append a single reset to records."""
        self._append_action(ResetAction(reset_external_vars))
 
    def recording(self):
        """Record a string of all user actions, for persistance.""" 
        return json.dumps([action.freeze() for action in self._actions]) 

    def thaw_recording(self, recording):
        return json.loads(recording)

    def replay(self, recording, mod, rewind_first=True, ignore_step=False):
        """Replay a previous recording."""
        if rewind_first:
            self.rewind()
        for frozen_action in self.thaw_recording(recording):
            action_type = frozen_action['type']
            if ignore_step and action_type =='step':
                pass
            else:
                del frozen_action['type']
                action = {
                    'validate_and_set': ValidateAndSetAction,
                    'step': StepAction,
                    'recalculate': RecalculateAction,
                    'reset': ResetAction
                }[action_type](**frozen_action)

                action.thaw(mod) 

    def rewind(self):
        """Set the action list back to no actions."""
        self._actions = []


class ValidateAndSetAction:
    """A single user action for setting a variable"""
    def __init__(self, variable_name, treatment_name, excerpt, amount):
        self.variable = variable_name
        self.treatment = treatment_name
        self.excerpt = excerpt
        try:
            json.dumps(amount)
            self.amount = amount 
        except TypeError:
            raise MinnetonkaError(
                f'Cannot save amount for later playback: {amount}')

    def supercedes(self, other_action):
        """Does this action supercede the other? Note: amounts do not matter."""
        if isinstance(other_action, ValidateAndSetAction):
            return (
                self.variable == other_action.variable and
                self.treatment == other_action.treatment and
                self.excerpt == other_action.excerpt)
        else: 
            return False 

    def freeze(self):
        """Freeze this to simple json."""
        return {
            'type': 'validate_and_set',
            'variable_name': self.variable, 
            'treatment_name': self.treatment, 
            'excerpt': self.excerpt,
            'amount': self.amount
          }

    def thaw(self, mod):
        """Apply once-frozen action to model."""
        res = mod.validate_and_set(
            self.variable, self.treatment, self.amount, self.excerpt)
        if not res['success']:
            raise MinnetonkaError(
                'Failed to replay action {}["{}"]{} = {},'.format(
                    variable, treatment, excerpt, amount) +
                'Result: {}'.format(res))

class StepAction:
    """A single user action for stepping the model."""
    def __init__(self, n, to_end):
        self.n = n
        self.to_end = to_end 

    def freeze(self):
        """Freeze this to simple json."""
        return {'type': 'step', 'n': self.n, 'to_end': self.to_end }

    def thaw(self, mod):
        """Apply once-frozen action to model."""
        mod.step(n=self.n, to_end=self.to_end)

    def supercedes(self, other_action):
        """Does this action supercede the prior action? No it does not"""
        return False 


class RecalculateAction:
    """A single user action to recalculate the model."""
    def __init__(self):
        pass 

    def freeze(self):
        """Freeze this to simple json."""
        return {'type': 'recalculate'}

    def thaw(self, mod):
        """Apply once-frozen action to model."""
        mod.recalculate()

    def supercedes(self, other_action):
        """Does this action supercede the prior action? No it does not"""
        return False 


class ResetAction:
    """A single user action to reset the simulation."""
    def __init__(self, reset_external_vars):
        self.reset_external_vars = reset_external_vars

    def freeze(self):
        """Freeze this to simple json."""
        return {
            'type': 'reset', 
            'reset_external_vars': self.reset_external_vars
        }

    def thaw(self, mod):
        """Apply once-frozen action to model."""
        mod.reset(reset_external_vars=self.reset_external_vars)

    def supercedes(self, other_action):
        """Does the action supercede the prior action?"""
        if self.reset_external_vars:
            # Remove everything already done
            return True 
        elif isinstance(other_action, ValidateAndSetAction):
            return False
        else:
            return True 


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
        self.set_initial_amounts()
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
                var.tary = 'multitary'
            elif var.antecedents(ignore_pseudo=True) == []:
                var.tary = 'unitary'
            else:   
                var.tary = 'unknown'

    def _label_multitary_succedents(self):
        """Label all succedents of multitary variables as multitary."""
        succedents = self._collect_succedents()
        multitaries = [v for v in self._variable_iterator() 
                       if v.tary == 'multitary']
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
        var.tary = 'multitary'
        for succ in succedents[var]:
            if succ.tary == 'unknown':
                self._label_all_succedents_multitary(succ, succedents)

    def _label_unknowns_unitary(self):
        """Label every unknown variable as unitary."""
        for v in self._variable_iterator():
            if v.tary == 'unknown':
                v.tary = 'unitary'

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

    def set_initial_amounts(self):
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

    def recalculate(self, at_start=False):
        """Recalculate all the variables without advancing step."""
        if at_start:
            for var in self._variables_ordered_for_init.values():
                var.recalculate_all()
        else:
            for var in self._variables_ordered_for_step.values():
                var.recalculate_all()

    def history(self, base=False):
        """Return history of all amounts of all variables in all treatments."""
        return {variable.name(): variable.history(base=base)
                for variable in self._variable_iterator()
                if variable.has_history()}

#
# Treatments and derived treatments
#
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

class AmountBetter:
    """A derived treatment to calculate how much better in A vs B"""
    def __init__(self, better_treatment, worse_treatment):
        self._better_treatment = better_treatment
        self._worse_treatemnt = worse_treatment

    def deriver(self, is_scored_as_golf, better_amount, worse_amount):
        """How much better is better_amount than worse_amount?"""
        if is_scored_as_golf:
            return worse_amount - better_amount
        else:
            return better_amount - worse_amount

    def depends_on(self):
        """What treatments does this amount better depend on?"""
        return [self._better_treatment, self._worse_treatemnt]

#
# Variable classes
#

# Variable class hierarchy
#
#    CommonVariable
#        Variable
#            Constant
#        Incrementer
#            Stock
#            Accum
#        Previous
#        Velocity
#        Cross
#    ModelPseudoVariable


class CommonVariable(type):
    """The common superclass for all Minnetonka variables and variable-like things."""
    def __getitem__(self, treatment_name):
        """
        Retrieve the current amount of the variable in the treatment with
        the name **treatment_name**.
        """
        if self._treatment_exists(treatment_name):
            return self.by_treatment(treatment_name).amount() 
        elif self.is_derived():
            if self._derived_treatment_exists(treatment_name):
                return self._derived_amount(treatment_name)
            else:   
                raise MinnetonkaError(
                    'Unknown derived treatment {} for variable {}'.
                    format(treatment_name, self.name())) 
        else:
            raise MinnetonkaError('Unknown treatment {} for variable {}'.
                format(treatment_name, self.name()))

    def __setitem__(self, treatment_name, amount):
        """
        Change the current amount of the variable in the treatment with the
        name **treatment_name**.
        """
        self.set(treatment_name, amount)

    def __repr__(self):
        return "{}('{}')".format(self._kind().lower(), self.name())

    def __str__(self):
        return "<{} {}>".format(self._kind(), self.name())

    def _kind(self):
        """'Variable' or 'Stock' or 'Accum' or whatever."""
        return type(self).__name__

    def create_variable_instances(self):
        """Create variable instances for this variable."""
        if self.tary == 'unitary':
            v = self()
            for treatment in self._model.treatments():
                if self.is_undefined_for(treatment.name):
                    self(treatment, undefined=True)
                else:
                    v._initialize_treatment(treatment)
        else:
            for treatment in self._model.treatments():
                self(treatment, undefined=self.is_undefined_for(
                    treatment.name))

    def note_model(self, model):
        """Keep track of the model, for future reference."""
        self._model = model

    def _treatment_exists(self, treatment_name):
        """Does this treatment exist for this variable?"""
        return treatment_name in self._by_treatment

    def by_treatment(self, treatment_name):
        """Return the variable instance associated with this treatment."""
        return self._by_treatment[treatment_name]

    def all_instances(self):
        """Return all the instances of this variable."""
        if self.tary == 'unitary':
            for val in self._by_treatment.values():
                if not val.undefined:
                    yield val
                    break
        else:
            for val in self._by_treatment.values():
                if not val.undefined:
                    yield val


    def set_all_initial_amounts(self):
        """Set the initial amounts of all the variable instances."""
        if self.tary == 'unitary':
            for treatment_name, var in self._by_treatment.items():
                if not var.undefined:
                    var.set_initial_amount(treatment_name)
                    return 
        else:
            for treatment_name, var in self._by_treatment.items():
                if not self.is_undefined_for(treatment_name):
                    var.set_initial_amount(treatment_name)

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
        for var in self.all_instances():
            var._recalculate()

    def calculate_all_increments(self, ignore):
        """Ignore this in general. Only meaningful for stocks."""
        pass

    def set(self, treatment_name, amount):
        """
        Change the current amount of the variable in the treatment named."""
        if treatment_name == '__all__':
            self.set_amount_all(amount)
        elif self._model.derived_treatment_exists(treatment_name):
            raise MinnetonkaError(
                'Cannot set {} in derived treatment {}.'.format(
                    self.name(), treatment_name))
        elif len(self._model.treatments()) == 1:
            self.set_amount_all(amount)
        elif self.tary == 'unitary':
            warnings.warn(
                'Setting amount of unitary variable {} '.format(self.name()) +
                'in only one treatment',
                MinnetonkaWarning)
            self.set_amount_all(amount)
        else:
            self.by_treatment(treatment_name).set_amount(amount)

    def set_amount_all(self, amount):
        """Set the amount for all treatments."""
        for var in self.all_instances():
            var.set_amount(amount)

    def delete_all_variable_instances(self):
        """Delete all variables instances."""
        for v in self.all_instances():
            v._treatment.remove_variable(v)
        self._by_treatment = {}

    def history(self, treatment_name=None, step=None, base=False):
        """Return the amount at a past timestep for a particular treatment."""
        if not self.is_derived() or base:
            return self._base_history(treatment_name=treatment_name, step=step)
        elif treatment_name is None:
            return self._derived_history(treatment_name=None, step=step)
        elif self._derived_treatment_exists(treatment_name):
            return self._derived_history(
                treatment_name=treatment_name, step=step)
        else:
            return self._base_history(treatment_name=treatment_name, step=step)

    def _derived_history(self, treatment_name=None, step=None):
        """Return the amount at a past timestep for a derived treatment."""
        # Or for all past timesteps
        if treatment_name is None and step is None:
            return self._full_derived_history()
        elif step is None:
            return self._history_of_derived_treatment(treatment_name)
        else:
            return self._history_of_derived_treatment_at_step(
                treatment_name, step)

    def _full_derived_history(self):
        """Return the full history of all derived treatments."""
        return {
            trt_name: self._history_of_derived_treatment(trt_name)
            for trt_name in self._model.derived_treatments()
            if self.derived_treatment_defined(trt_name)
        }

    def _history_of_derived_treatment(self, treatment_name):
        """Return the history of this derived treatment."""
        if self._derived_treatment_exists(treatment_name):
            if self._is_scored_as_combo():
                return self._history_of_derived_treatment_combo(treatment_name)
            else:
                return self._history_of_derived_treatment_simple(treatment_name)
        else:
            raise MinnetonkaError(
                'Unknown derived treatment {} for variable {}'.
                format(treatment_name, self.name())) 

    def _history_of_derived_treatment_combo(self, treatment_name):
        """Return the history of htis derived tremament, a combo."""
        dependency_histories = [
            self._model[dep]._history_of_derived_treatment(treatment_name) 
            for dep in self.depends_on()]
        return [amt for amt in map(
            lambda *amounts: 
                self._calculator.calculate(treatment_name, amounts),
            *dependency_histories)]

    def _history_of_derived_treatment_simple(self, treatment_name):
        """REturn the history of this derived treatment, not a combo."""
        treatment = self._model.derived_treatment(treatment_name)
        better_treatment, worse_treatment = treatment.depends_on()
        is_golf = self._is_scored_as_golf()
        return [
            treatment.deriver(is_golf, better, worse)
            for better, worse in zip(
                self.by_treatment(better_treatment)._history(),
                self.by_treatment(worse_treatment)._history())]

    def _history_of_derived_treatment_at_step(self, treatment_name, step):
        """Return the amount of hte derived treatment at a step in time."""
        if self._derived_treatment_exists(treatment_name):
            if self._is_scored_as_combo():
                return self._history_of_derived_treatment_at_step_combo(
                    treatment_name, step)
            else:
                return self._history_of_derived_treatment_at_step_simple(
                    treatment_name, step)
        else:
            raise MinnetonkaError(
                'Unknown derived treatment {} for variable {}'.
                format(treatment_name, self.name())) 

    def _history_of_derived_treatment_at_step_combo(self, treatment_name, step):
        """For the combo variable, return the amount of derived trt at step."""
        return self._calculator.calculate(
            treatment_name, 
            [self._model[dep]._history_of_derived_treatment_at_step(
                treatment_name, step)
             for dep in self.depends_on()])

    def _history_of_derived_treatment_at_step_simple(self, treatment_name,step):
        """For the non-combo, return the amount of derived trt at step."""
        treatment = self._model.derived_treatment(treatment_name)
        better_treatment, worse_treatment = treatment.depends_on()
        return treatment.deriver(
            self._is_scored_as_golf(),
            self._history_at_treatment_step(better_treatment, step),
            self._history_at_treatment_step(worse_treatment, step))

    def _base_history(self, treatment_name=None, step=None):
        """Return the amount at a past timestep for a base treatment. """
        # Or for all past timesteps
        if treatment_name is None and step is None:
            return {trt_name:self._history_of_treatment(trt_name)
                    for trt_name in self._by_treatment.keys()
                    if not self.is_undefined_for(trt_name)}
        elif step is None:
            return self._history_of_treatment(treatment_name)
        else: 
            return self._history_at_treatment_step(treatment_name, step)

    def _history_of_treatment(self, treatment_name):
        """Return all the historical amounts for a particular treatment."""
        return self.by_treatment(treatment_name)._history()

    def _history_at_treatment_step(self, treatment_name, step):
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
        """Return a dict of all current amounts, one for each treatment."""
        return {tmt: inst.amount() for tmt, inst in self._by_treatment.items()
                if not self.is_undefined_for(tmt)}

    def _derived_treatment_exists(self, treatment_name):
        """Does this derived treatment exist for this variable?"""
        return self._model.derived_treatment_exists(treatment_name)

    def derived_treatment_defined(self, treatment_name):
        """Does the treatment exist and are both the base treatments defined?"""
        if self._model.derived_treatment_exists(treatment_name): 
            treatment = self._model.derived_treatment(treatment_name)
            better_treatment_name, worse_treatment_name = treatment.depends_on()
            return not (
                self.is_undefined_for(better_treatment_name) or 
                self.is_undefined_for(worse_treatment_name))
        else:
            return False 

    def _derived_amount(self, treatment_name):
        """Treatment is known to be a derived treatment. Use it to calc amt."""
        treatment = self._model.derived_treatment(treatment_name)
        if self._is_scored_as_combo():
            return self._calculator.calculate(
                treatment_name, 
                [self._model.variable(vname)[treatment_name] 
                 for vname in self.depends_on()])
        else:
            return treatment.deriver(
                self._is_scored_as_golf(), 
                *[self[d] for d in treatment.depends_on()])

    def derived(self, scored_as='basketball'):
        """Mark this variable as derived, and how it is scored."""
        self._derived['derived'] = True 
        self._derived['scored_as'] = scored_as 
        return self

    def is_derived(self):
        """Does this variable support derived treatments?"""
        return self._derived['derived']

    def _is_scored_as_golf(self):
        """Is this variable scored as golf, with lower scores better?"""
        return self.is_derived() and self._derived['scored_as'] == 'golf'

    def _is_scored_as_combo(self):
        """Is this variable scored as a combo of golf and basketball?"""
        # some dependencies are scored as golf, some dependencies scored
        # as basketball
        return self.is_derived() and self._derived['scored_as'] == 'combo'

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
        # maybe I should show whether it is unitary
        print('Amounts: {}\n'.format(self.all()))

    def constraint(self, *args, **kwargs):
        """Add validator to the common variable."""
        if len(args) == 1:
            validator = args[0]
            self._validators.append(validator)
        else:
            self._validators.append(_Validator(*args, **kwargs))
        return self 

    def validate_and_set(self, treatment_name, amount, res, excerpt):
        """Validate the amount and if valid, make a change."""
        if excerpt:
            return self._validate_and_set_excerpt(
                treatment_name, amount, res, excerpt)
        else:
            return self._validate_and_set(treatment_name, amount, res)

    def _validate_and_set_excerpt(self, treatment_name, amount, res, excerpt):
        """Validate the amount and if valid, set some excerpt."""
        val, attr = self._isolate_excerpt(treatment_name, excerpt)
        if hasattr(val, 'validate'):
            try:
                valid, error_code, error_msg, suggestion = val.validate(
                    attr, amount)
            except Exception as e:
                return res.fail(
                    'Invalid', f'Validation error {str(e)} with {val}')
            if not valid:
                return res.fail(
                    error_code, error_msg, suggested_amount=suggestion)
        try:
            setattr(val, attr, amount)
            self._mark_externally_changed(treatment_name)
            return res.succeed()
        except Exception as e:
            return res.fail(
                'Unsettable', 
                'Error {} raised when setting amount of {} to {}'.format(
                    str(e), val.__class__.__name__, amount))

    def _isolate_excerpt(self, treatment_name, excerpt):
        """Find the object and attribute to be validated and set."""
        attrs = excerpt.split('.')
        if attrs[0] == '':
            attrs.pop(0)

        val = self._variable_value(treatment_name)
        for attr in attrs[:-1]:
            val = getattr(val, attr)

        return val, attrs[-1]

    def _mark_externally_changed(self, treatment_name):
        """Mark this variable as changed, even though its amount is same obj."""
        self.set(treatment_name, self._variable_value(treatment_name))

    def _variable_value(self, treatment_name):
        if treatment_name != '__all__':
            return self[treatment_name]
        elif self.tary == 'unitary':
            for treatment in self._model.treatments():
                return self[treatment.name]
        else:
            raise MinnetonkaError(
                f'validate_and_set for {self.name()} on multiple treatments')

    def _validate_and_set(self, treatment_name, amount, res):
        """Validate the amount and if valid set the variable to it."""
        valid, error_code, error_msg, suggested_amount = self._validate_amount(
            amount)
        if valid:
            self.set(treatment_name, amount)
            return res.succeed()
        elif suggested_amount is not None:
            return res.fail(
                error_code, error_msg, suggested_amount=suggested_amount)
        else:
            return res.fail(error_code, error_msg)

    def _validate_amount(self, new_amount):
        """Attempt to validate the amount, using all known validators."""
        for v in self._validators:
            valid, error_code, error_msg, suggested_amount = v.validate(
                new_amount, self.name())
            if not valid:
                return valid, error_code, error_msg, suggested_amount
        return True, None, None, None 

    def no_history(self):
        """Mark this variable as not having history."""
        self._has_history = False
        return self  

    def has_history(self):
        """Has a history, unless overridded by a subclass."""
        return self._has_history 

    def undefined_in(self, *treatment_names):
        """Mark the variable as not defined for some treatments."""
        # for now, this only affects show()
        self._exclude_treatments = treatment_names
        return self

    def is_undefined_for(self, treatment):
        """Is this variable not defined for this treatment?"""
        return treatment in self._exclude_treatments

    def substitute_description_for_amount(self, description):
        """Mark that this constant does not support amounts in details."""
        self._summary_description = description
        self._suppress_amount = True
        return self

    def summarizer(self, summary_description, callable):
        """Instead of providing the amount, run this callable to summarize."""
        self._summary_description = summary_description
        self._summarizer = callable 
        return self 

    def details(self):
        """Return a json-safe structure for the details of the variable."""
        deets = {"name": self.name(), "varies over time": True}
        history = self.history(base=True)
        if self.is_derived():
            derived_history = self.history(base=False)
            history = {**history, **derived_history}
        if hasattr(self, '_summarizer'):
            self._add_summary(deets, history)
        elif hasattr(self, "_suppress_amount") and self._suppress_amount: 
            self._add_summary_description_only(deets)
        else:
            self._add_history(deets, history)
        return deets

    def _add_summary(self, deets, history):
        """Add a summary to the deets."""
        deets['summary'] = {
            trt: [self._summarizer(amt, trt) for amt in amts]
            for trt, amts in history.items()}
        deets['summary description'] = self._summary_description
        deets['caucus'] = self._caucus_amount(history)

    def _caucus_amount(self, history):
        """Return some aggregation of the history."""        
        try:
            caucus_fn = self._caucuser
        except AttributeError:
            caucus_fn = mean 
        try: 
            caucus_amount = {
                trt: caucus_fn(amts) for trt, amts in history.items()}
        except:
            caucus_amount = {
                trt: 'error aggregating' for trt, amounts in history.items()}
        return caucus_amount

    def _add_summary_description_only(self, deets):
        """Add only a summary description to the deets."""
        deets['summary description'] = self._summary_description
        deets['caucus'] = self._summary_description

    def _add_history(self, deets, history):
        """Add amounts to deets"""
        deets['amounts'] = history  
        deets['caucus'] = self._caucus_amount(history)

    def caucuser(self, callable):
        """Instead of the arithmetic mean, run this callable for a caucus."""
        self._caucuser = callable 
        return self


class CommonVariableInstance(object, metaclass=CommonVariable):
    """
    Any of the variety of variable types.
    """

    def __init__(self, treatment=None, undefined=False):
        """Initialize this variable."""
        self.undefined = undefined 
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
        if self.undefined: 
            return None 
        elif self._extra_model_amount is None:
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

    def _history(self, step=None):
        """Return the amount at timestep step."""
        if step is None:
            return self._old_amounts + [self.amount()]
        elif step == len(self._old_amounts):
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
            self.set_initial_amount()

    def _step(self):
        """Advance this simple variable one time step."""
        self._amount = self._calculate_amount()

    def _recalculate(self):
        """Recalculate this simple variagble."""
        if self._extra_model_amount is None: 
            self._amount = self._calculate_amount()

    def set_initial_amount(self, treatment=None):
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
        # ignore for_init and for_sort since behavior is the same for simple 
        #variable
        return cls._calculator.depends_on(ignore_pseudo)

    def is_modified(self):
        """Has this instance been modified?"""
        return self._extra_model_amount is not None


class Variable(CommonVariable):
    """
    A variable whose amount is calculated from amounts of other variables.

    A variable has a value---called an 'amount'---that changes over simulated 
    time. A single
    variable can take a different amount in each model treatment. The amount 
    of a variable can be any Python object. A variable can be defined in terms 
    of the amounts of other variables.

    A variable differs from other variable-like objects (e.g.
    stocks) in that it keeps no state. Its amount depends entirely on its 
    definition, and the amounts of other variables used in the definition.

    A single variable can take a different amount in each model treatment.
    The amount of a variable in a particular treatmant can be found using
    subscription brackets, e.g. **Earnings['as is']**. See examples below.

    The amount of a variable can be changed explicitly, outside the model
    logic, e.g. **Earnings['as is'] = 2.1**. Once changed explicitly,
    the amount of 
    the variable never changes again, until the simulation is reset or 
    the amount is changed again explicitly. See examples below.

    See Also
    --------
    variable : Create a :class:`Variable`  

    :class:`Constant` : a variable that does not vary

    Examples
    --------
    Find the current amount of the variable **Earnings**, in the **as is**
    treatment.

    >>> Earnings['as is']
    2.0

    Change the current amount of the variable **Earnings** in the **as is**
    treatment.

    >>> Earnings['as is'] = 2.1

    Show everything important about the variable **Earnings**.

    >>> Earnings.show()
    Variable: Earnings
    Amounts: {'as is': 2.1, 'To be': 4.0}
    Definition: Earnings = variable('Earnings', lambda r, c: r - c, 'Revenue', 'Cost')
    Depends on: ['Revenue', 'Cost']
    [Variable('Revenue'), Variable('Cost')]
    """

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
        """Return all the variables this variable depends on."""
        m = self._model
        return [m[v] for v in self.depends_on(ignore_pseudo=ignore_pseudo)]

    def has_unitary_definition(self):
        """Returns whether the variable has a unitary definition."""
        return self._calculator.has_unitary_definition()

    def all(self):
        """
        Return a dict of all current amounts, one for each treatment.

        Example
        -------
        >>> Earnings.all()
        {'as is': 2.1, 'to be': 4.0}
        """
        return super().all()

    def history(self, treatment_name=None, step=None, base=False):
        """
        Return the amount at a past timestep for a particular treatment.

        Minnetonka tracks the past amounts of a variable
        over the course of a single simulation run,
        accessible with this function. 

        Parameters
        ----------
        treatment_name : str
            the name of some treatment defined in the model

        step : int
            the step number in the past 

        Example
        -------

        Create a model with a single variable RandomVariable.

        >>> import random
        >>> with model() as m:
        ...     RandomValue = variable(
        ...         'RandomValue', lambda: random.random() / 2)
        >>> RandomValue['']
        0.4292118957243861  

        Advance the simulation. RandomVariable changes value.

        >>> m.step()
        >>> RandomValue['']
        0.39110555756064735
        >>> m.step()
        >>> RandomValue['']
        0.23809270739004534

        Find the old values of RandomVarable.

        >>> RandomValue.history('', 0)
        0.4292118957243861
        >>> RandomValue.history('', 1)
        0.39110555756064735
        """
        return super().history(
            treatment_name=treatment_name, step=step, base=base)

    def show(self):
        """
        Show everything important about the variable.

        Example
        -------
        >>> Earnings.show()
        Variable: Earnings
        Amounts: {'as is': 2.1, 'To be': 4.0}
        Definition: Earnings = variable('Earnings', lambda r, c: r - c, 'Revenue', 'Cost')
        Depends on: ['Revenue', 'Cost']
        [Variable('Revenue'), Variable('Cost')]
        """
        return super().show()

    def __getitem__(self, treatment_name):
        """
        Retrieve the current amount of the variable in the treatment with
        the name **treatment_name**.

        Example
        -------
        Find the current amount of the variable **Earnings**, in the **as is**
        treatment.

        >>> Earnings['as is']
        2.0
        """
        return super().__getitem__(treatment_name)

    def __setitem__(self, treatment_name, amount):
        """
        Change the current amount of the variable in the treatment with the
        name **treatment_name**.

        Examples
        --------
        Change the current amount of the variable **Earnings** in the **as is**
        treatment to **2.1**.

        >>> Earnings['as is'] = 2.1

        Change the current amount of the variable **Taxes** in all treatments
        at once.

        >>> Earnings['__all__'] = 2.1
        """
        super().__setitem__(treatment_name, amount)


class VariableInstance(SimpleVariableInstance, metaclass=Variable):
    """
    A variable whose amount is calculated from the amounts of other variables.

    """

    def _calculate_amount(self):
        """Calculate the current amount of this plain variable."""
        if self.undefined:
            return None    # perhaps there should be special undefined value

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
        # could optimize by doing this all only once rather than on every call to
        # calculate
        try:
            defn = self._definition.by_treatment(treatment_name)
        except (KeyError, TypeError, AttributeError):
            defn = self._definition

        # must use callable() because foo() raises a TypeError exception
        # under two circumstances: both if foo is called with the wrong
        # number of arguments and if foo is not a callable
        if callable(defn):
            try:
                return defn(*depends_on_amounts)
            except Exception as e:
                raise MinnetonkaError((
                    'Error {} raised in treatment {} evaluating {} ' +
                    'with amounts {}').format(
                        e, treatment_name, self._definition, 
                        depends_on_amounts))

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
            self._definition.treatments_and_amounts()
            return False
        except AttributeError:
            return True 

    def add(self, augend, addend):
        """Add the two together. Augend might be a foreached object."""
        # It is kind of stupid to first try the special case, and then try
        # the general case. But adding tuples work generally, even though
        # they give the wrong result.
        try: 
            return self._definition.add(augend, addend)
        except AttributeError:
            return augend + addend

    def multiply(self, multiplicand, multiplier):
        """Multiply together. Multiplicand might be a foreached object."""
        try:
            return self._definition.multiply(multiplicand, multiplier)
        except AttributeError:
            return multiplicand * multiplier


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
    variable(variable_name, [description,] specifier, *dependencies)

    Create a variable.

    A variable has a value---called an 'amount'---that changes over simulated 
    time. A single
    variable can have a different amount in each model treatment. The amount 
    of a variable can be any Python object.  The amount of a variable in a 
    particular treatmant can be found 
    using subscription brackets, e.g. **Earnings['as is']**. 

    A variable differs from other variable-like objects (e.g.
    stocks) in that it keeps no state. At any timestep, its amount depends 
    entirely on its specifier, and the amounts of dependencies. 

    The `specifier` is a callable, and is called once at each timestep for each
    treatment, using as arguments the amounts of the dependencies in
    that treatment.

    The amount of a variable in a treatment can be changed explicitly, outside 
    the model logic, e.g. **Earnings['as is'] = 2.1**. Once changed explicitly,
    the amount of 
    the variable never changes again, until the simulation is reset or 
    the amount is changed again explicitly. See examples below.

    Parameters
    ----------
    variable_name : str
        Name of the variable. The name is unique within a single model.

    description : str, optional
        Docstring-like description of the variable. 

    specifier : callable
        The specifier is called at every timestep.  Zero or more
        `dependencies` are supplied.  

    dependencies : list of str
        Names of variables (or constants or stocks or ...) used as arguments 
        for the callable `specifier`. 
        Might be empty, if callable requires no arguments.

    Returns
    -------
    Variable
        the newly-created variable

    See Also
    --------
    :class:`Variable` : a variable, once created

    constant : Create a variable whose amount does not change

    stock : Create a system dynamics stock

    previous : Create a variable for the prior amount of some other variable

    :class:`PerTreatment` : for defining how a variable has a different amount
        for each treatment

    Examples
    --------
    A variable can take a different amount every timestep, via a lambda ...

    >>> RandomValue = variable('RandomValue', lambda: random.random() + 1)

    ... or via any Python callable.

    >>> RandomValue = variable('RandomValue', random.random)

    The callable can depend on the amount of another variable in the model ...

    >>> DischargeProgress = variable(
    ...     'DischargeProgress', lambda db: (current_step - db) / 4,
    ...     'DischargeBegins')

    ... or depend on multiple variables.

    >>> Earnings = variable('Earnings', lambda r, c: r - c, 'Revenue', 'Cost')

    A variable can use different callables in different treatments.

    >>> DischargeEnds = variable('DischargeEnds',
    ...     PerTreatment(
    ...         {'As is': lambda db: db + 10, 'To be': lambda db: db + 5}),
    ...     DischargeBegins')

    An callable can use the model itself, instead of a variable
    in the model.

    >>> Time = variable('Time', lambda md: md.TIME, '__model__')
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
                  {
                    '__doc__': docstring, 
                    '_calculator': calc, 
                    '_validators': list(),
                    '_derived': {'derived': False, 'scored_as': 'basketball'},
                    '_has_history': True,
                    '_exclude_treatments': []
                  }
            )
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
    """
    Specify different amounts for each treatment.

    A variable's amount can be any Python object, except a string or a callable.
    But how to indicate that the amount differs across the treatments? Use this
    class. 

    Parameters
    ----------
    treatments_and_amounts : dict
        The keys of the dict are treatment names and the value are amounts

    Examples
    --------
    Create a constant whose amount differs by treatment.

    >>> constant('KitchenOpens', 
    ...     PerTreatment({'As is': 17.0, 'Open early': 15.5}))

    Create a variable whose amount is calculated from different expressions
    in each treatment.

    >>> variable('KitchenCloses',
    ...     PerTreatment(
    ...         {'As is': lambda lst: lst + 15, 
    ...          'Open early': lambda lst: lst}),
    ...     'LastCustomerOrders')
    """

    def __init__(self, treatments_and_amounts):
        """Initialize PerTreatment."""
        self._treatments_and_amounts = treatments_and_amounts

    def treatments_and_amounts(self):
        """Return all treatments and values, as a dict."""
        return self._treatments_and_amounts

    def by_treatment(self, treatment_name):
        """Return definition associate with treatment name."""
        try:
            return self._treatments_and_amounts[treatment_name]
        except KeyError:
            raise MinnetonkaError("Treatment '{}' not defined".format(
                treatment_name))

    def serialize_definition(self):
        """Return the serialization of the definition of this calculator."""
        return 'PerTreatment({{{}}})'.format(', '.join(map(
            lambda k, v: self._serialize_treatment(k, v), 
            self._treatments_and_amounts.keys(), 
            self._treatments_and_amounts.values())))

    def _serialize_treatment(self, k, v):
        """Serialize the key snd value so it can be an item in a larger dict."""
        try:
            return '"{}": {}'.format(k, v.serialize_definition())
        except:
            return '"{}": {}'.format(k, v)

def per_treatment(**kwargs):
    """Alternative syntax for PerTreatment."""
    return PerTreatment(kwargs)

#
# Defining constants
#


def constant(constant_name, *args):
    """
    constant(constant_name, [description,] specifier [, *dependencies])
    
    Create a constant.

    A constant is similar to a variable except that its amount does not vary. 
    Its amount is set on
    initialization, and then does not change over the course of the 
    simulation run. When the model is reset, the constant can take a
    new amount. 

    The amount of a constant can be any Python object, except a string or
    a Python callable. It can be defined in terms of other variables, 
    using a callable as the specifier. See examples below.

    A single constant can take a different amount in each
    model treatment. The amount of a constant in a particular treatment
    can be found using the subscription brackets, e.g. **Interest['to be']**.
    See examples below.

    The amount of a constant can be changed explicitly, outside the model
    logic, e.g. **Interest['to be'] = 0.07**. Once changed, the amount of
    the constant remains the same, until the model is reset
    or the amount is again changed explicitly. See examples below.

    Parameters
    ----------
    constant_name : str
        Name of the constant. The name is unique within a single model.

    description: str, optional
        Docstring-like description of the constant.

    specifier : callable or Any
        The specifier can be a callable. If a callable, it is called once, at 
        the beginning of the simulation run. Zero or more `dependencies` 
        are supplied, names of variables whose amounts are provided when the
        callable is called. If not a callable, `specifier` can be any Python
        object except a string, but no `dependencies` are supplied. If 
        not a callable, the specifier is provided as the amount at the beginning
        of the simulation run.

    dependencies : list of str
        Names of variables (or stocks or constants or ...) used as arguments 
        for the callable `specifier`. Empty
        list unless `specifier` is a callable. See examples below.

    Returns
    -------
    Constant
        the newly-created constant

    See Also
    --------
    :class:`Constant` : a constant, once created

    variable : Create a variable whose amount might change over the simulation

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
    """
    A constant, whose amount does not change.

    A constant is similar to a variable except that its amount does not vary. 
    Its amount is set on
    initialization, and then does not change over the course of the 
    simulation. When the model is reset, the constant can take a
    new amount. 

    The amount of a constant can be any Python object, except a string or
    a Python callable. It can be defined in terms of other variables, 
    using a callable in the definition. 

    A single constant can take a different amount in each
    model treatment. The amount of a constant in a particular treatment
    can be found using the subscription brackets, e.g. **InterestRate['to be']**.
    See examples below.

    The amount of a constant can be changed explicitly, outside the model
    logic, e.g. **InterestRate['to be'] = 0.07**. Once changed, the amount of
    the constant remains the same, until the model is reset
    or the amount is again changed explicitly. See examples below.

    See also
    --------
    constant : Create a :class:`Constant`

    :class:`Variable`: a variable whose amount might vary

    Examples
    --------
    Find the current amount of the constant **InterestRate**, in the **to be**
    treatment.

    >>> InterestRate['to be']
    0.08

    Change the current amount of the constant **InterestRate** in the **to be**
    treatment.

    >>> InterestRate['to be'] = 0.075

    Show everything important about the constant **InterestRate**.

    >>> InterestRate.show()
    Constant: InterestRate
    Amounts: {'as is': 0.09, 'to be': 0.075}
    Definition: PerTreatment({"as is": 0.09, "to be": 0.08})
    Depends on: []
    []
    """
    def all(self):
        """
        Return a dict of all current amounts, one for each treatment.

        Example
        -------
        >>> InterestRate.all()
        {'as is': 0.09, 'to be': 0.08}
        """
        return super().all()

    def history(self, treatment_name=None, step=None, base=False):
        """
        Return the amount at a past timestep for a particular treatment.

        Minnetonka tracks the past amounts of a constant
        over the course of a single simulation run,
        accessible with this function.  Of course, constants do not change
        value, except by explicit setting, outside of model logic. So 
        **history()** serves to return the history of those extra-model 
        changes.

        Parameters
        ----------
        treatment_name : str
            the name of some treatment defined in the model

        step : int
            the step number in the past 

        Example
        -------
        Create a model with a single constant InterestRate.

        >>> import random
        >>> with model(treatments=['as is', 'to be']) as m:
        ...     InterestRate = variable('InterestRate',
        ...         PerTreatment({"as is": 0.09, "to be": 0.08}))
        >>> InterestRate['to be']
        0.08

        Advance the simulation. InterestRate stays the same

        >>> m.step()
        >>> InterestRate['to be']
        0.08
        >>> m.step()
        >>> InterestRate['to be']
        0.08

        Change the amount of InterestRate explicitly.

        >>> InterestRate['to be'] = 0.075

        Find the old values of RandomVarable.

        >>> InterestRate.history('to be', 0)
        0.08
        >>> InterestRate.history('to be', 1)
        0.08
        >>> InterestRate.history('to be', 2)
        0.075
        """
        return super().history(
            treatment_name=treatment_name, step=step, base=base)

    def show(self):
        """
        Show everything important about the constant.

        Example
        -------
        >>> InterestRate.show()
        Constant: InterestRate
        Amounts: {'as is': 0.09, 'to be': 0.075}
        Definition: PerTreatment({"as is": 0.09, "to be": 0.08})
        Depends on: []
        []
        """
        return super().show()

    def __getitem__(self, treatment_name):
        """
        Retrieve the current amount of the constant in the treatment with
        the name **treatment_name**.

        Example
        -------
        Find the current amount of the constant **InterestRate**, in the **to be**
        treatment.

        >>> InterestRate['to be']
        0.08
        """
        return super().__getitem__(treatment_name)

    def __setitem__(self, treatment_name, amount):
        """
        Change the current amount of the variable in the treatment with the
        name **treatment_name**.

        Examples
        --------
        Change the current amount of the constant **InterestRate** in the **to be**
        treatment to **0.075**.

        >>> InterestRate['to be'] = 0.075

        Change the current amount of the constant **InterestRate** in all treatments
        at once.

        >>> InterestRate['__all__'] = 0.06
        """
        super().__setitem__(treatment_name, amount)

    def has_history(self):
        """A constant has no history."""
        return False

    def details(self):
        """Return a json-safe structure for the details of the constant."""
        deets = {"name": self.name(), "varies over time": False}
        amounts = self.all()
        if self.is_derived():
            derived_amounts = self.all_derived()
            amounts = {**amounts, **derived_amounts}
        if hasattr(self, '_summarizer'):
            self._add_summary(deets, amounts)
        elif hasattr(self, "_suppress_amount") and self._suppress_amount: 
            self._add_summary_description_only(deets)
        else:
            self._add_amount(deets, amounts)
        return deets

    def _add_summary(self, deets, amounts):
        """Add a summary to the deets."""
        summary = {
            trt: self._summarizer(amt, trt) for trt, amt in amounts.items()}
        deets['summary'] = summary
        deets['summary description'] = self._summary_description
        deets['caucus'] = summary

    def _add_amount(self, deets, amounts):
        """Add amounts to deets"""
        deets['amount'] = amounts
        deets['caucus'] = amounts 

    def all_derived(self):
        """Return a dict of all derived treatments."""
        if self.is_derived():
            return {trt_name: self[trt_name]
                    for trt_name in self._model.derived_treatments()
                    if self.derived_treatment_defined(trt_name)}
        else:
            return {}


class ConstantInstance(VariableInstance, metaclass=Constant):
    """A variable that does not vary."""

    def _step(self):
        pass

    def _history(self, step=None):
        """No history for a constant. Everything is the current value."""
        if step is None:
            return [self.amount()]
        else:
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

    def _is_scored_as_combo(self):
        """Is this variable scored as a combo of golf and basketball?""" 
        # Incrementers cannot be scored as a combo because they keep state
        return False

    def recalculate_all(self):
        """Recalculdate all the variable instances, without changing step."""
        not_yet_stepped = self._model.STEP == 0
        for var in self._by_treatment.values():
            if not var.undefined:
                var._recalculate(not_yet_stepped)

class IncrementerInstance(CommonVariableInstance, metaclass=Incrementer):
    """A variable instance with internal state, that increments every step."""

    def _reset(self, external_vars):
        """Reset to beginning of simulation."""
        self.set_initial_amount(self._treatment.name)

    def set_initial_amount(self, treatment_name):
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

    def _recalculate(self, not_yet_stepped):
        """Recalculate without advancing a step."""
        if not_yet_stepped:
            self.set_initial_amount(self._treatment.name)
        else:
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
    """
    A system dynamics stock.

    In `system dynamics <https://en.wikipedia.org/wiki/System_dynamics>`_,
    a stock is used to model something that accumulates or depletes over
    time. 

    At any simulated period, the stock has an amount. The amount changes
    over time, incrementing or decrementing at each timestep. The amount
    can be a simple numeric like a Python integer or a Python float. 
    Or it might be some more complex Python object: a list,
    a tuple, a numpy array, or an instance of a user-defined class. In 
    any case, the stock's amount must support addition and multiplication. 
    (Addition and multiplication are supported
    for dicts, tuples, and named tuples via :func:`foreach`.)

    If the model in which the stock lives has multiple treatments, 
    the stock may have several amounts, one for each treatment. The amount of
    a stock in a particular treatment can be accessed using subscription 
    brackets, e.g. **Savings['to be']**.

    The amount of a stock in a treatment can be changed explicitly, outside
    the model logic, e.g. **Savings['to be'] = 16000**. Once changed explicitly,
    the amount of the stock never changes again (in that treatment),
    until the simulation is reset or the amount is changed again explicitly.

    See Also
    --------
    stock : Create a :class:`Stock`  

    :class:`Variable` : a variable whose amount is calculated from other vars

    :class:`Constant` : a variable that does not vary

    :class:`Previous` : a variable that has the previous amount of some other
        variable

    :class:`Accum`: a stock-like variable that uses current amounts

    Examples
    --------
    Find the current amount of the stock **Savings**, in the **to be** 
    treatment.

    >>> Savings['to be']
    16288.94

    Change the current amount of the stock **Savings** in the **to be**
    treatment.

    >>> Savings['to be'] = 16000

    Show everything important about the stock **Savings**.

    >>> Savings.show()
    Stock: Savings 
    Amounts: {'as is': 14802.442849183435, 'to be': 16000}
    Initial definition: 10000.0
    Initial depends on: []
    Incremental definition: Savings = stock('Savings', lambda i: i, ('Interest',), 10000.0)
    Incremental depends on: ['Interest']
    [variable('Interest')]
    """

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

    def all(self):
        """
        Return a dict of all current amounts, one for each treatment.

        Example
        -------
        >>> Savings.all()
        {'as is': 14090, 'to be': 16000}
        """
        return super().all()

    def history(self, treatment_name=None, step=None, base=False):
        """
        Return the amount at a past timestep for a particular treatment.

        Minnetonka tracks the past amounts of a stock
        over the course of a single simulation run,
        accessible with this function. 

        Parameters
        ----------
        treatment_name : str
            the name of some treatment defined in the model

        step : int
            the step number in the past 

        Example
        -------
        Create a model with a single stock **Year**.

        >>> with model() as m:
        ...     Year = stock('Year', 1, 2019)
        >>> Year['']
        2019

        Advance the simulation. **Year** changes value.

        >>> m.step()
        >>> Year['']
        2020
        >>> m.step()
        >>> Year['']
        2021

        Find the old values of **Year**

        >>> Year.history('', 0)
        2019
        >>> Year.history('', 1)
        2020
        """
        return super().history(
            treatment_name=treatment_name, step=step, base=base)

    def show(self):
        """
        Show everything important about the stock.

        Example
        -------
        >>> Savings.show()
        Stock: Savings 
        Amounts: {'as is': 14802.442849183435, 'to be': 16000} 
        Initial definition: 10000.0
        Initial depends on: [] 
        Incremental definition: Savings = stock('Savings', lambda i: i, ('Interest',), 10000.0)
        Incremental depends on: ['Interest'] 
        [variable('Interest')]
        """
        return super().show()

    def __getitem__(self, treatment_name):
        """
        Retrieve the current amount of the stock in the treatment with
        the name **treatment_name**.

        Example
        -------
        Find the current amount of the stock **Savings**, in the **as is**
        treatment.

        >>> Savings['as is']
        14802.442849183435
        """
        return super().__getitem__(treatment_name)

    def __setitem__(self, treatment_name, amount):
        """
        Change the current amount of the stock in the treatment with the
        name **treatment_name**.

        Examples
        --------
        Change the current amount of the stock **Savings** in the **as is**
        treatment to **2.1**.

        >>> Savings['as is'] = 14000

        Change the current amount of the stock **Taxes** in all treatments
        at once.

        >>> Savings['__all__'] = 10000
        """
        super().__setitem__(treatment_name, amount)


class StockInstance(IncrementerInstance, metaclass=Stock):
    """A instance of a system dynamics stock for a particular treatment."""

    def _calculate_increment(self, timestep):
        """Compute the increment."""
        full_step_incr = self._incremental.calculate(
            self._treatment.name,
            [v.amount() for v in self._increment_depends_on_instances])
        self._increment_amount = self._incremental.multiply(
            full_step_incr, timestep)

    def _step(self):
        """Advance the stock by one step."""
        self._amount = self._incremental.add(
            self._amount, self._increment_amount)

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
    stock(stock_name, [description,] increment [,[increment_dependencies,] initial [, initial_dependencies]]) 

    Create a system dynamics stock.

    In `system dynamics <https://en.wikipedia.org/wiki/System_dynamics>`_,
    a stock is used to model something that accumulates or depletes over
    time. The stock defines both an initial amount and an increment.

    At any simulated period, the stock has an amount. The amount changes
    over time, incrementing or decrementing at each timestep. The amount
    can be a simple numeric like a Python integer or a Python float. 
    Or it might be some more complex Python object: a list,
    a tuple, a numpy array, or an instance of a user-defined class. In 
    any case, the stock's amount must support addition and multiplication.
    (Addition and multiplication are supported
    for dicts, tuples, and named tuples via :func:`foreach`.)

    If the model in which the stock lives has multiple treatments, 
    the stock may have several amounts, one for each treatment. The amount of
    a stock in a particular treatment can be accessed using subscription 
    brackets, e.g. **Savings['to be']**.

    A stock definition has two parts: an initial and an increment.
    The initial is either a callable or any non-callable Python object 
    except a string. If a callable, the initial has a (possibly empty) tuple 
    of dependencies. If a non-callable, `initial_dependencies` is an
    empty tuple.

    If `initial` is a callable,
    that callable is called once for each treatment, at model initialization, 
    with the initial amount of each of the dependencies for that
    treatment. The names of
    the dependencies are provided: `initial_dependencies` is a tuple of strings.
    Each dependency named can either be a (plain) variable (i.e. an instance
    of :class:`Variable`) or a stock or a constant
    or any of the other variable elements of a model. The result of the 
    execution of the callable becomes the initial amount of the stock, for
    that treatment.

    The stock increment is also either a callable or any non-callable Python
    object except a string. If a callable, the increment has a (possibly empty)
    tuple of dependencies. If a non-callable, `increment_dependencies` is an
    empty tuple. 

    If `increment` is a callable, the callable is called once every
    period for each treatment, using as arguments the amounts of each of the 
    dependencies in that treatment. Each
    dependency can be the name of a (plain) variable (i.e. an instance
    of :class:`Variable`) or a stock or a constant or any of the
    variable elements of a model. The callable is given the amounts of the 
    variables at the previous timestep, not the current timestep, to 
    determine the increment of the stock for this period. 

    The increment is how much the stock's amount changes in each unit of time.
    If the timestep of the model is 1.0, the stock's amount will
    change by exactly that increment. If the timestep is not 1.0, the stock's 
    amount will change by a different quantity. For example, if the timestep
    is 0.5, the stock's amount will change by half the increment, at
    every step. (For more on the timestep, see :func:`model`.)

    The initial amount and the increment amount may vary by treatment, either
    because one or more of the the dependencies vary by treatment,
    or because of an explicit :class:`PerTreatment` expression. See examples
    below.

    The amount of a stock in a treatment can be changed explicitly, outside
    the model logic, e.g. **Savings['to be'] = 1000**. Once changed explicitly,
    the amount of the stock never changes again (in that treatment),
    until the simulation is reset or the amount is changed again explicitly.

    Parameters
    ----------
    stock_name : str
        Name of the stock. The name must be unique within the model.

    description : str, optional
        Docstring-like description of the stock.

    increment : callable or Any
        The increment can be either a callable or any Python object, except a 
        string. If a callable, the increment is called once for each treatment
        at every timestep, with arguments the amounts of
        `increment_dependencies` in that treatment. The result of the callable 
        execution for
        a single treatment is the unit time change in amount for that treatment.
        See examples below.

        If `increment` is not a callable, it is interpreted as the unit time 
        change in amount, unchanging with each timestep. 

        Using :class:`PerTreatment`, a different amount or different callable 
        can be provided for different treatments. See examples below.

    increment_dependencies : tuple of str, optional
        Names of dependencies---i.e. names of (plain) variables or constants or 
        other stocks or ...---
        used as arguments for the callable `increment`. Might be an empty tuple, 
        the default, either
        if callable `increment` requires no arguments, or if `increment` is not
        a callable.

    initial: callable or Any, optional
        The initial can be either a callable or any Python object, except a 
        string. If a callable, the initial is called once for each treatment
        at the beginning of the simulation, with arguments of the amounts of 
        `initial_dependencies`. The results of the callable execution for a single
        treatment becomes the initial amount of the stock, for that treatment.

        If `initial` is not a callable, it is interpreted as the initial amount
        for the stock. 

        Using :class:`PerTreatment`, a different amount or different callable 
        can be provided for different treatments. See examples below.

    initial_dependencies: tuple of str, optional
        Names of dependencies---i.e. names of (plain) variables or constants or 
        other stocks or ...---
        used as arguments for the callable `initial`. Might be an empty tuple, 
        the default, either
        if callable `initial` requires no arguments, or if `increment` is not
        a callable.

    Returns
    -------
    Stock
        the newly-created stock

    See Also
    --------
    variable : Create a variable whose amount might change

    constant : Create a variable whose amount does not change

    accum : Create an accum, much like a stock except that it uses the 
        amounts of the variables in the current period, instead of the 
        previous period.

    :class:`PerTreatment` : for defining how an increment or initial
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

    The increment can be a callable with no dependencies. Note the empty
    tuple of dependencies.

    >>> stock('MenuItemCount', lambda: random.randint(0,2), (), 20)

    The initial amount can be a callable. If the initial amount is a 
    callable, the increment must also be a callable. Note the empty tuples.

    >>> stock('MenuItemCount', 
    .,,     lambda: random.randint(15, 18), (),
    ...     lambda: random.randint(20, 22), ())

    Dependencies can be provided for the increment callable.

    >>> stock('Savings', lambda interest: interest, ('Interest',), 0)

    Dependencies can be provided for the initial callable.

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
    incr_vars = _maybe_correct_vars(incr_vars)
    init_vars = _maybe_correct_vars(init_vars)
    return _create_stock(
        stock_name, docstring, incr_def, incr_vars, init_def, init_vars)

def _maybe_correct_vars(vars):
    """Change vars from string to singleton tuple of string, if necessary."""
    if isinstance(vars, str):
        return (vars,)
    else:
        return vars 

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
                  increment_definition, increment_dependencies,
                  initial_definition, initial_dependencies):
    """Create a new stock."""
    initial = create_calculator(initial_definition, initial_dependencies)
    incr = create_calculator(increment_definition, increment_dependencies)
    newstock = type(stock_name, (StockInstance,),
                    {
                        '__doc__': docstring, 
                        '_initial': initial,
                        '_incremental': incr,
                        '_validators': list(),
                        '_derived': {'derived': False},
                        '_has_history': True,
                        '_exclude_treatments': []
                    }
                )
    Model.add_variable_to_current_context(newstock)
    return newstock


#
# Accum class
#

class Accum(Incrementer):
    """
    A stock-like incrementer, with a couple of differences from a stock.

    An accum is much like a :class:`Stock`, modeling something that
    accumulates or depletes over time. Like a stock, an accum defines
    both an initial amount and an increment.

    There is an important difference between a stock and an accum: an accum 
    is incremented with the current amounts
    of its dependencies, not the amounts in the last period. 
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
    (Addition is supported
    for dicts, tuples, and named tuples via :func:`foreach`.)

    If the model in which the accum lives has multiple treatments, the 
    accum may have several amounts, one for each treatment in the model. The 
    amount of an accum in a particular treatment can be accessed using 
    subscription brackets, e.g. **RevenueYearToDate['as is']**.

    The amount of an accum in a treatment can be changed explicitly, outside
    the model logic, e.g. **RevenueYearToDate['as is'] = 1000**. 
    Once changed explicitly,
    the amount of the accum never changes again (in that treatment),
    until the simulation is reset or the amount is changed again explicitly.

    See Also
    --------
    accum : Create an :class:`Accum`

    :class:`Stock`: a system dynamics stock

    :class:`Variable` : a variable whose amount is calculated from other vars

    :class:`Constant` : a variable that does not vary

    :class:`Previous` : a variable that has the previous amount of some other
        variable

    Examples
    --------
    Find the current amount of the accum **RevenueYearToDate** in the 
    **cautious** treatment.

    >>> RevenueYearToDate['cautious']
    224014.87326935912

    Change the current amount of the accum **RevenueYearToDate** in the 
    **cautious** treatment.

    >>> RevenueYearToDate['cautious'] = 200000

    Show everything important about the accum **RevenueYearToDate**

    >>> RevenueYearToDate.show()
    Accum: RevenueYearToDate
    Amounts: {'as is': 186679.06105779926, 'cautious': 200000, 'aggressive': 633395.3052889963}
    Initial definition: 0
    Initial depends on: []
    Incremental definition: RevenueYearToDate = accum('RevenueYearToDate', lambda x: x, ('Revenue',), 0)
    Incremental depends on: ['Revenue']
    [variable('Revenue')]
    """
    def _check_for_cycle_in_depends_on(cls, checked_already, dependents=None):
        """Check for cycles involving this accum."""
        for dname in cls.depends_on(for_init=True):
            d = cls._model.variable(dname)
            d.check_for_cycle(checked_already, dependents=dependents)
        for dname in cls.depends_on(for_init=False):
            d = cls._model.variable(dname)
            d.check_for_cycle(checked_already, dependents=dependents)

    def all(self):
        """
        Return a dict of all current amounts, one for each treatment.

        Example
        -------
        >>> RevenueYearToDate.all()
        {'as is': 186679.06105779926,
         'cautious': 224014.87326935912,
         'aggressive': 633395.3052889963}
        """
        return super().all()

    def history(self, treatment_name=None, step=None, base=False):
        """
        Return the amount at a past timestep for a particular treatment.

        Minnetonka tracks the past amounts of an accum
        over the course of a single simulation run,
        accessible with this function. 

        Parameters
        ----------
        treatment_name : str
            the name of some treatment defined in the model

        step : int
            the step number in the past 

        Example
        -------
        Create a model with an accum and three treatments

        >>> with model(treatments=['as is', 'cautious', 'aggressive']) as m:
        ...     RevenueYearToDate = accum('RevenueYearToDate', 
        ...         lambda x: x, ('Revenue',), 0)
        ...     Revenue = variable('Revenue', 
        ...         lambda lst, mst, w: lst + w * (mst - lst),
        ...         'Least', 'Most', 'Weather')
        ...     Weather = variable('Weather', 
        ...         lambda: random.random())
        ...     Least = constant('Least', 
        ...         PerTreatment(
        ...             {'as is': 0, 'cautious': 0, 'aggressive': -100000}))
        ...     Most = constant('Most', 
        ...         PerTreatment(
        ...             {'as is': 100000, 'cautious': 120000, 
        ...              'aggressive': 400000}))

        Advance the simulation. **RevenueYearToDate** changes value.

        >>> m.step()
        >>> RevenueYearToDate['aggressive']
        240076.8319119932
        >>> m.step()
        >>> RevenueYearToDate['aggressive']
        440712.80369068065
        >>> m.step()
        >>> RevenueYearToDate['aggressive']
        633395.3052889963

        Find the old values of **RevenueYearToDate**

        >>> RevenueYearToDate.history('aggressive', 1)
        240076.8319119932
        >>> RevenueYearToDate.history('aggressive', 2)
        440712.80369068065
        """
        return super().history(
            treatment_name=treatment_name, step=step, base=base)

    def show(self):
        """
        Show everything important about the accum.

        Example
        -------
        >>> RevenueYearToDate.show()
        Accum: RevenueYearToDate
        Amounts: {'as is': 186679.06105779926, 'cautious': 200000, 'aggressive': 633395.3052889963}
        Initial definition: 0
        Initial depends on: []
        Incremental definition: RevenueYearToDate = accum('RevenueYearToDate', lambda x: x, ('Revenue',), 0)
        Incremental depends on: ['Revenue']
        [variable('Revenue')]
        """
        return super().show()

    def __getitem__(self, treatment_name):
        """
        Retrieve the current amount of the accum in the treatment with
        the name **treatment_name**.

        Example
        -------
        Find the current amount of the accum **RevenueYearToDate**, 
        in the **as is** treatment.

        >>> RevenueYearToDate['as is']
        186679.06105779926
        """
        return super().__getitem__(treatment_name)

    def __setitem__(self, treatment_name, amount):
        """
        Change the current amount of the accum in the treatment with the
        name **treatment_name**.

        Examples
        --------
        Change the current amount of the accum **RevenueYearToDate** 
        in the **as is** treatment to **2.1**.

        >>> RevenueYearToDate['as is'] = 190000

        Change the current amount of the accum **RevenueYearToDate** 
        in all treatments at once.

        >>> RevenueYearToDate['__all__'] = 0
        """
        super().__setitem__(treatment_name, amount)


class AccumInstance(IncrementerInstance, metaclass=Accum):
    """Like ACCUM in SimLang, for a particular treatment instance."""

    def _step(self):
        """Advance the accum by one step."""
        increment = self._incremental.calculate(
            self._treatment.name,
            [v.amount() for v in self._increment_depends_on_instances]
            )
        self._amount = self._incremental.add(self._amount, increment)

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


def accum(accum_name, *args):
    """
    accum(accum_name, [description,] increment [,[increment_dependencies,] initial [, initial_dependencies]]) 

    Create a system dynamics accum. 

    An accum is much like a :class:`Stock`, modeling something that
    accumulates or depletes over time. Like a stock, an accum defines
    both an initial amount and an increment.

    There is an important difference between a stock and an accum: an accum 
    is incremented with the current amounts
    of its dependencies, not the amounts in the last period. 
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
    (Addition is supported
    for dicts, tuples, and named tuples via :func:`foreach`.)

    If the model in which the accum lives has multiple treatments, the 
    accum may have several amounts, one for each treatment in the model. The 
    amount of an accum in a particular treatment can be accessed using 
    subscription brackets, e.g. **RevenueYearToDate['as is']**.

    An accum definition has two parts: an initial and an increment.
    The initial is either a callable or any non-callable Python object
    except a string. If a callable, the initial has a (posssibly empty) tuple
    of dependencies. If a non-callable, `iniitial_dependences` is an 
    empty tuple.

    If `initial` is a callable,
    that callable is called once for each treatment, at model initialization, 
    with the initial amounts of each of the dependencies. The names of
    the dependencies are provided: `initial_dependencies` is a tuple of strings.
    Each dependency named can either be a (plain) variable (i.e. an instance
    of :class:`Variable`) or a stock or a constant
    or any of the other variable elements of a model. The result of the 
    execution of the callable becomes the initial amount of the accum, for
    that treatment.

    The accum increment is also either a callable or any non-callable Python
    object except a string. If a callable, the increment has a (possibly empty)
    tuple of dependencies. If a non-callable, `increment_dependencies` is an
    empty tuple.    

    If `increment` is a callable, the callable is called once every
    period for each treatment, using as arguments the amounts of each of the 
    dependencies in that treatment. Each
    dependency can be the name of a (plain) variable (i.e. an instance
    of :class:`Variable`) or a stock or a constant or any of the
    variable elements of a model.  The callable is given the amounts of the 
    variables at the current period, to 
    determine the increment of the accume for this period. 

    The increment is how much the accum's amount changes in each period.
    Note that this is another difference between an accum and a stock: 
    for a stock the amount incremented depends on the timestep; for an 
    accum it does not. For example, if both a stock **S** and an accum **A**
    have an increment of 10, and the timestep is 0.5, **S** will increase
    by 5 every period but **A** will increase by 10.

    The initial amount and the increment amount may vary by treatment, either
    because one or more of the the dependencies vary by treatment,
    or because of an explicit :class:`PerTreatment` expression. 

    The amount of an accum in a treatment can be changed explicitly, outside
    the model logic, e.g. **RevenueYearToDate['as is'] = 1000**. 
    Once changed explicitly,
    the amount of the accum never changes again (in that treatment),
    until the simulation is reset or the amount is changed again explicitly.

    Parameters
    ----------
    accum_name : str
        Name of the accum. The name must be unique within the model.

    description : str, optional
        Docstring-like description of the accum.

    increment : callable or Any
        The increment can be either a callable or any Python object, except a 
        string. If a callable, the increment is called once for each treatment
        at every timestep, with arguments the amounts of
        `increment_dependencies` in that treatment. The result of the callable 
        execution for
        a single treatment is the change in amount for that treatment.
        See examples below.

        If `increment` is not a callable, it is interpreted as the 
        change in amount, unchanging with each timestep. 

        Using :class:`PerTreatment`, a different amount or different callable 
        can be provided for different treatments. See examples below.

    increment_dependencies : tuple of str, optional
        Names of dependencies---i.e. names of (plain) variables or constants or 
        other stocks or ...---
        used as arguments for the callable `increment`. Might be an empty tuple, 
        the default, either
        if callable `increment` requires no arguments, or if `increment` is not
        a callable.  

    initial : callable or Any, optional
        The initial can be either a callable or any Python object, except a 
        string. If a callable, the initial is called once for each treatment
        at the beginning of the simulation, with arguments of the amounts of 
        `initial_dependencies`. The results of the callable execution for a single
        treatment becomes the initial amount of the stock, for that treatment.

        If `initial` is not a callable, it is interpreted as the initial amount
        for the stock. 

        Using :class:`PerTreatment`, a different amount or different callable 
        can be provided for different treatments. See examples below.

    initial_dependencies: tuple of str, optional
        Names of dependencies---i.e. names of (plain) variables or constants or 
        other stocks or ...---
        used as arguments for the callable `initial`. Might be an empty tuple, 
        the default, either
        if callable `initial` requires no arguments, or if `increment` is not
        a callable.

    Returns
    -------
    Accum
        the newly-created accum

    See Also
    --------
    variable : Create a non-stock variable

    constant : Create a non-stock variable whose amount does not change

    stock : Create an stock, much like a stock except that it uses the 
        amounts of the variables in the prior period, instead of the current
        period

    :class:`PerTreatment` : for defining how an increment or initial
        varies from treatment to treatment

    Examples
    --------
    An accum that collects all the revenue to date.

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
    incr_vars = _maybe_correct_vars(incr_vars)
    init_vars = _maybe_correct_vars(init_vars)
    return _create_accum(
        accum_name, docstring, incr_def, incr_vars, init_def, init_vars)


def _create_accum(accum_name, docstring,
                  increment_definition=0, increment_dependencies=None,
                  initial_definition=0, initial_dependencies=None):
    """Create a new accum."""
    initial = create_calculator(initial_definition, initial_dependencies)
    increment = create_calculator(increment_definition, increment_dependencies)
    new_accum = type(accum_name, (AccumInstance,),
                     {
                        '__doc__': docstring, 
                        '_initial': initial,
                        '_incremental': increment,
                        '_validators': list(),
                        '_derived': {'derived': False},
                        '_has_history': True,
                        '_exclude_treatments': []
                    }
                )
    Model.add_variable_to_current_context(new_accum)
    return new_accum


#
# previous: a variable that accesses previous value of another variable
#

class Previous(CommonVariable):
    """
    A previous.

    A previous is a variable whose amount is that of some other variable in 
    the prior timestep. A previous allows a reach into the past from within
    the model.

    If the model in which the previous lives has multiple treatments, and its
    prior has a different amount for each treatment, so will the previous.
    The amount of the previous in a particular treatment can be accessed
    using subscription brackets, e.g. **YesterdaySales['as is']**.

    See Also
    --------
    previous : Create a :class:`Previous`

    :class:`Variable` : a variable whose amount is calculated from other vars

    Examples
    --------
    Find yesterday's sales, when the timestep is one day.

    >>> YesterdaySales['as is']
    13

    Show everything important about the previous **YesterdaySales**.

    >>> YesterdaySales.show()
    Previous: YesterdaySales
    Amounts: {'as is': 13, 'to be': 9}
    Previous variable: Sales
    [variable('Sales')]
    """
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

    def all(self):
        """
        Return a dict of all current amounts, one for each treatment.

        Example
        -------
        >>> PreviousEarnings.all()
        {'as is': 1.9, 'to be': 2.4}
        """
        return super().all()

    def history(self, treatment_name=None, step=None, base=False):
        """
        Return the amount at a past timestep for a particular treatment.

        Minnetonka tracks the past amounts of a previous
        over the course of a single simulation run,
        accessible with this function. 

        Parameters
        ----------
        treatment_name : str
            the name of some treatment defined in the model

        step : int
            the step number in the past 

        Example
        -------
        Create a model with a stock **Year**, and a previous **LastYear**.

        >>> with model() as m:
        ...     Year = stock('Year', 1, 2019)
        ...     LastYear = previous('LastYear', 'Year', None)

        Advance the simulation ten years.

        >>> m.step(10)

        Find the value of both **Year** and **LastYear** in year 5.

        >>> Year.history('', 5)
        2024
        >>> LastYear.history('', 5)
        2023
        """
        return super().history(
            treatment_name=treatment_name, step=step, base=base)

    def show(self):
        """
        Show everything important about the previous.

        Example
        -------
        >>> YesterdaySales.show()
        Previous: YesterdaySales
        Amounts: {'as is': 13, 'to be': 9}
        Previous variable: Sales
        [variable('Sales')]
        """
        return super().show()

    def __getitem__(self, treatment_name):
        """
        Retrieve the current amount of the previous in the treatment with
        the name **treatment_name**.

        Example
        --------
        Find the current amount of the variable **PriorEarnings**, in the 
        **as is** treatment.

        >>> PriorEarnings['as is']
        1.9
        """
        return super().__getitem__(treatment_name)

    def set(self, treatment_name, amount):
        """An error. Should not set a previous."""
        raise MinnetonkaError(
            'Amount of {} cannot be changed outside model logic'.format(self))


class PreviousInstance(SimpleVariableInstance, metaclass=Previous):
    """A variable that takes the previous amount of another variable."""

    def wire_instance(self, model, treatment_name):
        """Set the variable this instance depends on."""
        self._previous_instance = model.variable_instance(
            self._earlier, treatment_name)

    def _calculate_amount(self):
        """Calculate the current amount of this previous."""
        if self.undefined:
            return None 
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
    previous(previous_name, [description,], prior [, initial_amount])

    Create a previous.

    Create a new previous, a variable whose amount is the amount of another
    variable---the one named by `prior`---in the previous timestep. 

    If the model in which the previous lives has multiple treatments, and its
    prior has a different amount for each treatment, so will the previous.
    The amount of the previous in a particular treatment can be accessed
    using subscription brackets, e.g. **YesterdaySales['as is']**.

    When the model is initialized, the amount of the previous is either set
    to `initial_amount`, or if no initial amount is provided, it is set to
    the amount of `prior`.

    Parameters
    ----------
    variable_name : str
        Name of the previous. The name must be unique within a single model.

    description : str, optional
        Docstring-like description of the previous.

    prior : str
        The name of a variable (or constant or stock or ...). The amount of
        the prior in the last timestep becomes the new amount of the previous
        in this timestep. 

    initial_amount : Any, optional
        Any non-string and non-callable Python object. But typically this is 
        some kind of numeric: an int or a float or a numpy array of floats or 
        the like. If provided, when the model is initialized, the initial 
        amount of `prior` is set to `initial_amount`.

    Returns
    -------
    Previous
        the newly created previous

    See Also
    --------
    variable : Create a variable whose amount might change

    constant : Create a variable whose amount does not change

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
                  {
                    '__doc__': docstring, 
                    '_earlier': earlier_var_name,
                    '_init_amount': init_amount,
                    '_validators': list(),
                    '_derived': {'derived': False},
                    '_has_history': True,
                    '_exclude_treatments': []
                    }
                )
    Model.add_variable_to_current_context(newvar)
    return newvar

#
# cross: a variable that takes the amount of another variable in another 
# treatment
#

class Cross(CommonVariable):
    """A variable that takes the amount of another in a particular treatment"""
    def _check_for_cycle_in_depends_on(self, checked_already, dependents):
        """Check for cycles among the depends on for this cross"""
        reference = self._model.variable(self._referenced_variable)
        reference.check_for_cycle(checked_already, dependents=dependents)

    def _show_definition_and_dependencies(self):
        """Print the definition and the variables it depends on."""
        print('Cross variable: {} in treatment {}'.format(
            self._referenced_variable, self._referenced_treatment))

    def antecedents(self, ignore_pseudo=False):
        """Return all the depends_on_variables."""
        return [self._model[self._referenced_variable]]

    def has_unitary_definition(self):
        """Return whether the cross has a unitary definition."""
        return True

    def set(self, treatment_name, amount):
        """An error. Should not set a cross"""
        raise MinnetonkaError(
            'Amount of {} cannot be changed outside model logic'.format(
                self))

class CrossInstance(SimpleVariableInstance, metaclass=Cross):
    """A variable that takes the amount of another var in a particular trtmt"""
    def wire_instance(self, model, treatment_name):
        """Set the variable this instances depends on."""
        del(treatment_name)
        self._cross_instance = model.variable_instance(
            self._referenced_variable, self._referenced_treatment)

    def _calculate_amount(self):
        """Calculate the current amount of this cross."""
        if self.undefined:
            return None 
        else:
            return self._cross_instance.amount()

    @classmethod
    def depends_on(cls, for_init=False, for_sort=False, ignore_pseudo=False):
        """Return the variables this variable depends on."""
        return [cls._referenced_variable]


def cross(variable_name, referenced_variable_name, treatment):
    """For pulling the amount from a different treatment."""
    return _create_cross(variable_name, '', referenced_variable_name, treatment)

def _create_cross(
        variable_name, docstring, referenced_variable_name, treatment):
    newvar = type(variable_name, (CrossInstance,), {
            '__doc__': docstring,
            '_referenced_variable': referenced_variable_name,
            '_referenced_treatment': treatment,
            '_validators': list(),
            '_derived': {'derived': False},
            '_has_history': True,
            '_exclude_treatments': []
        })
    Model.add_variable_to_current_context(newvar)
    return newvar

#
# derivn: a variable that is the (first order) derivative of another
#

class Velocity(CommonVariable):
    """A variable that is the (first order) derivative of another."""

    def _check_for_cycle_in_depends_on(self, checked_already, dependents):
        """Check for cycles among the depends on."""
        self._model[self._position_varname].check_for_cycle(
            checked_already, dependents=dependents)

    def _show_definition_and_dependencies(self):
        """Print the definition and variables it depends on."""
        print('First order derivative of: {}'.format(self._position_varname))

    def antecedents(self, ignore_pseudo=False):
        """Return all the depends_on variables."""
        return [self._model[self._position_varname]]

    def has_unitary_definition(self):
        """Returns whether the velocity has a unitary definition."""
        return True

    def calculate_all_increments(self, timestep):
        """Capture the timestep and last value of position."""
        # This is a bit of a hack, but when stocks are calculating increments
        # it is a good time to capture the last position and the time step
        for var in self.all_instances():
            var.capture_position(timestep)

    def set(self, treatment_name, amount):
        """An error. Should not set a velocity."""
        raise MinnetonkaError(
            'Amount of {} cannot be changed outside model logic'.format(self))


class VelocityInstance(SimpleVariableInstance, metaclass=Velocity):
    """A variable that is the (first order) derivative of another variable."""

    def wire_instance(self, model, treatment_name):
        """Set the variable this instance depends on."""
        self._position_instance = model.variable_instance(
            self._position_varname, treatment_name)

    def capture_position(self, timestep):
        """Capture the current position (soon to be last position) + timestep"""
        self._timestep = timestep 
        self._last_position = self._position_instance.amount() 

    def _calculate_amount(self):
        """Calculate the current amount of this velocity."""
        if self.undefined:
            return None 
        current_position = self._position_instance.amount()
        if current_position is None:
            return 0
        elif self._last_position is None:
            return self._zero(current_position)
        else:
            step_incr = self.subtract(current_position, self._last_position)
            return self.divide(step_incr, self._timestep)

    def _zero(self, obj):
        """Return the zero with the same shape as obj."""
        if isinstance(obj, int): 
            return 0
        elif isinstance(obj, float):
            return 0.0
        elif isinstance(obj, np.ndarray):
            return np.zeros(obj.shape)
        elif isinstance(obj, dict):
            return {k: self._zero(v) for k, v in obj.items()} 
        elif isnamedtuple(obj) or isinstance(obj, MinnetonkaNamedTuple):
            typ = type(obj)
            return typ(*(self._zero(o) for o in obj))
        elif isinstance(obj, tuple):
            return tuple(self._zero(o) for o in obj)
        else:
            raise MinnetonkaError(
                'Do not know how to find initial velocity of {}'.format(obj) +
                'as it is {}'.format(type(obj)))

    def subtract(self, minuend, subtrahend):
        """Subtract subtrahend from minuend."""
        try:
            return minuend - subtrahend
        except TypeError:
            fn = self._across_fn(minuend)
            return fn(minuend, subtrahend, self.subtract)

    def divide(self, dividend, divisor):
        """Subtract dividend by divisor."""
        try:
            return dividend / divisor
        except TypeError:
            fn = self._across_fn(dividend)
            return fn(dividend, divisor, self.divide)

    def _across_fn(self, obj):
        """Return function that applies another function across collection."""
        if isinstance(obj, dict):
            return self._across_dicts
        elif isnamedtuple(obj):
            return self._across_named_tuples
        elif isinstance(obj, tuple):
            return self._across_tuples
        else:
            raise MinnetonkaError(
                'Velocity argument {} must be numeric, dict, '.format(obj) +
                'tuple, or numpy array, not {}'.format(type(obj)))

    def _across_dicts(self, arg1, arg2, fn):
        """arg1 is a dict. Apply fn to it and arg2."""
        try: 
            return {k: fn(v, arg2[k]) for k,v in arg1.items()}
        except TypeError:
            # arg2 might be constant rather than a dict
            return {k: fn(v, arg2) for k,v in arg1.items()}

    def _across_named_tuples(self, arg1, arg2, fn):
        """arg1 is an ordinary named tuple. Apply fn to it and arg2."""
        try:
            typ = type(arg1)
            return typ(*(fn(a1, a2) for a1, a2 in zip(arg1, arg2)))
        except TypeError:
            # arg2 might be constant rather than a namedtuple
            return typ(*(fn(a1, arg2) for a1 in arg1))

    def _across_tuples(self, arg1, arg2, fn):
        """arg1 is a tuple. Apply fn to it and arg2"""
        try: 
            return tuple(fn(a1, a2) for a1, a2 in zip(arg1, arg2))
        except TypeError:
            # arg2 might be constant rather than a namedtuple
            return tuple(fn(a1, arg2) for a1 in arg1)

    @classmethod
    def depends_on(cls, for_init=False, for_sort=False, ignore_pseudo=False):
        """Return the variables this variable depends on.

        :param for_init: return only the variables used in initialization
        :param for_sort: return only the variables relevant for sorting vars
        :param ignore_pseudo: do not return names of pseudo-variables
        :return: list of all variable names this variable depends on
        """
        return [cls._position_varname]

    def set_initial_amount(self, treatment=None):
        """Set the step 0 amount for this velocity."""
        self._last_position = None
        self._amount = self._calculate_amount()


def velocity(variable_name, *args):
    """Create a new velocity."""
    if len(args) == 1:
        position = args[0]
        docstring = '' 
    elif len(args) == 2:
        docstring, position = args 
    elif len(args) == 0:
        raise MinnetonkaError(
            'Velocity {} names no position variable'.format(variable_name))
    else:
        raise MinnetonkaError('Too many arguments for velocity {}: {}'.format(
            variable_name, args))
    return _create_velocity(variable_name, docstring, position)

def _create_velocity(velocity_varname, docstring, position_varname):
    """Create a new velocity."""
    newvar = type(velocity_varname, (VelocityInstance,),
                {
                    '__doc__': docstring, 
                    '_position_varname': position_varname, 
                    '_last_position': None, 
                    '_validators': list(),
                    '_derived': {'derived': False},
                    '_has_history': True,
                    '_exclude_treatments': []
                    }
                )
    Model.add_variable_to_current_context(newvar)
    return newvar

#
# foreach: for iterating across a dict within a variable
#

def foreach(by_item_callable):
    """
    Return a new callable iterates across dicts or tuples.

    Variables often take simple values: ints or floats. But sometimes they
    take more complex values: dicts or tuples. Consider a business model
    of a group of five restaurants, all owned by the same company. Each 
    individual restaurant is managed differently, with its own opening
    and closing hours, its own table count, its own daily revenue, its own 
    mix of party sizes.
    But although the variables take different values for each restaurant, they
    participate in the same structure. Earnings is always revenue minus cost.
    (This restaurant example is borrowed from `a book on business modeling
    <https://www.amazon.com/Business-Modeling-Practical-Guide-Realizing/dp/0123741513>`_.)

    A good approach to model the restaurants is to have each variable take a 
    Python dict as a value, with the name of the restaurant as the key and a 
    numeric value for each key. For example the variable **DailyRevenue** might 
    take a value of ``{'Portia': 7489, 'Nola': 7136, 'Viola': 4248, 
    'Zona': 6412, 'Adelina': 4826}``, assuming the restaurants are named 
    Portia, Nola, etc.

    But how should the `specifier` of variable **DailyEarnings** be modeled if
    both **DailyRevenue** and **DailyCosts** are dicts? Although earnings is
    revenue minus cost, the specifier cannot be 
    ``lambda revenue, cost: revenue - cost`` because revenue and cost are both
    dicts, and subtraction is unsupported for dicts in Python.

    One approach is to write a custom specifier for **DailyEarnings**, a Python
    custom function that takes the two dicts, and returns a third dict,
    subtracting each restaurant's cost from its revenues. A better approach
    is to use foreach: ``foreach(lambda revenue, cost: revenue - cost)``. See
    example below.

    :func:`foreach` takes a single callable that operates on individual
    values (e.g. the revenue and cost of a single restaurant), and returns a
    callable that operates on dicts as a whole (e.g. the revenue of all the
    restaurants as a dict, the costs of all the restaurants as a dict). 

    The dict that is the amount of the second variable must contain the 
    same keys as the dict that is the amount of the first variable, or else the
    foreach-generated callable raises a :class:`MinnetonkaError`. The second
    dict can contain additional keys, not present in the first dict. Those
    additional keys are ignored. Similarly, the third dict (if present) must
    contain the same keys as the first dict. And so on.

    :func:`foreach` also works on tuples. For example, suppose instead of a 
    dict, 
    the revenue of the restaurants were represented as a tuple, with the first 
    element of the tuple being Portia's revenue, the second element Nola's 
    revenue and so on. The cost of the restaurants are also
    represented as a tuple. Then the specifier of **DailyEarnings** could
    be provided as ``foreach(lambda revenue, cost: revenue - cost)``, the
    same :func:`foreach` expression as with dicts.

    The tuple that is the amount of the second variable can be shorter (of
    lessor length) than the dict that is the amount of the first variable. The
    foreach-generated callable uses the shortest of the tuples as the length
    of the resultant tuple. 

    If the first variable has an amount that is a tuple, the second 
    variable can be a scalar, as can the third variable, etc. When encountering
    a scalar in subsequent amounts, the foreach-generated callable interprets 
    the scalar as if an iterator repeated the scalar, as often as 
    needed for the length of the first tuple. 

    :func:`foreach` works on Python named tuples as well. The result of the
    foreach-generated specifier is a named tuple of the same type as the first
    dependency.

    :func:`foreach` can be nested, if the amounts of the dependencies are
    nested dicts, or nested tuples, or nested named tuples, or some nested
    combination of dicts, tuples, or named tuples. See example below.

    :func:`foreach` can be used in defining variables, stocks, accums, or 
    constants, anywhere that a callable can be provided.

    Parameters
    ----------
    by_item_callable : callable
        A callable that is to be called on individual elements, either elements
        of a tuple or elements of a named tuple or values of a dict.

    Returns
    -------
    callable
        A new callable that can be called on dicts or tuples or named tuples,
        calling `by_item_callable` as on each element of the dict or tuple or
        named tuple.

    Examples
    --------
    Suppose there are five restaurants. Each of the restaurants has a weekly
    cost and a weekly revenue. (In practice the cost and revenue would 
    themselves be variables, not constants, and dependent on other variables, 
    like weekly customers, order mix, number of cooks, number of waitstaff, 
    etc.)

    >>> with model(treatments=['as is', 'to be']) as m:
    ...     Revenue = constant('Revenue',
    ...         {'Portia': 44929, 'Nola': 42798, 'Viola': 25490, 'Zona': 38477,
    ...          'Adelina': 28956})
    ...     Cost = constant('Cost',
    ...         {'Portia': 40440, 'Nola': 42031, 'Viola': 28819, 'Zona': 41103,
    ...          'Adelina': 25770})

    Earnings is revenue minus cost, for each restaurant.

    >>> with m:
    ...     Earnings = variable('Earnings', 
    ...         foreach(lambda revenue, cost: revenue - cost), 
    ...         'Revenue', 'Cost')
    >>> Earnings['']
    {'Portia': 4489, 
     'Nola': 767, 
     'Viola': -3329, 
     'Zona': -2626, 
     'Adelina': 3186}

    Stocks can also use :func:`foreach`. Suppose each restaurant has a stock
    of regular customers. Every week, some customers are delighted with the
    restaurant and become regulars. Every week some of the regulars attrit, 
    growing tired with the restaurant they once frequented, or move away
    to somewhere else, and are no longer able to enjoy the restaurant regularly.

    >>> with model(treatments=['as is', 'to be']) as m:
    ...     Regulars = stock('Regulars', 
    ...         foreach(lambda add, lost: add - lost), 
    ...         ('NewRegulars', 'LostRegulars'),
    ...         {'Portia': 420, 'Nola': 382, 'Viola': 0, 'Zona': 294, 
    ...          'Adelina': 23})
    ...     NewRegulars = constant('NewRegulars', 
    ...         {'Portia': 4, 'Nola': 1, 'Viola': 1, 'Zona': 2, 'Adelina': 4})
    ...     LostRegulars = variable('LostRegulars',
    ...         foreach(lambda regulars, lossage_prop: regulars * lossage_prop),
    ...         'Regulars', 'LossageProportion')
    ...     LossageProportion = constant('LossageProportion', 
    ...         PerTreatment({'as is': 0.01, 'to be': 0.0075}))
    
    >>> Regulars['as is']
    {'Portia': 420, 'Nola': 382, 'Viola': 0, 'Zona': 294, 'Adelina': 23}

    >>> m.step()
    >>> Regulars['as is']
    {'Portia': 419.8, 'Nola': 379.18, 'Viola': 1.0, 'Zona': 293.06, 
     'Adelina': 26.77}

    A variable can take an amount that is a dict of named tuples (or any
    other combination of dicts, named tuples, and tuples). Nested foreach
    calls can work on these nested variables.

    In this example, menu items change over time, as items are added or
    removed for each restaurant. Note that **Shape** serves only to give
    other variables the right shape: a dict of restaurants with values that
    are instances of **Course** named tuples.

    >>> import collections
    >>> import random
    >>> Course=collections.namedtuple(
    ...     'Course', ['appetizer', 'salad', 'entree', 'desert'])
    >>> with model() as m:
    ...     MenuItemCount = stock('MenuItemCount', 
    ...         foreach(foreach(lambda new, old: new - old)), 
    ...         ('AddedMenuItem', 'RemovedMenuItem'),
    ...         {'Portia': Course(6, 4, 12, 7), 
    ...          'Nola': Course(3, 3, 8, 4),
    ...          'Viola': Course(17, 8, 9, 12), 
    ...          'Zona': Course(10, 4, 20, 6),
    ...          'Adelina': Course(6, 9, 9, 3)})
    ...     AddedMenuItem = variable('AddedMenuItem', 
    ...         foreach(foreach(lambda s: 1 if random.random() < 0.1 else 0)),
    ...         'Shape')
    ...     RemovedMenuItem = variable('RemovedMenuItem',
    ...         foreach(foreach(lambda s: 1 if random.random() < 0.08 else 0)),
    ...         'Shape')
    ...     Shape = constant('Shape', 
    ...         lambda: {r: Course(0, 0, 0, 0) 
    ...             for r in ['Portia', 'Nola', 'Viola', 'Zona', 'Adelina']})

    >>> MenuItemCount['']
    {'Portia': Course(appetizer=6, salad=4, entree=12, desert=7),
     'Nola': Course(appetizer=3, salad=3, entree=8, desert=4),
     'Viola': Course(appetizer=17, salad=8, entree=9, desert=12),
     'Zona': Course(appetizer=10, salad=4, entree=20, desert=6),
     'Adelina': Course(appetizer=6, salad=9, entree=9, desert=3)}

    >>> m.step(10)
    >>> MenuItemCount['']
    {'Portia': Course(appetizer=7, salad=3, entree=12, desert=4),
     'Nola': Course(appetizer=4, salad=4, entree=8, desert=4),
     'Viola': Course(appetizer=18, salad=6, entree=8, desert=13),
     'Zona': Course(appetizer=12, salad=5, entree=18, desert=9),
     'Adelina': Course(appetizer=4, salad=8, entree=7, desert=3)}
    """
    return Foreach(by_item_callable)

class Foreach:
    """Implements the foreach, and also supports addition and multiplication."""
    def __init__(self, by_item_callable):
        self._by_item = by_item_callable

    def __call__(self, item1, *rest_items):
        return self._foreach_fn(item1)(item1, *rest_items)

    def _foreach_fn(self, item):
        """Return the appropriate foreach function for the argument."""
        if isinstance(item, dict):
            return self._across_dicts
        elif isnamedtuple(item):
            return self._across_namedtuples
        elif isinstance(item, tuple):
            return self._across_tuples
        else:
            raise MinnetonkaError(
                'First arg of foreach {} must be dictionary or tuple'.format(
                    item))

    def _across_dicts(self, dict1, *rest_dicts):
        """Execute by_item on every item across dict."""
        try:
            return {k: self._by_item(
                        dict1[k], 
                        *[self._maybe_element(r, k) for r in rest_dicts])
                    for k in dict1.keys()}
        except KeyError:
            raise MinnetonkaError('Foreach encountered mismatched dicts')

    def _maybe_element(self, maybe_dict, k):
        """Return maybe_dict[k], or just maybe_dict, if not a dict."""
        # It's kind of stupid that it tries maybe_dict[k] repeatedly
        try:
            return maybe_dict[k]
        except TypeError:
            return maybe_dict

    def _across_namedtuples(self, *nts):
        """Execute by_item_callable across namedtuples and scalars."""
        if self._is_all_same_type_or_nontuple(*nts):
            tuples = (self._repeat_if_necessary(elt) for elt in nts)
            typ = type(nts[0])
            return typ(*(self._by_item(*tupes) for tupes in zip(*tuples)))
        else:
            raise MinnetonkaError(
                'Foreach encountered mismatched named tuples: {}'.format(nts))

    def _across_tuples(self, *tuples):
        """Execute by_item_callable across tuples and scalars."""
        tuples = (self._repeat_if_necessary(elt) for elt in tuples)
        return tuple((self._by_item(*tupes) for tupes in zip(*tuples)))

    def _is_all_same_type_or_nontuple(self, first_thing, *rest_things):
        """Return whether everything is either the same type, or a scalar."""
        first_type = type(first_thing)
        return all(type(thing) == first_type or not isinstance(thing, tuple)
                   for thing in rest_things)

    def _repeat_if_necessary(self, elt):
        """Make an infinite iter from a scalar."""
        return elt if isinstance(elt, tuple) else itertools.repeat(elt)

    def add(self, augend, addend):
        """Add together across foreach."""
        try: 
            inner_add = self._by_item.add
            add = lambda a, b: inner_add(a, b)
        except AttributeError:
            add = lambda a, b: a + b 

        if isinstance(augend, dict):
            return {k: add(augend[k], addend[k]) for k in augend.keys()}
        elif isnamedtuple(augend):
            return type(augend)(*(add(a1, a2) for a1, a2 in zip(augend,addend)))
        elif isinstance(augend, tuple):
            return tuple(add(a1, a2) for a1, a2 in zip(augend, addend))
        else:
            raise MinnetonkaError(
                'Cannot add {} and {}'.format(augend, addend))

    def multiply(self, foreach_item, factor):
        """Multiply foreach_item by (simple) factor."""
        try:
            inner_mult = self._by_item.multiply 
            mult = lambda a, b: inner_mult(a, b)
        except AttributeError:
            mult = lambda a, b: a * b 

        if isinstance(foreach_item, dict):
            return {k: mult(v, factor) for k, v in foreach_item.items()}
        elif isnamedtuple(foreach_item):
            return type(foreach_item)(*(mult(v, factor) for v in foreach_item))
        elif isinstance(foreach_item, tuple):
            return tuple(mult(v, factor) for v in foreach_item)
        else:
            raise MinnetonkaError(
                'Cannot multiply {} by {}'.format(foreach_item, factor))


def isnamedtuple(x):
    """Returns whether x is a namedtuple."""
    # from https://bit.ly/2SkthFu
    t = type(x)
    b = t.__bases__
    if len(b) != 1 or b[0] != tuple: return False
    f = getattr(t, '_fields', None)
    if not isinstance(f, tuple): return False
    return all(type(n)==str for n in f)

#
# Validating new values
#

class _Validator:
    """For validating proposed new value for a common variable."""
    def __init__(self, test, error_code, error_message_gen, 
                 suggested_amount=None):
        self._test = test
        self._error_code = error_code
        self._error_message_gen = error_message_gen
        self._suggested_amount = suggested_amount

    def validate(self, amount, name):
        """Is this amount valid?"""
        if self._test(amount):
            return True, None, None, None
        else: 
            return (
                False, self._error_code, self._error_message_gen(amount, name),
                self._suggested_amount)


def constraint(var_names, test, error_code, error_message_gen):
    """Define a new constraint among variables with var_names."""
    new_constraint = _Constraint(var_names, test, error_code, error_message_gen)
    Model.add_constraint_to_current_context(new_constraint)
    return new_constraint


class _Constraint:
    """A constraint among multiple variables, tested by Model.validate_all()."""
    def __init__(self, var_names, test, error_code, error_message_gen):
        self._var_names = var_names
        self._test = test
        self._error_code = error_code
        self._error_message_gen = error_message_gen

    def fails(self, model):
        """Validate constraint against all treatments. Return error or None."""
        def _fail_dict(msg, treatment):
            return {
                        'error_code': self._error_code,
                        'inconsistent_variables': self._var_names,
                        'error_message': msg,
                        'treatment': treatment.name
                    }

        for treatment in model.treatments():
            if self._is_defined_for_all(model, treatment):
                amounts = [model[v][treatment.name] for v in self._var_names]
                try:
                    test_result = self._test(*amounts)
                except Exception as err:
                    return _fail_dict(
                        f'Constraint raised exception {str(err)}', treatment)
                if not test_result:
                    try:
                        err_message = self._error_message_gen(
                            self._var_names, amounts, treatment.name)
                    except Exception as err:
                        return _fail_dict(
                            f'Constraint raised exception {str(err)}', 
                            treatment)
                    return _fail_dict(err_message, treatment)
        return None

    def _is_defined_for_all(self, model, treatment):
        """Are all variables defined for the treatment?"""
        return all(not model[v].is_undefined_for(treatment.name)
                   for v in self._var_names)

#
# Constructing results to send across network
#

class _Result:
    def __init__(self, excerpt=None, **kwargs):
        self._result_in_progress = kwargs
        if excerpt:
            self._result_in_progress['excerpt'] = excerpt

    def add(self, **kwargs):
        self._result_in_progress = {**self._result_in_progress, **kwargs}

    def fail(self, error_code, error_message, **kwargs):
        self.add(
            success=False, error_code=error_code, error_message=error_message,
            **kwargs)
        return self._result_in_progress 

    def succeed(self):
        self.add(success=True) 
        return self._result_in_progress

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

    def __truediv__(self, other):
        """Divide by other."""
        if isinstance(other, tuple):
            try:
                return type(self)(*(x / y for x, y in zip(self, other)))
            except:
                return NotImplemented
        else:
            try:
                return type(self)(*(x / other for x in self))
            except:
                return NotImplemented

    def __round__(self, ndigits=0):
        """Round the named tuple."""
        return type(self)(*(round(x, ndigits) for x in self))

    def __le__(self, other):
        """Implement <="""
        if type(self) == type(other):
            return all(s <= o for s, o in zip(self, other))
        elif isinstance(other, int) or isinstance(other, float):
            return all(s <= other for s in self)
        else:
            return NotImplemented

    def __ge__(self, other):
        """Implement >="""
        if type(self) == type(other):
            return all(s >= o for s, o in zip(self, other))
        elif isinstance(other, int) or isinstance(other, float):
            return all(s >= other for s in self)
        else:
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

def mean(number_list):
    return safe_div(sum(number_list), len(number_list))

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

