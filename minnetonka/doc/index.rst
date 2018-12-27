
Welcome to Minnetonka's documentation!
======================================

.. automodule:: minnetonka
	:members: model, variable, constant, stock, previous, accum, foreach

.. autoclass:: Model
	:members: step, reset, recalculate, variable 

.. autoclass:: Variable
	:members: show, history, all
	:special-members: __getitem__, __setitem__

.. autoclass:: Constant
	:members: show, history, all
	:special-members: __getitem__, __setitem__

.. autoclass:: Previous
	:members: show, history, all
	:special-members: __getitem__

.. autoclass:: Stock
	:members: show, history, all
	:special-members: __getitem__, __setitem__

.. autoclass:: Accum
	:members: show, history, all
	:special-members: __getitem__, __setitem__

.. autoclass:: PerTreatment

.. toctree::
   :maxdepth: 2
   :caption: Contents:



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
