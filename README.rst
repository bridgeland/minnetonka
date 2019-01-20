Minnetonka is a Python package for 
`business modeling and simulation 
<https://www.amazon.com/Business-Modeling-Practical-Guide-Realizing/dp/0123741513>`__.


Motivation
==========

Over the last 25 years, I built many business simulation models, initially in
`iThink <https://www.iseesystems.com/>`__, then in 
`Powersim <http://www.powersim.com/>`__, and since 2006 in 
`Forio SimLang <https://forio.com/epicenter/docs/public/model_code/forio_simlang/language_overview/>`__. 
`Forio SimLang is great <https://hangingsteel.com/2013/03/11/forio-simulate/>`__
for business modeling, powerful and expressive, with great support for arrays. 
But lately I have encountered several occasions in 
which SimLang is a good solution for modeling part of the problem, and an 
existing Python package is a good solution for modeling another part of the 
problem. 

For example, in 2016 I built a model in SimLang that included some simple
`graph logic <https://en.wikipedia.org/wiki/Graph_(discrete_mathematics)>`__. 
There are no graph primitives in SimLang, so I created the graphs using the
primitives available: arrays
and enums and floats. There are no built-in analytics to find the shortest
path in a weighted graph, so I wrote one, using the `Floyd-Warshall algorithm 
<https://en.wikipedia.org/wiki/Floyd%E2%80%93Warshall_algorithm>`__.

That model would have been far simpler had I used the Python package 
`NetworkX <http://networkx.github.io/>`__ for the graph logic, and used SimLang
for the other 80% of the model. What I needed was the expressivity of SimLang, 
within a Python package, so I could easily integrate with NetworkX.

Minnetonka is that combination: the power of Forio SimLang for business
modeling, delivered as a Python package. Minnetonka is an appropriate solution
when you want to model a business (or other social organization) in Python.

Features
========

A Minnetonka model is a collection of variables. Each variable takes a value,
a value that can be of any Python data type: an integer, a float, a tuple, an
array, a dict, etc. 

A Minnetonka model is simulated over time. The variables in a model take
a succession of values during the simulation.

Minnetonka variables are defined in terms of other Minnetonka variables via
Python functions, allowing arbitary Python code
to be executed at every simulation time step, for every variable.

Minnetonka supports 
`stocks and flows <https://en.wikipedia.org/wiki/Stock_and_flow>`__. Stocks
allow circular dependencies among variables, to model the circular causality
underlying many business situations. 

Minnetonka introduces **treatments**, a primitive for value modeling. A single
variable can take one value in one treatment and take a different value in 
a different treatment. For example, business earnings might be $20 million 
per year in the as-is treatment, and $25 million per year in a to-be treatment, 
with a planned investment generating additional earnings.

Getting Started
===============

- `Building a model in Minnetonka <building_model.html>`__
- `API reference <https://bridgeland.github.io/minnetonka/>`__

Installation
============

Dependendencies
===============

Minnetonka requires Python 3.6, and depends on NumPy and SciPy. 

Contact
========

`Send me an email <dave@hangingsteel.com>`__.

License
=======

`Apache License, Version 2. <https://www.apache.org/licenses/LICENSE-2.0>`__

Naming
======

Minnetonka is named after 
`Lake Minnetonka <https://en.wikipedia.org/wiki/Lake_Minnetonka>`__, the 
ninth largest lake in the US state of Minnesota. 


