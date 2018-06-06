Minnetonka is a Python package for 
`business modeling and simulation 
<https://www.amazon.com/Business-Modeling-Practical-Guide-Realizing/dp/0123741513>`__,
based on `system dynamics <http://web.mit.edu/jsterman/www/BusDyn2.html>`__,
and borne of frustration.

Frustration
===========

Over the last 25 years, I built many business
simulation models in traditional system dynamics platforms, initially in
`iThink <https://www.iseesystems.com/>`__, then in 
`Powersim <http://www.powersim.com/>`__, and since 2006 in 
`Forio SimLang <https://forio.com/epicenter/docs/public/model_code/forio_simlang/language_overview/>`__. 
While building these models, I encountered a recurring frustration. The 
system dynamics platforms provide good primitives for most of the
requirements, covering 80% of what is needed. But the 
remaining 20% of the requirements are difficult to accomplish within the 
confines of iThink or Powersim or Forio SimLang.

For example, two ago I built a model in SimLang that included some simple
`graph logic <https://en.wikipedia.org/wiki/Graph_(discrete_mathematics)>`__. 
There are no graph primitives in traditional system dynamics 
languages, so I created the graphs using the primitives available: arrays
and enums and floats. There are no built-in analytics to find the shortest
path in a weighted graph, so I wrote one, using the `Floyd-Warshall algorithm 
<https://en.wikipedia.org/wiki/Floyd%E2%80%93Warshall_algorithm>`__.
Unfortunately my implementation of Floyd-Warshall in Forio SimLang was 
slow—even on small graphs—and it could not have been made any faster, given
the constraints of the language in which it was built. 

Other people have written graph packages in traditional programming languages.
For example, `NetworkX <http://networkx.github.io/>`__ is a Python package
for graphs. NetworkX includes a fast implementation of Floyd-Warshall, 
as well as 33 other algorithms for finding shortest paths in a graph. 
NetworkX is open source; anyone can use it for free. Instead of implementing
my own, why not just use NetworkX?

The traditional system dynamics platforms are closed systems. It is
difficult to integrate an outside package like NetworkX with a model built in 
SimLang, or Powersim, or iThink. These system dynamics platforms 
were initially built many years ago, when closed software platforms were the 
norm. Now most leading-edge software developers work in open ecosystems, like the 
Python ecosystem, or the R ecosystem, or the Javascript ecosystem. In these 
ecosystems, there are many existing packages available, packages that already
solve particular problems. 

For example, 
as of the summer of 2018, there are 140,000 open source Python projects, 140,000
solutions to particular problems that can be integrated into a solution. 
Developers in Python solve problems by
using a variety of existing packages. Much of the strength of Python lies in the 
ease of integrating existing packages into a coherent whole.

I could have developed the model in Python and used NetworkX. But graph 
logic was only part of the problem. For the rest of the model, I needed standard
system dynamics primitives: stocks and flows and time steps. In short I needed
a Python package for system dynamics, so I could build a model that included
both system dynamics and graphs. 

Features
========

**Minnetonka** is a system dynamics package in Python. Like the traditional
system dyanmics platforms, Minnetonka provides primitives for stocks and flows, 
and for creating
a simulation model of many variables, with circular dependencies among the
variables. But Minnetonka is built on top of Python. 

Variables in Minnetonka take values that are Python data types: integers, 
floats, tuples, arrays, dicts, etc. Minntetonka variables are defined in terms of
other Minnetonka variables via Python functions, allowing arbitary Python code
to be executed at every simulation time step, for every variable.

Minnetonka introduces **treatments**, a primitive for value modeling, supporting
comparison of an existing as-is situation with various to-be interventions.

Getting Started
===============

- `Building a model in Minnetonka <building_model.html>`__
- `API reference <api_reference.html>`__

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

Apache License, Version 2. (Link to license here.)

Naming
======

Minnetonks is named after 
`Lake Minnetonka <https://en.wikipedia.org/wiki/Lake_Minnetonkalake>`__, the 
ninth largest in the US state of Minnesota. 


