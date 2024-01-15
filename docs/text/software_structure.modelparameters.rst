Model Parameters
~~~~~~~~~~~~~~~~

The model parameters are the masses, lengths, ... of the double pendulum.
In this repository model parameters are stored in yml files. The keys of the
yml files are

- m1, m2, l1, l2, r1, r2, g, b1, b2, cf1, cf2, I1, I2, Ir, gr, tl1, tl2

While most classes and functions allow to directly set the model parameters in
the initialization/function calls most of the time it is more convenient to use
the model_parameters class. An object of that class stores all model
parameters, reads/writes yml files in the correct format and can be parsed to
most functions/classes in this library.

Some parameter sets are used so frequently (e.g. those that have been
identified for the real hardware) that they have been given names.
More precisely the name consists of a ``design`` and a ``model_id``.
The design refers to a hardware design where specific materials have be used.
The model_id refers to a set of model parameters which have been identified (or
estimated) for that design.
The naming convention for the design follows

- design_A.0, design_B.0, design_C.0, design_hD.0, ...

i.e. ``design_`` followed by a capital letter and a number. The number can
potentially used for minor changes in the design (such as different motor
units) in the future.  The h in front of the capital letter indicates that the
design is only hypothetical at the moment, i.e. it has not been realized with
real hardware.

The model_id format is

- model_1.0, model_1.1, model_2.0, model_3.0, ...

i.e. ``model_`` folowed by two integers. The first integer is the id which
identifies the model parameter set. The second integer can be used to simplify
the model.

- 0: full model
- 1: damping friction, coulomb friction and motor inertia are set to 0
- 2: motor inertia and coulomb friction set to 0
- 5: motor inertia set to 0

more simplified models may be used in the future.

