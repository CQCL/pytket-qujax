Changelog
~~~~~~~~~

0.15.0 (November 2023)
----------------------

* Updated pytket version requirement to 1.22.

0.14.0 (October 2023)
---------------------

* Updated pytket version requirement to 1.21.

0.13.0 (August 2023)
--------------------

* Update pytket version requirement to 1.18.
* Update qujax version requirement to 1.0.

0.12.0 (June 2023)
------------------

* Update pytket version requirement to 1.16.
* Update qujax version requirement to 0.3.4.
* Drop support for Python 3.8; add support for 3.11.

0.11.0 (December 2022)
----------------------

* Update qujax version requirement to 0.3.1
* Add support for simulator=unitary in tk_to_qujax
* Updated pytket version requirement to 1.10.

0.10.0 (November 2022)
----------------------

* Updated pytket version requirement to 1.9.

0.9.0 (November 2022)
---------------------

* Update qujax version requirement to 0.3.0
* Add simulator argument to `tk_to_qujax` to enable
  conversion to densitytensor simulator

0.8.0 (November 2022)
---------------------

* Updated pytket version requirement to 1.8.

0.7.1 (October 2022)
--------------------

* Added `tk_to_param` function

0.7.0 (October 2022)
--------------------

* Updated pytket version requirement to 1.7.

0.6.1 (September 2022)
----------------------

* Added support for a blend of constants and symbolic
  arguments (or more complex operations) within a single gate.

0.6.0 (September 2022)
----------------------

* Added support for a blend of numerical and symbolic
  parameterised gates.
* Added automatic conversion of non-parameterised gates
  not found in qujax.gates (in symbolic implementation only)
* Raises error if measurements found
* Updated pytket version requirement to 1.6.

0.5.0 (September 2022)
----------------------

* Consolidate `tk_to_qujax_args` with `tk_to_qujax_args_symbolic`,
  `tk_to_qujax` with `tk_to_qujax_symbolic`,
  `print_circuit` with `print_circuit_symbolic`
  and removed all the symbolic versions.
  Now, all the functions have a default symbol_map argument
  which tells wherther or not to make it symbolic.
* Renamed `qujax_to_tk` to `qujax_args_to_tk`

0.4.0 (August 2022)
-------------------

* Add tk_to_qujax_args, tk_to_qujax_args_symbolic
* Add print_circuit, print_circuit_symbolic
* Updated qujax version requirement to 0.2.5.

0.3.0 (August 2022)
-------------------

* Updated qujax version requirement to 0.2.4.

0.2.0 (August 2022)
-------------------

* Updated qujax version requirement to 0.1.5.
* Updated pytket version requirement to 1.5.

0.1.1 (July 2022)
-----------------

* minor fix

0.1.0 (July 2022)
-----------------

* add tk_to_qujax and tk_to_qujax_symbolic
* update to qujax version 0.1.3
