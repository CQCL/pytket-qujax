# Copyright 2019-2022 Cambridge Quantum Computing
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Methods to allow conversion between qujax and pytket
"""

from typing import Tuple, Sequence, Optional, List, Union, Callable, Any
from functools import wraps

import qujax  # type: ignore
from jax import numpy as jnp
from sympy import lambdify, Symbol  # type: ignore
from pytket import Qubit, Circuit  # type: ignore
from pytket._tket.circuit import Command  # type: ignore


def _tk_qubits_to_inds(tk_qubits: Sequence[Qubit]) -> Tuple[int, ...]:
    """
    Convert Sequence of pytket qubits objects to tuple of integers qubit indices.

    :param tk_qubits: Sequence of pytket qubit object
        (as stored in ``pytket.Circuit.qubits``).
    :type tk_qubits: Sequence[Qubit]
    :return: Tuple of qubit indices.
    :rtype: Tuple[int]
    """
    return tuple(q.index[0] for q in tk_qubits)


def _append_func(f: Callable, func_to_append: Callable) -> Callable:
    """
    Decorates function f by applying func_to_append on the output of f.
    Retains signature of f.
    I.e. returns a function that executes func_to_append(f(*args **kwargs))
    with the same signature as f.

    :param f: Base function, whose signature is retained.
    :type f: Callable
    :param func_to_append: Function to apply to the output of
    :type func_to_append: Callable
    :return: Decorated function that executes func_to_append(f(*args **kwargs))
    :rtype: Callable
    """

    @wraps(f)
    def g(*args: Any, **kwargs: Any) -> Any:
        return func_to_append(*f(*args, **kwargs))

    return g


def _symbolic_command_to_gate_and_param_inds(
    command: Command, symbol_map: dict
) -> Tuple[Union[str, Callable[[jnp.ndarray], jnp.ndarray]], Sequence[int]]:
    """
    Convert pytket command to qujax (gate, parameter indices) tuple.

    :param command: pytket circuit command from .get_commands()
    :type command: Command
    :param symbol_map: ``dict``, maps symbolic pytket parameters following the order
        in this dict.
    :type symbol_map: dict
    :return: tuple of gate and parameter indices
        gate will given as either a string in qujax.gates (if command is not symbolic
        has single symbolic argument with no operations) or a function that maps
        parameters to a jax unitary matrix.
    :rtype: Tuple[Union[str, Callable[[jnp.ndarray], jnp.ndarray]], Sequence[int]]
    """
    gate_str = command.op.type.name
    free_symbols = list(command.op.free_symbols())

    if gate_str in qujax.gates.__dict__:
        qujax_gate = getattr(qujax.gates, gate_str)
        if callable(qujax_gate):
            gate_params = command.op.params
            if len(free_symbols) == 0:
                gate = qujax_gate(*gate_params)
            elif len(free_symbols) == 1 and isinstance(free_symbols[0], Symbol):
                gate = gate_str
            else:
                gate_lambda = lambdify(free_symbols, gate_params)
                gate = _append_func(gate_lambda, qujax_gate)
        else:
            gate = gate_str
    else:
        if len(free_symbols) == 0:
            gate = jnp.array(command.op.get_unitary())  # type: ignore
        else:
            raise TypeError("Parameterised gate not found in qujax.gates")

    param_inds = tuple(symbol_map[symbol] for symbol in free_symbols)  # type: ignore
    return gate, param_inds


def tk_to_qujax_args(
    circuit: Circuit, symbol_map: Optional[dict] = None
) -> Tuple[
    Sequence[Union[str, Callable[[jnp.ndarray], jnp.ndarray]]],
    Sequence[Sequence[int]],
    Sequence[Sequence[int]],
    int,
]:
    """
    Converts a pytket circuit into a tuple of arguments representing
    a qujax quantum circuit.
    Assumes all circuit gates can be found in ``qujax.gates``
    The ``symbol_map`` argument controls the interpretation of any symbolic parameters
    found in ``circuit.free_symbols()``.

    - If ``symbol_map`` is ``None``, circuit.free_symbols() will be ignored.
      Parameterised gates will be determined based on whether they are stored as
      functions (parameterised) or arrays (non-parameterised) in qujax.gates. The order
      of qujax circuit parameters is the same as in circuit.get_commands().
    - If ``symbol_map`` is provided as a ``dict``, assign qujax circuit parameters to
      symbolic parameters in ``circuit.free_symbols()``; the order of qujax circuit
      parameters will be given by this dict. Keys of the dict should be symbolic pytket
      parameters as in ``circuit.free_symbols()`` and the values indicate
      the index of the qujax circuit parameter -- integer indices starting from 0.

    The conversion can also be checked with ``print_circuit``.

    :param circuit: Circuit to be converted (without any measurement commands).
    :type circuit: pytket.Circuit
    :param symbol_map:
        If ``None``, parameterised gates determined by ``qujax.gates``. \n
        If ``dict``, maps symbolic pytket parameters following the order in this dict.
    :type symbol_map: Optional[dict]
    :return: Tuple of arguments defining a (parameterised) quantum circuit
        that can be sent to ``qujax.get_params_to_statetensor_func``. The elements of
        the tuple (qujax args) are as follows

        - Sequence of gate name strings to be found in ``qujax.gates``.
        - Sequence of sequences describing which qubits gates act on.
        - Sequence of sequences of parameter indices that gates are using.
        - Number of qubits.

    :rtype: Tuple[Sequence[str], Sequence[Sequence[int]], Sequence[Sequence[int]], int]
    """
    if symbol_map:
        assert (
            set(symbol_map.keys()) == circuit.free_symbols()
        ), "Circuit keys do not much symbol_map"
        assert set(symbol_map.values()) == set(
            range(len(circuit.free_symbols()))
        ), "Incorrect indices in symbol_map"

    gate_name_seq = []
    qubit_inds_seq = []
    param_inds_seq = []
    param_index = 0
    for c in circuit.get_commands():
        gate_name = c.op.type.name
        if gate_name == "Barrier":
            continue
        if gate_name == "Measure":
            raise TypeError(
                "Measurements not supported in qujax. \n"
                "qujax produces statetensor corresponding "
                "to all qubits."
            )

        if symbol_map:
            gate, param_inds = _symbolic_command_to_gate_and_param_inds(c, symbol_map)
        else:
            if gate_name not in qujax.gates.__dict__:
                raise TypeError(
                    "Gate not found in qujax.gates. \n pytket-qujax can automatically "
                    "convert arbitrary non-parameterised gates when specified in a "
                    "symbolic circuit and absent from the symbol_map argument. \n"
                    "Arbitrary parameterised gates can be added to a local "
                    "qujax.gates installation and/or submitted via pull request."
                )
            gate = gate_name
            n_params = len(c.op.params)
            param_inds = tuple(range(param_index, param_index + n_params))
            param_index += n_params

        gate_name_seq.append(gate)
        qubit_inds_seq.append(_tk_qubits_to_inds(c.qubits))
        param_inds_seq.append(param_inds)

    return gate_name_seq, qubit_inds_seq, param_inds_seq, circuit.n_qubits


def tk_to_qujax(
    circuit: Circuit, symbol_map: Optional[dict] = None
) -> qujax.UnionCallableOptionalArray:
    """
    Converts a pytket circuit into a function that maps circuit parameters
    to a statetensor. Assumes all circuit gates can be found in ``qujax.gates``
    The ``symbol_map`` argument controls the interpretation of any symbolic parameters
    found in ``circuit.free_symbols()``.

    - If ``symbol_map`` is ``None``, circuit.free_symbols() will be ignored.
      Parameterised gates will be determined based on whether they are stored as
      functions (parameterised) or arrays (non-parameterised) in qujax.gates. The order
      of qujax circuit parameters is the same as in circuit.get_commands().
    - If ``symbol_map`` is provided as a ``dict``, assign qujax circuit parameters to
      symbolic parameters in ``circuit.free_symbols()``; the order of qujax circuit
      parameters will be given by this dict. Keys of the dict should be symbolic pytket
      parameters as in ``circuit.free_symbols()`` and the values indicate
      the index of the qujax circuit parameter -- integer indices starting from 0.

    The conversion can be checked by examining the output from ``tk_to_qujax_args``
    or ``print_circuit``.

    :param circuit: Circuit to be converted (without any measurement commands).
    :type circuit: pytket.Circuit
    :param symbol_map:
        If ``None``, parameterised gates determined by ``qujax.gates``. \n
        If ``dict``, maps symbolic pytket parameters following the order in this dict.
    :type symbol_map: Optional[dict]
    :return: Function which maps parameters (and optional statetensor_in)
        to a statetensor.
        If the circuit has no parameters, the resulting function
        will only take the optional ``statetensor_in`` as an argument.
    :rtype: Callable[[jnp.ndarray, jnp.ndarray = None], jnp.ndarray]
        or Callable[[jnp.ndarray = None], jnp.ndarray]
        if no parameters found in circuit
    """

    return qujax.get_params_to_statetensor_func(*tk_to_qujax_args(circuit, symbol_map))


def print_circuit(
    circuit: Circuit,
    symbol_map: Optional[dict] = None,
    qubit_min: int = 0,
    qubit_max: int = jnp.inf,  # type: ignore
    gate_ind_min: int = 0,
    gate_ind_max: int = jnp.inf,  # type: ignore
    sep_length: int = 1,
) -> List[str]:
    """
    Returns and prints basic string representation of circuit.

    For more information on the ``symbol_map`` parameter refer to the
    ``tk_to_qujax`` or ``tk_to_qujax_args`` documentation.

    :param circuit: Circuit to be converted (without any measurement commands).
    :type circuit: pytket.Circuit
    :param symbol_map:
        If ``None``, parameterised gates determined by ``qujax.gates``. \n
        If ``dict``, maps symbolic pytket parameters following the order in this dict.
    :type symbol_map: Optional[dict]
    :param qubit_min: Index of first qubit to display.
    :type qubit_min: int
    :param qubit_max: Index of final qubit to display.
    :type qubit_max: int
    :param gate_ind_min: Index of gate to start circuit printing.
    :type gate_ind_min: int
    :param gate_ind_max: Index of gate to stop circuit printing.
    :type gate_ind_max: int
    :param sep_length: Number of dashes to separate gates.
    :type sep_length: int
    :return: String representation of circuit
    :rtype: List[str]
    """
    g, q, p, nq = tk_to_qujax_args(circuit, symbol_map)
    return qujax.print_circuit(  # type: ignore
        g, q, p, nq, qubit_min, qubit_max, gate_ind_min, gate_ind_max, sep_length
    )


def qujax_args_to_tk(
    gate_seq: Sequence[str],
    qubit_inds_seq: Sequence[Sequence[int]],
    param_inds_seq: Sequence[Sequence[int]],
    n_qubits: Optional[int] = None,
    param: Optional[jnp.ndarray] = None,
) -> Circuit:
    """
    Convert qujax args into pytket Circuit.

    :param gate_seq: Sequence of gates. Each element is a string matching an array
        or function in qujax.gates
    :type gate_seq: Sequence[str]
    :param qubit_inds_seq: Sequences of qubits (ints) that gates are acting on.
    :type qubit_inds_seq: Sequence[Sequence[int]]
    :param param_inds_seq: Sequence of parameter indices that gates are using,
        i.e. [[0], [], [5, 2]] tells qujax that the first gate uses the first parameter,
        the second gate is not parameterised and the third gates used the fifth and
        second parameters.
    :type param_inds_seq: Sequence[Sequence[int]]
    :param n_qubits: Number of qubits, if fixed.
    :type n_qubits: int
    :param param: Circuit parameters, if parameterised. Defaults to all zeroes.
    :type param: jnp.ndarray
    :return: Circuit
    :rtype: pytket.Circuit
    """
    if any(not isinstance(gate_name, str) for gate_name in gate_seq):
        raise TypeError(
            "qujax_args_to_tk currently only supports gates as strings"
            "(found in qujax.gates)"
        )

    qujax.check_circuit(gate_seq, qubit_inds_seq, param_inds_seq, n_qubits)

    if n_qubits is None:
        n_qubits = max([max(qi) for qi in qubit_inds_seq]) + 1

    if param is None:
        n_params = max([max(p) + 1 if len(p) > 0 else 0 for p in param_inds_seq])
        param = jnp.zeros(n_params)  # type: ignore

    param_inds_seq = [jnp.array(p, dtype="int32") for p in param_inds_seq]  # type: ignore

    c = Circuit(n_qubits)

    for g, q, p in zip(gate_seq, qubit_inds_seq, param_inds_seq):
        g_apply_func = c.__getattribute__(g)
        g_apply_func(*param[p], *q)

    return c
