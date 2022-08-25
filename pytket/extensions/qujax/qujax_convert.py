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

from typing import Tuple, Sequence, Optional, List

import qujax  # type: ignore
from jax import numpy as jnp
from pytket import Qubit, Circuit  # type: ignore


def _tk_qubits_to_inds(tk_qubits: Sequence[Qubit]) -> Tuple[int, ...]:
    """
    Convert Sequence of pytket qubits objects to tuple of integers qubit indices.

    :param tk_qubits: Sequence of pytket qubit object
        (as stored in pytket.Circuit.qubits).
    :type tk_qubits: Sequence[Qubit]
    :return: Tuple of qubit indices.
    :rtype: tuple
    """
    return tuple(q.index[0] for q in tk_qubits)


def tk_to_qujax_args(
    circuit: Circuit,
) -> Tuple[Sequence[str], Sequence[Sequence[int]], Sequence[Sequence[int]], int]:
    """
    Converts a pytket circuit into a tuple of arguments representing
    a qujax quantum circuit.
    Assumes all circuit gates can be found in qujax.gates.
    Parameter argument of created function will be ordered as in circuit.get_commands()
    - pytket automatically reorders some gates, consider using Barriers.

    :param circuit: Circuit to be converted.
    :type circuit: pytket.Circuit.
    :return: Tuple of arguments defining a (parameterised) quantum circuit
        that can be sent to qujax.get_params_to_statetensor_func.
    :rtype: tuple
        gate_name_seq: Sequence of gate name strings to be found in qujax.gates.
        qubit_inds_seq: Sequence of sequences describing which qubits gates act on.
        param_inds_seq: Sequence of sequences parameter indices that gates are using.
        n_qubits: Number of qubits.
    """
    gate_name_seq = []
    qubit_inds_seq = []
    param_inds_seq = []
    param_index = 0
    for c in circuit.get_commands():
        gate_name = c.op.type.name
        if gate_name == "Barrier":
            continue
        gate_name_seq.append(gate_name)
        qubit_inds_seq.append(_tk_qubits_to_inds(c.qubits))
        n_params = len(c.op.params)
        param_inds_seq.append(jnp.arange(param_index, param_index + n_params))
        param_index += n_params

    return gate_name_seq, qubit_inds_seq, param_inds_seq, circuit.n_qubits


def tk_to_qujax(circuit: Circuit) -> qujax.UnionCallableOptionalArray:
    """
    Converts a pytket circuit into a function that maps circuit parameters
    to a statetensor. Assumes all circuit gates can be found in qujax.gates.
    Parameter argument of created function will be ordered as in circuit.get_commands()
    - pytket automatically reorders some gates, consider using Barriers.

    :param circuit: Circuit to be converted.
    :type circuit: pytket.Circuit
    :return: Function which maps parameters (and optional statetensor_in)
        to a statetensor.
        If the circuit has no parameters, the resulting function
        will only take the optional statetensor_in as an argument.
    :rtype: Callable[[jnp.ndarray, jnp.ndarray = None], jnp.ndarray]
        or Callable[[jnp.ndarray = None], jnp.ndarray]
        if no parameters found in circuit
    """

    return qujax.get_params_to_statetensor_func(*tk_to_qujax_args(circuit))


def tk_to_qujax_args_symbolic(
    circuit: Circuit, symbol_map: Optional[dict] = None
) -> Tuple[Sequence[str], Sequence[Sequence[int]], Sequence[Sequence[int]], int]:
    """
    Converts a pytket circuit with symbolics parameters and a symbolic parameter map
    into a tuple of arguments representing a qujax quantum circuit.
    Assumes all circuit gates can be found in qujax.gates.
    Note that the behaviour of tk_to_qujax_args_symbolic(circuit)
    is different to tk_to_qujax_args(circuit),
    tk_to_qujax_args_symbolic will look for parameters in circuit.free_symbols()
    and if there are none it will assume that none of the gates require parameters.
    On the other hand, tk_to_qujax_args will work out which gates are parameterised
    based on e.g. circuit.get_commands()[0].op.params

    :param circuit: Circuit to be converted.
    :type circuit: pytket.Circuit
    :param symbol_map: dict that maps elements of circuit.free_symbols() (sympy)
        to parameter indices.
    :type symbol_map: Optional[dict]
    :return: Tuple of arguments defining a (parameterised) quantum circuit
        that can be sent to qujax.get_params_to_statetensor_func.
    :rtype: tuple
        gate_name_seq: Sequence of gate name strings to be found in qujax.gates.
        qubit_inds_seq: Sequence of sequences describing which qubits gates act on.
        param_inds_seq: Sequence of sequences parameter indices that gates are using.
        n_qubits: Number of qubits.
    """
    if symbol_map is None:
        free_symbols = circuit.free_symbols()
        n_symbols = len(free_symbols)
        symbol_map = dict(zip(free_symbols, range(n_symbols)))
    else:
        assert (
            set(symbol_map.keys()) == circuit.free_symbols()
        ), "Circuit keys do not much symbol_map"
        assert set(symbol_map.values()) == set(
            range(len(circuit.free_symbols()))
        ), "Incorrect indices in symbol_map"

    gate_name_seq = []
    qubit_inds_seq = []
    param_inds_seq = []
    for c in circuit.get_commands():
        gate_name = c.op.type.name
        if gate_name == "Barrier":
            continue
        gate_name_seq.append(gate_name)
        qubit_inds_seq.append(_tk_qubits_to_inds(c.qubits))
        param_inds_seq.append(
            jnp.array([symbol_map[symbol] for symbol in c.op.free_symbols()])  # type: ignore
        )

    return gate_name_seq, qubit_inds_seq, param_inds_seq, circuit.n_qubits


def tk_to_qujax_symbolic(
    circuit: Circuit, symbol_map: Optional[dict] = None
) -> qujax.UnionCallableOptionalArray:
    """
    Converts a pytket circuit with symbolics parameters and a symbolic parameter map
    into a function that maps circuit parameters to a statetensor.
    Assumes all circuit gates can be found in qujax.gates.
    Note that the behaviour of tk_to_qujax_symbolic(circuit)
    is different to tk_to_qujax(circuit),
    tk_to_qujax_symbolic will look for parameters in circuit.free_symbols()
    and if there are none it will assume that none of the gates require parameters.
    On the other hand, tk_to_qujax will work out which gates are parameterised
    based on e.g. circuit.get_commands()[0].op.params

    :param circuit: Circuit to be converted.
    :type circuit: pytket.Circuit
    :param symbol_map: dict that maps elements of circuit.free_symbols() (sympy)
        to parameter indices.
    :type symbol_map: Optional[dict]
    :return: Function which maps parameters
        (and optional statetensor_in) to a statetensor.
        If the circuit has no parameters, the resulting function
        will only take the optional statetensor_in as an argument.
    :rtype: Callable[[jnp.ndarray, jnp.ndarray = None], jnp.ndarray]
        or Callable[[jnp.ndarray = None], jnp.ndarray]
        if no parameters found in circuit
    """
    return qujax.get_params_to_statetensor_func(
        *tk_to_qujax_args_symbolic(circuit, symbol_map)
    )


def print_circuit(
    circuit: Circuit,
    qubit_min: int = 0,
    qubit_max: int = jnp.inf,  # type: ignore
    gate_ind_min: int = 0,
    gate_ind_max: int = jnp.inf,  # type: ignore
    sep_length: int = 1,
) -> List[str]:
    """
    Returns and prints basic string representation of circuit.

    :param circuit: Circuit to be converted.
    :type circuit: pytket.Circuit
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
    :rtype: List[str] each element represents a qubit
    """
    g, q, p, nq = tk_to_qujax_args(circuit)
    return qujax.print_circuit(  # type: ignore
        g, q, p, nq, qubit_min, qubit_max, gate_ind_min, gate_ind_max, sep_length
    )


def print_circuit_symbolic(
    circuit: Circuit,
    symbol_map: Optional[dict] = None,
    qubit_min: int = 0,
    qubit_max: int = jnp.inf,  # type: ignore
    gate_ind_min: int = 0,
    gate_ind_max: int = jnp.inf,  # type: ignore
    sep_length: int = 1,
) -> List[str]:
    """
    Returns and prints basic string representation of circuit
    with symbolic parameters.

    Note that the behaviour of print_circuit(circuit)
    is different to print_circuit_symbolic(circuit),
    print_circuit_symbolic will look for parameters in circuit.free_symbols()
    and if there are none it will assume that none of the gates require parameters.
    On the other hand, print_circuit will work out which gates are parameterised
    based on e.g. circuit.get_commands()[0].op.params

    :param circuit: Circuit to be converted.
    :type circuit: pytket.Circuit
    :param symbol_map: dict that maps elements of circuit.free_symbols() (sympy)
        to parameter indices.
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
    :rtype: List[str] each element represents a qubit
    """
    g, q, p, nq = tk_to_qujax_args_symbolic(circuit, symbol_map)
    return qujax.print_circuit(  # type: ignore
        g, q, p, nq, qubit_min, qubit_max, gate_ind_min, gate_ind_max, sep_length
    )


def qujax_to_tk(
    gate_seq: Sequence[str],
    qubit_inds_seq: Sequence[Sequence[int]],
    param_inds_seq: Sequence[Sequence[int]],
    n_qubits: Optional[int] = None,
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
    :param n_qubits: Number of qubits, if fixed.
    :type n_qubits: int
    :return: Circuit
    :rtype: pytket.Circuit
    """
    if any(not isinstance(gate_name, str) for gate_name in gate_seq):
        raise TypeError("qujax_to_tk only currently only supports gates as strings")

    qujax.check_circuit(gate_seq, qubit_inds_seq, param_inds_seq, n_qubits)

    if n_qubits is None:
        n_qubits = max([max(qi) for qi in qubit_inds_seq]) + 1

    c = Circuit(n_qubits)

    for g, q, p in zip(gate_seq, qubit_inds_seq, param_inds_seq):
        g_apply_func = c.__getattribute__(g)
        g_apply_func(*p, *q)

    return c
