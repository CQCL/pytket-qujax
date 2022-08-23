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

from typing import Tuple, Sequence, Optional
from jax import numpy as jnp
from qujax import UnionCallableOptionalArray, get_params_to_statetensor_func  # type: ignore
from pytket import Qubit, Circuit  # type: ignore


def _tk_qubits_to_inds(tk_qubits: Sequence[Qubit]) -> Tuple[int, ...]:
    """
    Convert Sequence of tket qubits objects to Tuple of integers qubit indices.

    :param tk_qubits: Sequence of tket qubit object
        (as stored in pytket.Circuit.qubits).
    :type tk_qubits: Sequence[Qubit]
    :return: Tuple of qubit indices.
    :rtype: tuple
    """
    return tuple(q.index[0] for q in tk_qubits)


def tk_to_qujax(circuit: Circuit) -> UnionCallableOptionalArray:
    """
    Converts a tket circuit into a function that maps circuit parameters
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

    return get_params_to_statetensor_func(
        gate_name_seq, qubit_inds_seq, param_inds_seq, circuit.n_qubits
    )


def tk_to_qujax_symbolic(
    circuit: Circuit, symbol_map: Optional[dict] = None
) -> UnionCallableOptionalArray:
    """
    Converts a tket circuit with symbolics parameters and a symbolic parameter map
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

    return get_params_to_statetensor_func(
        gate_name_seq, qubit_inds_seq, param_inds_seq, circuit.n_qubits
    )
