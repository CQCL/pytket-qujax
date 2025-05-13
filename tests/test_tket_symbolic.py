# Copyright Quantinuum
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

from collections.abc import Sequence

import pytest
from jax import grad, jit, random
from jax import numpy as jnp
from sympy import Symbol

from pytket.circuit import Circuit, OpType
from pytket.extensions.qujax import (
    qujax_args_to_tk,
    tk_to_qujax,
    tk_to_qujax_args,
)


def _test_circuit(
    circuit: Circuit, symbols: Sequence[Symbol], test_two_way: bool = False
) -> None:
    params = random.uniform(random.PRNGKey(0), (len(symbols),)) * 2
    param_map = dict(zip(symbols, params, strict=False))
    symbol_map = dict(zip(symbols, range(len(symbols)), strict=False))

    circuit_inst = circuit.copy()
    circuit_inst.symbol_substitution(param_map)
    true_sv = circuit_inst.get_statevector()
    true_probs = jnp.square(jnp.abs(true_sv))

    apply_circuit = tk_to_qujax(circuit, symbol_map)
    jit_apply_circuit = jit(apply_circuit)

    apply_circuit_dt = tk_to_qujax(circuit, symbol_map, simulator="densitytensor")
    jit_apply_circuit_dt = jit(apply_circuit_dt)

    if not len(params):
        test_sv = apply_circuit().flatten()
        test_jit_sv = jit_apply_circuit().flatten()

        test_dt = apply_circuit_dt()
        n_qubits = test_dt.ndim // 2
        test_dm_diag = jnp.diag(test_dt.reshape(2**n_qubits, 2**n_qubits))
        test_jit_dm_diag = jnp.diag(
            jit_apply_circuit_dt().reshape(2**n_qubits, 2**n_qubits)
        )
    else:
        test_sv = apply_circuit(params).flatten()
        test_jit_sv = jit_apply_circuit(params).flatten()
        test_dt = apply_circuit_dt(params)
        n_qubits = test_dt.ndim // 2
        test_dm_diag = jnp.diag(test_dt.reshape(2**n_qubits, 2**n_qubits))
        test_jit_dm_diag = jnp.diag(
            jit_apply_circuit_dt(params).reshape(2**n_qubits, 2**n_qubits)
        )

    assert jnp.allclose(test_sv, true_sv)
    assert jnp.allclose(test_jit_sv, true_sv)
    assert jnp.allclose(test_dm_diag.real, true_probs)
    assert jnp.allclose(test_jit_dm_diag, true_probs)

    if len(params):
        cost_func = lambda p: jnp.square(apply_circuit(p)).real.sum()
        grad_cost_func = grad(cost_func)
        assert isinstance(grad_cost_func(params), jnp.ndarray)

        cost_jit_func = lambda p: jnp.square(jit_apply_circuit(p)).real.sum()
        grad_cost_jit_func = grad(cost_jit_func)
        assert isinstance(grad_cost_jit_func(params), jnp.ndarray)

    if test_two_way:
        circuit_commands = [
            com for com in circuit.get_commands() if str(com.op) != "Barrier"
        ]
        circuit_2 = qujax_args_to_tk(*tk_to_qujax_args(circuit, symbol_map), params)  # type: ignore
        assert all(
            g.op.type == g2.op.type
            for g, g2 in zip(circuit_commands, circuit_2.get_commands(), strict=False)
        )
        assert all(
            g.qubits == g2.qubits
            for g, g2 in zip(circuit_commands, circuit_2.get_commands(), strict=False)
        )


def test_H() -> None:
    symbols: Sequence = []

    circuit = Circuit(3)
    circuit.H(0)

    _test_circuit(circuit, symbols, True)


def test_CX() -> None:
    symbols = [Symbol("p0")]

    circuit = Circuit(2)
    circuit.H(0)
    circuit.Rz(symbols[0], 0)
    circuit.CX(0, 1)

    _test_circuit(circuit, symbols, True)


def test_CX_qrev() -> None:
    symbols = [Symbol("p0"), Symbol("p1")]

    circuit = Circuit(2)
    circuit.Rx(symbols[0], 0)
    circuit.Rx(symbols[1], 1)
    circuit.CX(1, 0)

    _test_circuit(circuit, symbols, True)


def test_CZ() -> None:
    symbols = [Symbol("p0")]

    circuit = Circuit(2)
    circuit.H(0)
    circuit.Rz(symbols[0], 0)
    circuit.CZ(0, 1)

    _test_circuit(circuit, symbols, True)


def test_single_symbol_manipulation() -> None:
    symbols = [Symbol("p0")]

    circuit = Circuit(2)
    circuit.H(0)
    circuit.Rz(1.2 * symbols[0], 0)
    circuit.CZ(0, 1)

    _test_circuit(circuit, symbols, False)


def test_single_symbol_manipulaton_multiple_appearances() -> None:
    symbols = [Symbol("p0")]

    circuit = Circuit(2)
    circuit.H(0)
    circuit.Rz(1.2 * symbols[0], 0)
    circuit.Rz(0.5 * symbols[0], 1)
    circuit.Rx(symbols[0], 1)
    circuit.CZ(0, 1)

    _test_circuit(circuit, symbols, False)


def test_multiple_symbol_manipulaton() -> None:
    symbols = [Symbol("p0"), Symbol("p1"), Symbol("p2")]

    circuit = Circuit(2)
    circuit.H(0)
    circuit.Rz(1.2 * symbols[0], 0)
    circuit.Rz(0.5 * symbols[1], 1)
    circuit.XXPhase(symbols[0], 0, 1)
    circuit.YYPhase(0.2 * symbols[0], 0, 1)
    circuit.ZZPhase(0.3 * symbols[1], 0, 1)
    circuit.YYPhase(symbols[1], 0, 1)
    circuit.U3(0.1 * symbols[0], 0.7 * symbols[1], 0.8 * symbols[2], 1)
    circuit.CZ(0, 1)

    _test_circuit(circuit, symbols, False)


def test_CZ_qrev() -> None:
    symbols = [Symbol("p0")]

    circuit = Circuit(2)
    circuit.H(0)
    circuit.Rz(symbols[0], 0)
    circuit.CZ(1, 0)

    _test_circuit(circuit, symbols, True)


def test_symbolic_numeric_blend_circuit() -> None:
    symbols = [Symbol("p0")]

    circuit = Circuit(2)
    circuit.H(0)
    circuit.Rz(symbols[0], 1)
    circuit.Rx(0.1, 1)
    circuit.Rz(0.2, 1)

    _test_circuit(circuit, symbols)


def test_symbolic_numeric_blend_gate() -> None:
    symbols = [Symbol("p0"), Symbol("p1"), Symbol("p2")]

    circuit = Circuit(1)
    circuit.add_gate(OpType.U2, [symbols[0] * 3.0, symbols[1] + symbols[2] + 2], [0])

    _test_circuit(circuit, symbols)

    circuit = Circuit(1)
    circuit.H(0)
    circuit.Rx(symbols[2], 0)
    circuit.add_gate(OpType.U2, [symbols[0], symbols[1]], [0])

    _test_circuit(circuit, symbols)


def test_not_in_qujaxgates() -> None:
    symbols = [Symbol("p0")]

    circuit = Circuit(3)
    circuit.Rx(symbols[0], 0)
    circuit.XXPhase3(0.1, 0, 1, 2)

    _test_circuit(circuit, symbols)


def test_CX_Barrier_Rx() -> None:
    symbols = [Symbol("p0"), Symbol("p1")]

    circuit = Circuit(3)
    circuit.CX(0, 1)
    circuit.add_barrier([0, 2])
    circuit.Rx(symbols[0], 0)
    circuit.Rx(symbols[1], 2)

    _test_circuit(circuit, symbols)


def test_measure_error() -> None:
    symbols = [Symbol("p0")]
    symbol_map = dict(zip(symbols, range(len(symbols)), strict=False))

    circuit = Circuit(3, 3)
    circuit.Rx(symbols[0], 0)
    circuit.Measure(0, 0)

    with pytest.raises(TypeError):
        _ = tk_to_qujax(circuit, symbol_map)


def test_circuit1() -> None:
    n_qubits = 4
    depth = 1
    symbols = [Symbol(f"p{j}") for j in range(n_qubits * (depth + 1))]

    circuit = Circuit(n_qubits)
    k = 0
    for i in range(n_qubits):
        circuit.Ry(symbols[k], i)
        k += 1
    for _ in range(depth):
        for i in range(0, n_qubits - 1, 2):
            circuit.CX(i, i + 1)
        for i in range(1, n_qubits - 1, 2):
            circuit.CX(i, i + 1)
        circuit.add_barrier(list(range(n_qubits)))
        for i in range(n_qubits):
            circuit.Ry(symbols[k], i)
            k += 1

    _test_circuit(circuit, symbols)


def test_circuit2() -> None:
    n_qubits = 3
    depth = 1
    symbols = [Symbol(f"p{j}") for j in range(2 * n_qubits * (depth + 1))]

    circuit = Circuit(n_qubits)
    k = 0
    for i in range(n_qubits):
        circuit.H(i)
    for i in range(n_qubits):
        circuit.Rz(symbols[k], i)
        k += 1
    for i in range(n_qubits):
        circuit.Rx(symbols[k], i)
        k += 1
    for _ in range(depth):
        for i in range(n_qubits - 1):
            circuit.CZ(i, i + 1)
        circuit.add_barrier(list(range(n_qubits)))
        for i in range(n_qubits):
            circuit.Rz(symbols[k], i)
            k += 1
        for i in range(n_qubits):
            circuit.Rx(symbols[k], i)
            k += 1

    _test_circuit(circuit, symbols)


def test_HH() -> None:
    circuit = Circuit(3)
    circuit.H(0)

    apply_circuit = tk_to_qujax(circuit)

    st1 = apply_circuit()
    st2 = apply_circuit(st1)
    all_zeros_sv = (jnp.arange(st2.size) == 0).astype(int)
    assert jnp.all(jnp.abs(st2.flatten() - all_zeros_sv) < 1e-5)


def test_exception_symbol_map() -> None:
    symbols = [Symbol("p0"), Symbol("p1"), Symbol("bad_bad_symbol")]

    circuit = Circuit(2)
    circuit.Rx(symbols[0], 0)
    circuit.Rx(symbols[1], 1)
    circuit.CX(1, 0)

    with pytest.raises(AssertionError):
        _test_circuit(circuit, symbols)
