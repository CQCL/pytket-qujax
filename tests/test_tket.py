# Copyright 2020-2022 Cambridge Quantum Computing
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

from typing import Union, Any
from jax import numpy as jnp, jit, grad, random  # type: ignore
import qujax  # type: ignore
import pytest

from pytket.circuit import Circuit, Qubit  # type: ignore
from pytket.pauli import Pauli, QubitPauliString  # type: ignore
from pytket.utils import QubitPauliOperator  # type: ignore
from pytket.extensions.qujax import tk_to_qujax, tk_to_qujax_args, qujax_args_to_tk, tk_to_param


def _test_circuit(
    circuit: Circuit, param: Union[None, jnp.ndarray], test_two_way: bool = False
) -> None:
    true_sv = circuit.get_statevector()

    apply_circuit = tk_to_qujax(circuit)
    jit_apply_circuit = jit(apply_circuit)

    if param is None:
        test_sv = apply_circuit().flatten()
        test_jit_sv = jit_apply_circuit().flatten()
    else:
        test_sv = apply_circuit(param).flatten()
        test_jit_sv = jit_apply_circuit(param).flatten()

        assert jnp.allclose(param, tk_to_param(circuit))

    assert jnp.all(jnp.abs(test_sv - true_sv) < 1e-5)
    assert jnp.all(jnp.abs(test_jit_sv - true_sv) < 1e-5)

    if param is not None:
        cost_func = lambda p: jnp.square(apply_circuit(p)).real.sum()
        grad_cost_func = grad(cost_func)
        assert isinstance(grad_cost_func(param), jnp.ndarray)

        cost_jit_func = lambda p: jnp.square(jit_apply_circuit(p)).real.sum()
        grad_cost_jit_func = grad(cost_jit_func)
        assert isinstance(grad_cost_jit_func(param), jnp.ndarray)

    if test_two_way:
        circuit_commands = [
            com for com in circuit.get_commands() if str(com.op) != "Barrier"
        ]
        circuit_2 = qujax_args_to_tk(*tk_to_qujax_args(circuit), param)  # type: ignore
        assert all(
            g.op.type == g2.op.type
            for g, g2 in zip(circuit_commands, circuit_2.get_commands())
        )
        assert all(
            g.qubits == g2.qubits
            for g, g2 in zip(circuit_commands, circuit_2.get_commands())
        )


def test_H() -> None:
    circuit = Circuit(3)
    circuit.H(0)

    _test_circuit(circuit, None, True)


def test_CX() -> None:
    param = jnp.array([0.25])  # type: ignore

    circuit = Circuit(2)
    circuit.H(0)
    circuit.Rz(param[0], 0)
    circuit.CX(0, 1)

    _test_circuit(circuit, param, True)


def test_CX_callable() -> None:
    param = jnp.array([0.25])  # type: ignore

    def H() -> Any:
        return qujax.gates.H

    def Rz(p: float) -> Any:
        return qujax.gates.Rz(p)

    def CX() -> Any:
        return qujax.gates.CX

    gate_seq = [H, Rz, CX]
    qubit_inds_seq = [[0], [0], [0, 1]]
    param_inds_seq = [[], [0], []]

    apply_circuit = qujax.get_params_to_statetensor_func(
        gate_seq, qubit_inds_seq, param_inds_seq
    )

    circuit = Circuit(2)
    circuit.H(0)
    circuit.Rz(param[0], 0)
    circuit.CX(0, 1)
    true_sv = circuit.get_statevector()

    test_st = apply_circuit(param)
    test_sv = test_st.flatten()
    assert jnp.all(jnp.abs(test_sv - true_sv) < 1e-5)

    jit_apply_circuit = jit(apply_circuit)
    test_jit_sv = jit_apply_circuit(param).flatten()
    assert jnp.all(jnp.abs(test_jit_sv - true_sv) < 1e-5)


def test_CX_qrev() -> None:
    param = jnp.array([0.2, 0.8])  # type: ignore

    circuit = Circuit(2)
    circuit.Rx(param[0], 0)
    circuit.Rx(param[1], 1)
    circuit.CX(1, 0)

    _test_circuit(circuit, param, True)


def test_CZ() -> None:
    param = jnp.array([0.25])  # type: ignore

    circuit = Circuit(2)
    circuit.H(0)
    circuit.Rz(param[0], 0)
    circuit.CZ(0, 1)

    _test_circuit(circuit, param, True)


def test_CZ_qrev() -> None:
    param = jnp.array([0.25])  # type: ignore

    circuit = Circuit(2)
    circuit.H(0)
    circuit.Rz(param[0], 0)
    circuit.CZ(1, 0)

    _test_circuit(circuit, param, True)


def test_CX_Barrier_Rx() -> None:
    param = jnp.array([0, 1 / jnp.pi])  # type: ignore

    circuit = Circuit(3)
    circuit.CX(0, 1)
    circuit.add_barrier([0, 2])
    circuit.Rx(param[0], 0)
    circuit.Rx(param[1], 2)

    _test_circuit(circuit, param)


def test_circuit1() -> None:
    n_qubits = 4
    depth = 1

    param = random.uniform(random.PRNGKey(0), (n_qubits * (depth + 1),)) * 2

    circuit = Circuit(n_qubits)

    k = 0
    for i in range(n_qubits):
        circuit.Ry(param[k], i)
        k += 1

    for _ in range(depth):
        for i in range(0, n_qubits - 1, 2):
            circuit.CX(i, i + 1)
        for i in range(1, n_qubits - 1, 2):
            circuit.CX(i, i + 1)
        circuit.add_barrier(range(0, n_qubits))
        for i in range(n_qubits):
            circuit.Ry(param[k], i)
            k += 1

    _test_circuit(circuit, param)


def test_circuit2() -> None:
    n_qubits = 3
    depth = 1

    param = random.uniform(random.PRNGKey(0), (2 * n_qubits * (depth + 1),)) * 2

    circuit = Circuit(n_qubits)

    k = 0
    for i in range(n_qubits):
        circuit.H(i)
    for i in range(n_qubits):
        circuit.Rz(param[k], i)
        k += 1
    for i in range(n_qubits):
        circuit.Rx(param[k], i)
        k += 1

    for _ in range(depth):
        for i in range(0, n_qubits - 1):
            circuit.CZ(i, i + 1)
        circuit.add_barrier(range(0, n_qubits))
        for i in range(n_qubits):
            circuit.Rz(param[k], i)
            k += 1
        for i in range(n_qubits):
            circuit.Rx(param[k], i)
            k += 1

    _test_circuit(circuit, param)


def test_HH() -> None:
    circuit = Circuit(3)
    circuit.H(0)

    apply_circuit = tk_to_qujax(circuit)

    st1 = apply_circuit()
    st2 = apply_circuit(st1)

    all_zeros_sv = jnp.array(jnp.arange(st2.size) == 0, dtype=int)  # type: ignore

    assert jnp.all(jnp.abs(st2.flatten() - all_zeros_sv) < 1e-5)


def test_measure_error() -> None:
    circuit = Circuit(3, 3)
    circuit.H(0)
    circuit.Measure(0, 0)

    with pytest.raises(TypeError):
        _ = tk_to_qujax(circuit)


def test_quantum_hamiltonian() -> None:
    n_qubits = 5

    strings_zz = [
        QubitPauliString({Qubit(j): Pauli.Z, Qubit(j + 1): Pauli.Z})
        for j in range(n_qubits - 1)
    ]
    coefs_zz = random.normal(random.PRNGKey(0), shape=(len(strings_zz),))
    tket_op_dict_zz = dict(zip(strings_zz, coefs_zz.tolist()))
    strings_x = [QubitPauliString({Qubit(j): Pauli.X}) for j in range(n_qubits)]
    coefs_x = random.normal(random.PRNGKey(0), shape=(len(strings_x),))
    tket_op_dict_x = dict(zip(strings_x, coefs_x.tolist()))
    tket_op = QubitPauliOperator({**tket_op_dict_zz, **tket_op_dict_x})

    gate_str_seq_seq = [["Z", "Z"]] * (n_qubits - 1) + [["X"]] * n_qubits
    qubit_inds_seq = [[i, i + 1] for i in range(n_qubits - 1)] + [
        [i] for i in range(n_qubits)
    ]
    st_to_exp = qujax.get_statetensor_to_expectation_func(
        gate_str_seq_seq, qubit_inds_seq, jnp.concatenate([coefs_zz, coefs_x])
    )

    state = random.uniform(random.PRNGKey(0), shape=(2**n_qubits,))
    state /= jnp.linalg.norm(state)

    tket_exp = tket_op.state_expectation(state)  # type: ignore
    jax_exp = st_to_exp(state.reshape((2,) * n_qubits))

    assert jnp.abs(tket_exp - jax_exp) < 1e-5
