import itertools, functools
from qiskit.opflow import I, X, Y, Z
import numpy as np


def pauliOperator2Dict(pauliOperator):
    """
    Converts pauli operator to dict: pauli string as keys and coefficient as value
    Example:
    {I: 0.4, X: 0.6, Z: 0.1, Y: 0.5}.
    
    Args:
        pauliOperator (PauliSumOp): Pauli operator

    Returns:
        dict: Dictionary containing pauli strings as key and coefficient as value
    """
    paulis = pauliOperator.to_pauli_op().oplist
    paulis_dict = {}

    for x in paulis:
        if x.coeff == 0.0:
            continue
        label = x.primitive.to_label()
        coeff = x.coeff
        paulis_dict[label] = coeff

    return paulis_dict


def string2Operator(string):
    """
    Converts pauli string to Qiskit's pauli operator

    Args:
        string (string): Pauli string

    Returns:
        PauliOp: Pauli operator representing pauli string
    """
    operator = 1
    for s in string:
        if s == "I":
            operator = operator ^ I
        if s == "X":
            operator = operator ^ X
        if s == "Y":
            operator = operator ^ Y
        if s == "Z":
            operator = operator ^ Z
    return operator


def hamiltonianOperator(n, parameters=None):
    """
    Generates Hamiltonian with n qubit number with given parameters

    Args:
        n (integer): Qubit count
        parameters (array, optional): Coefficients of pauili strings. Defaults to None. If None, all coefficients are 1.

    Raises:
        ValueError: Parameter and qubit count count doesn't match

    Returns:
        PauliSumOp: Hamiltonian as pauli strings sum
    """
    if parameters is None:
        parameters = [1 for i in range(4**n)]
    if len(parameters) != 4**n:
        raise ValueError(
            'Not valid shape, give parameters matching with 4^qubit count')
    ret = 0
    for string in range(4**n):
        if parameters[string] == 0:
            continue
        tempString = string
        literal = 1
        for i in range(n):
            if tempString % 4 == 0:
                literal = I ^ literal
            if tempString % 4 == 1:
                literal = X ^ literal
            if tempString % 4 == 2:
                literal = Y ^ literal
            if tempString % 4 == 3:
                literal = Z ^ literal
            tempString = tempString // 4
        ret += parameters[string] * literal
    return ret


def decomposeHamiltonian(H):
    """ 
    Decomposes given hamiltonian into sum of pauli strings

    Args:
        H (Numpy Matrix): Hamiltonian matrix to decompose

    Raises:
        ValueError: Hamiltonian shape

    Returns:
        PauliSumOp: Decomposed hamiltonian into qiskit PauliSumOp representation
    """
    x, y = H.shape
    N = int(np.log2(len(H)))
    if len(H) - 2 ** N != 0 or x != y:
        raise ValueError(
            "Hamiltonian should be in the form (2^n x 2^n), for any n>=1")
    pauilis = [I, X, Y, Z]
    decomposedH = 0
    for term in itertools.product(pauilis, repeat=N):
        matrices = [i.to_matrix() for i in term]
        # coefficient of the pauli string = (1/2^N) * (Tr[pauliOp x H])
        coeff = np.trace(functools.reduce(np.kron, matrices) @ H) / (2**N)
        coeff = np.real_if_close(coeff).item()
        if coeff == 0:
            continue
        obs = 1
        for i in term:
            obs = obs ^ i
        decomposedH += coeff * obs
    return decomposedH