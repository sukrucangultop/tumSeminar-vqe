import numpy as np
import math
from scipy.optimize import minimize
from qiskit.circuit import ParameterVector
from qiskit.opflow import commutator
from .Helpers import string2Operator, decomposeHamiltonian, pauliOperator2Dict

from qiskit import *
from qiskit.opflow import I, X, Y, Z, CircuitStateFn, StateFn, PauliExpectation, CircuitSampler
from qiskit.utils import QuantumInstance
from qiskit.opflow.primitive_ops import PauliSumOp

class ADAPTVQE:

    def __init__(self, H, initialState = None, gradientThreshold=1e-2, maxIterationCount = 10, shots = 8192, optimizer = "COBYLA", tol = 1e-3, verbose=False):
        """
        Initializes qubit-ADAPT-VQE algorithm

        Args:
            H (PauliSumOp or ndaarray): Hamiltonian in matrix or PauliSumOp form
            initialState (QunatumCircuit, optional): Initial quantum circuit. If you want to start from Hartree Fock state, add that circuit to here. Defaults to None.
            gradientThreshold (double, optional): Iteration stop threshold for gradients. Defaults to 1e-2.
            maxIterationCount (int, optional): Maximum iteration count. Defaults to 10.
            shots (int, optional): Measurement in VQE and gradient estimation shot count. Defaults to 8192.
            optimizer (str, optional): Optimizer name. Defaults to "COBYLA".
            tol (double, optional): Optimizer tolerance. Defaults to 1e-3.
            verbose (bool, optional): If set True, prints the iteration information. Defaults to False.

        Raises:
            ValueError: [description]
            ValueError: [description]
        """
        if type(H) == PauliSumOp:
            self.H = H
            self.n = self._getQubitCount(H)
        elif type(H) == np.ndarray:
            (x, y) = H.shape
            if not ((x & (x-1) == 0) and x != 0 and ((y & (y-1) == 0) and y != 0 and x == y)):
                raise ValueError("Hamiltonian should be in the shape (2^n, 2^n)")
            self.n = int(math.log2(x))
            self.H = decomposeHamiltonian(H)
        else:
            raise ValueError("Only supported hamiltonian types are: PauliSumOp and numpy.ndarray")
        self.n = self._getQubitCount(H)
        self.shots = shots
        self.optimizer = optimizer
        self.tol = tol
        self.gradientThreshold = gradientThreshold
        self.params = None
        self.optimizedParams = None
        self.maxIterationCount = maxIterationCount
        self.pauliDict = pauliOperator2Dict(H)
        self.initialState = initialState
        self.verbose = verbose
        self.log = ""

    def _getQubitCount(self, pauliSumOp):
        """ 
        Gets qubit count according to Hamiltonian size

        Args:
            pauliSumOp (PauliSumOp): Hamiltonian as PauliSumOP

        Returns:
            integer: Qubit count to represent Hamiltonian
        """
        return len(pauliSumOp.to_pauli_op().oplist[0].primitive.to_label())

    def _getCompleteMinimalPool(self, n):
        """
        Creates complete minimal pool for qubit count according to qubit-adapt paper

        Raises:
            ValueError: Qubit count validation

        Returns:
            list: List of operators generating complete pool
        """
        if n < 2:
            raise ValueError("Minimal pool has at least 2 qubits")
        if n == 2:
            return [Y ^ Z, I ^ Y]

        oldPool = self._getCompleteMinimalPool(n-1)
        nY = 1
        nYminus1 = 1
        for i in range(n-2):
            nY = nY ^ I
            nYminus1 = nYminus1^I
        nY = (nY ^ I ^ Y)
        nYminus1 = (nYminus1 ^ Y ^ I)
        newPool = [x ^ Z for x in oldPool]
        newPool.append(nY)
        newPool.append(nYminus1)
        return newPool

    def _expZBasis(self, circuit, quantum_register, pauli_idexes, parameter=0):
        """f
        The implementation of exp(iZZ..Z t), where Z is the Pauli Z operator, t is a parameter.

        Args:
            circuit (QuantumCircuit): initial Quantum Circuit
            quantum_register (QuantumRegister): Register holding current ansatz
            pauli_idexes ([type]): the indexes from quantum_register that correspond to entries not equal to I: 
                            e.g. if we have XIYZI then the 
                            pauli_idexes = [0,2,3].
            parameter (double, optional): the parameter t in exp(iZZ..Z t). Defaults to 0.
        """
        # the first CNOTs
        for i in range(len(pauli_idexes) - 1):
            circuit.cx(quantum_register[pauli_idexes[i]],
                    quantum_register[pauli_idexes[i + 1]])
        circuit.rz(parameter, quantum_register[pauli_idexes[-1]])
        # the second CNOTs
        for i in reversed(range(len(pauli_idexes) - 1)):
            circuit.cx(quantum_register[pauli_idexes[i]],
                    quantum_register[pauli_idexes[i + 1]])
        circuit.barrier(quantum_register)
                    
    def _expPauli(self, pauli, circuit, parameter=0):
        """
        The circuit for the exp(i P t), where P is the Pauli term, t is the parameter.
        :return: QuantumCircuit that implements exp(i P t) or 
                control version of it.

        Args:
            pauli (string): the string for the Pauli term e.g. "XIXY".
            circuit (QuantumCircuit): QuantumCircuit containing current ansatz
            parameter (double, optional): the parameter t in exp(i<P> t). Defaults to 0.

        Raises:
            Exception: [description]

        Returns:
            [type]: [description]
        """
        quantum_register = circuit.qregs[0]
        if len(pauli) != len(quantum_register):
            raise Exception("Pauli string doesn't match to the quantum register")

        pauli_circuit = circuit
        circuit_bracket = QuantumCircuit(quantum_register)
        pauli_idexes = []

        for i in range(len(quantum_register)):
            if pauli[i] == 'I':
                continue
            elif pauli[i] == 'Z':
                pauli_idexes.append(i)
            elif pauli[i] == 'X':
                circuit_bracket.h(quantum_register[i])
                pauli_idexes.append(i)
            elif pauli[i] == 'Y':
                circuit_bracket.u(np.pi / 2, np.pi / 2, np.pi / 2, quantum_register[i])
                pauli_idexes.append(i)

        pauli_circuit = pauli_circuit.compose(circuit_bracket)
        self._expZBasis(pauli_circuit, quantum_register, pauli_idexes, parameter)
        pauli_circuit = pauli_circuit.compose(circuit_bracket)

        return pauli_circuit

    def _getGradient(self, circuit, hamiltonian, operator):
        """
        Estimates the gradient of the new operator with <psi|[H, operator]|psi>.

        Args:
            circuit (QuantumCircuit): Current state vector as circuit
            hamiltonian (PauliSumOp): Huamiltonian
            operator (PauliOp): Operator from the pool

        Returns:
            complex number: Gradient of the operator
        """
        psi = CircuitStateFn(primitive=circuit, coeff=1.)
        op = commutator(hamiltonian, 1j*operator)  
        backend = Aer.get_backend('qasm_simulator') 
        q_instance = QuantumInstance(backend, shots=self.shots)

        measurable_expression = StateFn(op, is_measurement=True).compose(psi) 
        expectation = PauliExpectation().convert(measurable_expression)  
        sampler = CircuitSampler(q_instance).convert(expectation) 
        gradient = sampler.eval()
        self.log += "Gradient of operator {} is {} \n".format(operator, gradient)
        return gradient

    def _quantumStatePreparation(self, circuit, parameters):
        """
        Classical VQE quantum state preparation with binding parameters to circuit

        Args:
            circuit (QuantumCircuit): Quantumm circuit representing parametrised ansatz
            parameters (list): Parameters for binding to circuit

        Returns:
            QuantumCircuit: Parameter binded ansatz
        """
        return circuit.assign_parameters({self.params: parameters}, inplace=False)

    def _quantumModule(self, parameters, measure):
        """
        Classical VQE quantum module, estimates expecation value of ansatz.

        Args:
            parameters (list): Parameters for ansatz
            measure (PauliOp): Pauli string for measurement basis.

        Returns:
            double: Measurement result in standard basis.
        """
        if measure == 'I'*self.n:
            return 1
        else:
            q = QuantumRegister(self.n)
            circuit = self.ansatz
            circuit = self._quantumStatePreparation(circuit, parameters)
        return self._measureStardardBasis(circuit, string2Operator(measure))
    
    def _measureStardardBasis(self, circuit, operator):
        """
        Measures circuit in given operator basis with converting it to standard basis.

        Args:
            circuit (QuantumCircuit): Ansatz to measure
            operator (PauliOp): Observable to measure

        Returns:
            double: Measurement results
        """
        backend = BasicAer.get_backend('qasm_simulator')
        q_instance = QuantumInstance(backend, shots=self.shots)
        psi = CircuitStateFn(circuit)
        measurable_expression = StateFn(operator, is_measurement=True).compose(psi) 
        expectation = PauliExpectation().convert(measurable_expression)  
        sampler = CircuitSampler(q_instance).convert(expectation) 
        mean = sampler.eval().real
        return mean

    def _vqe(self, parameters):
        """
        Classical VQR step. Estimates the expectation value of ansatz with hamiltonian.

        Args:
            parameters (list): Ansatz parameters.

        Returns:
            double: Expectation value estimate.
        """
        classical_adder = 0.0
        for key, value in self.pauliDict.items():
            mean = self._quantumModule(parameters, key)
            module = value * mean
            classical_adder += module
        return classical_adder
    
    def _optimizeVQE(self, initialParameters):
        """
        Runs Classical VQE.

        Args:
            initialParameters (list): initial parameters of ansatz.

        Returns:
            (double, parameters): Eigenvalue estimation and optimized parameters tuple
        """
        vqeResult = minimize(self._vqe, initialParameters, method=self.optimizer, tol=self.tol)
        return vqeResult.x, vqeResult.fun

    def run(self):
        """
        Runs qubit-adapt-VQE,

        Returns:
            (double, QuantumCircuit, list): Estimated eigenvalue, Final ansatz, optimized paramters of ansatz tuple
        """
        if self.initialState is None:
            q = QuantumRegister(self.n)
            mainCircuit = QuantumCircuit(q)
        else:
            mainCircuit = self.initialState
        self.ansatz = mainCircuit
        minEigenValue = None
        previousParameters = []
        for i in range(1, self.maxIterationCount+1):
            self.log += "========================\n Iteration: {} \n".format(i)
            # Complete minimal pool for qubit-adapt
            pool = self._getCompleteMinimalPool(self.n)
            gradients = []
            # Prepare ansatz with optimized parameters in previous iteration
            if i > 1:
                bound_circuit = mainCircuit.bind_parameters({self.params: previousParameters})
            else:
                bound_circuit = mainCircuit
            # Get gradient of each operator in the pool for current ansatz.
            for index, operator in enumerate(pool):
                gradients.append(np.absolute(self._getGradient(bound_circuit, self.H, operator)))
            maxGradient = max(gradients)
            selectedOperator = pool[gradients.index(maxGradient)]
            self.log += "Selected operator {} with max gradient {} \n".format(selectedOperator, maxGradient)
            # There is an issue with operator pool, most of the time in the first iteration exact gradient is zero since initial state is eighter in |0> state 
            # or Hartree Fock initial state. So I am skipping first threshold check in first step to at least add a new operator to circuit since each opeartor can be reversed with more operators from pool.
            if maxGradient < self.gradientThreshold and i>1:
                self.log += "Max gradient {} is smaller than threshold {}, terminating algorithm!\n".format(maxGradient, self.gradientThreshold)
                if self.verbose:
                    print(self.log)
                    self.log = ""
                self.optimizedParams = previousParameters
                return minEigenValue, self.ansatz, previousParameters
            if self.params is None:
                self.params = ParameterVector('param_list', i)
            else:
                self.params.resize(i)
            # Grow the ansatz with adding opeartor simulation to current ansatz
            mainCircuit = self._expPauli(selectedOperator.primitive.to_label(), mainCircuit, self.params[i-1])
            self.ansatz = mainCircuit
            # Add new parameter with 0 initial state.
            previousParameters = np.append(previousParameters, [0], axis=0)
            previousParameters, minEigenValue = self._optimizeVQE(previousParameters)
            self.log += "Eigenvalue : {} \n======================== \n\n".format(minEigenValue)
            if self.verbose:
                print(self.log)
                self.log = ""
        self.optimizedParams = previousParameters
        return minEigenValue, self.ansatz, previousParameters

    def __str__(self):
        return str(self.draw())

    def draw(self, output="text", **kwargs):
        """
        Draw the ansatz circuit to different formats with the final paramters.
        Args:
            output (str, optional): The output method used for drawing the
            circuit. Valid choices are ``text``, ``latex``, ``latex_source``,
            ``mpl``. Default is ``text``.
            kwargs: Additional keyword arguments to be passed to
            qiskit.Quantumcircuit.draw().
        Returns:
            `PIL.Image` (output='latex') or `matplotlib.figure`
            (output='mpl') or `str` (output='latex_src') or `TextDrawing`
            (output='text').
        """
        if self.optimizedParams.size != 0:
            circuit = self.ansatz.bind_parameters({self.params: self.optimizedParams})
            return circuit.draw(output, **kwargs)
        else:
            return self.ansatz.draw(output, **kwargs)