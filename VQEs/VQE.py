import numpy as np
import math
from random import random
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from .Helpers import *

from qiskit import *
from qiskit.opflow import CircuitStateFn, StateFn, PauliExpectation, CircuitSampler
from qiskit.opflow.primitive_ops import PauliSumOp
from qiskit.utils import QuantumInstance

class VQE:

    def __init__(self, H, ansatzDepth = 2, shots = 8192,  initialParameters = None, optimizer = "COBYLA", tol = 1e-3):
        """
        Initializes VQE instance

        Args:
            H (PauliSumOp or ndaarray): Hamiltonian in matrix or PauliSumOp form
            ansatzDepth (int, optional): Depth of the ansatz containing rotation gates and linear entanglers. Defaults to 2.
            shots (int, optional): Shots to estimate expectation of ansatz. Defaults to 8192.
            initialParameters ([type], optional): Initial parameters of ansatz. If none they are randomly selected. Defaults to None.
            optimizer (string, optional): Optimizer name. Defaults to "COBYLA".
            tol (double, optional): Optimizer tolerance. Defaults to 1e-3.

        Raises:
            ValueError: Hamiltonian shape validation
            ValueError: Hamiltonian inpit type validation
            ValueError: Parameter length validation
        """
        self.ansatzDepth = ansatzDepth
        self.shots = shots
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
        if initialParameters == None:
            self.initialParameters = [random() for i in range(self.n*2*(self.ansatzDepth+1))]
        elif len(initialParameters) != self.n*2*(self.ansatzDepth+1):
            raise ValueError("Parameters length should be 2*qubit_count*(depth+1)={}".format(self.n*2*(self.ansatzDepth+1)))
        else:
            self.initialParameters = initialParameters
        self.circuit = QuantumCircuit(self.n)
        self.pauliDict = pauliOperator2Dict(self.H)
        self.logs = []
        self.optimizer = optimizer
        self.tol = tol

    def _getQubitCount(self, pauliSumOp):
        """ 
        Gets qubit count according to Hamiltonian size

        Args:
            pauliSumOp (PauliSumOp): Hamiltonian as PauliSumOP

        Returns:
            integer: Qubit count to represent Hamiltonian
        """
        return len(pauliSumOp.to_pauli_op().oplist[0].primitive.to_label())

    def _quantumStatePreparation(self, circuit, parameters):
        """ 
        Crates ansatz containing (single qubit rotation gates + linear entaglements with CNOTs) x ansatzDepth + final single qubit rotation gates

        Example with ansatzDepth = 1:

             ┌──────────┐┌──────────┐ ░                 ░  ┌──────────┐┌───────────┐
        q_0: ┤ Ry(θ[0]) ├┤ Rz(θ[4]) ├─░───■─────────────░──┤ Ry(θ[8]) ├┤ Rz(θ[12]) ├
             ├──────────┤├──────────┤ ░ ┌─┴─┐           ░  ├──────────┤├───────────┤
        q_1: ┤ Ry(θ[1]) ├┤ Rz(θ[5]) ├─░─┤ X ├──■────────░──┤ Ry(θ[9]) ├┤ Rz(θ[13]) ├
             ├──────────┤├──────────┤ ░ └───┘┌─┴─┐      ░ ┌┴──────────┤├───────────┤
        q_2: ┤ Ry(θ[2]) ├┤ Rz(θ[6]) ├─░──────┤ X ├──■───░─┤ Ry(θ[10]) ├┤ Rz(θ[14]) ├
             ├──────────┤├──────────┤ ░      └───┘┌─┴─┐ ░ ├───────────┤├───────────┤
        q_3: ┤ Ry(θ[3]) ├┤ Rz(θ[7]) ├─░───────────┤ X ├─░─┤ Ry(θ[11]) ├┤ Rz(θ[15]) ├
             └──────────┘└──────────┘ ░           └───┘ ░ └───────────┘└───────────┘
        Args:
            circuit (QuantumCircuit): Circuit to create ansatz on
            parameters (list): Parameters for rotation gates

        Returns:
            QuantumCircuit: Circuit with parametrised ansatz
        """
        q = circuit.qregs[0]
        
        for d in range(self.ansatzDepth):
            for qubitIndex in range(self.n):
                circuit.ry(parameters[d*self.n*2 + qubitIndex], q[qubitIndex])
                circuit.rz(parameters[d*self.n*2 + qubitIndex + self.n], q[qubitIndex])
            for i in range(self.n-1):
                circuit.cx(q[i], q[i+1])
        for lastLayerQubitIndex in range(self.n):
            circuit.ry(parameters[self.ansatzDepth*self.n*2 + lastLayerQubitIndex], q[lastLayerQubitIndex])
            circuit.rz(parameters[self.ansatzDepth*self.n*2 + lastLayerQubitIndex + self.n], q[lastLayerQubitIndex])  
        self.circuit = circuit
        return circuit

    def _quantumModule(self, parameters, measure):
        """ 
        Wraps quantum part of VQE algorithm where estimation of expectation value occurs.

        Args:
            parameters (list): Parameters for ansatz
            measure (string): String literal of observable

        Returns:
            double: estimated expectation value of ansatz according to observable given as measure
        """
        if measure == 'I'*self.n:
            return 1
        else:
            q = QuantumRegister(self.n)
            circuit = QuantumCircuit(q)
            circuit = self._quantumStatePreparation(circuit, parameters)
        return self._measureStardardBasis(circuit, string2Operator(measure))
    
    def _measureStardardBasis(self, circuit, operator):
        """ 
        Measures circuit containing ansatz according to operator with shots given as parameter to instance.

        Args:
            circuit (QuantumCircuit): Quantum circuit to measure(containing ansatz with assigned parameters)
            operator (PauliOp): Basis for measurement

        Returns:
            double: Expectation value of ansatz
        """
        backend = BasicAer.get_backend('qasm_simulator')
        q_instance = QuantumInstance(backend, shots=self.shots)
        #Wave function representation of circuit
        psi = CircuitStateFn(circuit)
        #Create measurement expression to convert to standard basis.
        measurable_expression = StateFn(operator, is_measurement=True).compose(psi) 
        #Convert expectation to standard basis
        expectation = PauliExpectation().convert(measurable_expression)  
        #Measure circuits shots times
        sampler = CircuitSampler(q_instance).convert(expectation) 
        mean = sampler.eval().real
        return mean

    def _vqeStep(self, parameters):
        """
        VQE step for eigenvalue upper bound estimation. Creates circuits according to hamiltonian decomposition and sums the estimations of parts to get final estimation for eigenvalue.

        Args:
            parameters (list): List of parameters

        Returns:
            double: Minimum eigenvalue estimation
        """
        classicalAdder = 0.0
        for key, value in self.pauliDict.items():
            mean = self._quantumModule(parameters, key)
            module = value * mean
            classicalAdder += module
        self.logs.append(classicalAdder)
        return classicalAdder
    
    def run(self, plotIterations = True):
        """ Runs VQE algorithm to find minimum eigenvalue estimation

        Args:
            plotIterations (bool, optional): If true, plots the expectation/iteration plot. Defaults to True.

        Returns:
            OptimizeResult: Optimization result of the VQE algorithm
        """
        vqeResult = minimize(self._vqeStep, self.initialParameters, method=self.optimizer, tol=self.tol)
        if plotIterations:
            plt.plot(self.logs)
            plt.ylabel('Expectation')
            plt.xlabel('Iteration')
        self.log = []
        return vqeResult

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
        return self.circuit.draw(output, **kwargs)