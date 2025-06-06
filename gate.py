import numpy as np

class Qubit:
    """Класс для представления состояния кубита"""
    def __init__(self, alpha=1, beta=0):
        if alpha == 0 and beta == 0:
            raise ValueError("Нулевой вектор недопустим для кубита!")
        self.state = np.array([alpha, beta], dtype=complex)
        self.normalize()
    
    def normalize(self):
        """Нормализация состояния"""
        norm = np.linalg.norm(self.state)
        self.state /= norm
    
    def apply_gate(self, gate):
        """Применение однокубитного гейта"""
        self.state = gate @ self.state
    

# Гейты Паули
X = np.array([
    [0, 1], 
    [1, 0]
])

Y = np.array([
    [0, -1j], 
    [1j,  0]
])

Z = np.array([
    [1,  0], 
    [0, -1]
])

class TwoQubit:
    def __init__(self, state=None, qubit1=None, qubit2=None):
        if qubit1 is not None and qubit2 is not None:
            self.state = np.kron(qubit1.state, qubit2.state)
        elif state is not None:
            self.state = state
        else:
            self.state = np.array([1, 0, 0, 0], dtype=complex)  # |00⟩
        self.normalize()

    def normalize(self):
        norm = np.linalg.norm(self.state)
        self.state /= norm

    def apply_cnot(self):
        cnot_matrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ])
        self.state = cnot_matrix @ self.state



# Пример использования
qx = Qubit(1,0)  
print("Состояние до X:", qx.state)
qx.apply_gate(X)   # Применяем гейт X
print("Состояние после X:", qx.state)

qy = Qubit(1, 0)  
print("Состояние до Y:", qy.state)
qy.apply_gate(Y)   # Применяем гейт Y
print("Состояние после Y:", qy.state)

qz = Qubit(0, 1)  
print("Состояние до Z:", qz.state)
qz.apply_gate(Z)   # Применяем гейт Z
print("Состояние после Z:", qz.state)

q1 = Qubit(0, 1)  # |0>
q2 = Qubit(1, 0)  # |0>
sys = TwoQubit(qubit1=q1, qubit2=q2)
sys.apply_cnot()
print("Состояние двух кубитов после CNOT:", sys.state)