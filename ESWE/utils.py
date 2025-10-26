import numpy as np

# Constantes do sistema
GRAVIDADE = 9.81  # m/s^2
DENSIDADE_FLUIDO = 1000  # kg/m^3 
T_ACTIVE = 0.05 # s - Tempo ativo da simulação SPH 3D (para cálculo da Força Média)

def calculate_gradient(data_field: np.ndarray, cell_size: float):
    """
    Simula o cálculo do gradiente ∇ de um campo escalar (e.g., altura da superfície livre, eta)
    usando Diferenças Centrais (aproximação simples de 2ª ordem).

    Fórmula (Discretizada):
    ∇f(i) ≈ (f(i+1) - f(i-1)) / (2 * dx)
    
    Args:
        data_field: O campo escalar 1D (ex: [eta_0, eta_1, ..., eta_N]).
        cell_size: O tamanho da célula da grade (dx ou dy).

    Returns:
        Um array NumPy com o gradiente (apenas na direção X, para simplificação).
    """
    gradient = np.zeroslike(data_field, dtype=float)

    if data_field.size > 2:
        gradient[1:-1] = (data_field[2:] - data_field[:-2]) / (2 * cell_size)

    gradient[0] = (data_field[1] - data_field[0]) / cell_size
    gradient[-1] = (data_field[-1] - data_field[-2]) / cell_size

    return gradient

def calculate_divergence(u_field: np.ndarray, h_field: np.ndarray, cell_size: float):
    flux = u_field * h_field
    divergence = calculate_gradient(flux, cell_size)
    return divergence

def calculate_norm(vector_field: np.ndarray):
    return np.abs(vector_field)
