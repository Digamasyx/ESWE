from tkinter import SEL
import numpy as np
from utils import GRAVIDADE, DENSIDADE_FLUIDO, T_ACTIVE, calculate_gradient, calculate_divergence, calculate_norm

class ESWE2D_STATE:
    """
    Representa o estado do sistema 2D ESWE em um passo de tempo.

    Para simplificação e foco no mecanismo híbrido, a grade é representada como 1D.
    - O eixo X representa a direção da propagação da onda.
    """
    def __init__(self, num_cells: int, initial_depth: float, cell_size: float):
        self.num_cells = num_cells
        self.cell_size = cell_size # Tamanho da célula da grade (dx)

        # Altura da Superfície Livre (Free-surface height) - η
        # É a altura acima do nível de repouso (datum)
        self.eta = np.zeros(num_cells, dtype=float)

        # Velocidade Horizontal (Horizontal velocity) - u
        # Aqui, u é um vetor 1D (velocidade na direção X)
        self.u = np.zeros(num_cells, dtype=float)

        # Altura da Coluna de Água (Water column height) - h
        # h = H + η, onde H é a profundidade do leito (bathymetry)
        self.H = initial_depth * np.ones(num_cells, dtype=float)
        self.h = self.H + self.eta

    def set_initial_wave(self, amplitude: float, center_cell: int):
        """Inicializa um pulso de onda simples no centro da grade."""
        self.eta[center_cell] = amplitude
        self.h = self.H + self.eta
        print(f"--- Estado Inicial ESWE: Onda de amplitude {amplitude} na célula {center_cell} ---")

class ESWE2D_Solver:

    def __init__(self, state: ESWE2D_STATE, dt: float):
        self.state = state
        self.dt = dt

    def calculate_momentum_change(self, force_ext: np.ndarray) -> np.ndarray:
        """
        Calcula a mudança na velocidade horizontal (Δu/Δt).

        A equação de momentum do ESWE é:
        ∂u/∂t = - (u ⋅ ∇) u - g ∇η + (1 / (ρh)) F_ext

        Onde:
        - g ∇η: Termo de Pressão (ou Gradiente de Pressão, relacionado à superfície livre).
        - (u ⋅ ∇) u: Termo de Advecção (não implementado completamente aqui, será mockado).
        - (1 / (ρh)) F_ext: Termo Híbrido, responsável por injetar o impulso 3D.

        Args:
            force_ext: O vetor F_ext (Densidade de Força Externa) mapeado do SPH.

        Returns:
            O campo de aceleração (∂u/∂t).
        """
        eta = self.state.eta
        h = self.state.h
        u = self.state.u
        dx = self.state.cell_size

        # 1. Termo de Advecção (u ⋅ ∇) u
        # Simplificação: Em 1D, (u ⋅ ∇)u é aproximadamente u * (∂u/∂x)
        # Assumindo 0 para focar na Força Externa, ou um mock simples:
        # du_dx = calculate_gradient(u, dx)
        # advection_term = u * du_dx
        advection_term = np.zeros_like(u)  # Mock simplificado

        # 2. Termo de Pressão (g ∇η)
        grad_eta = calculate_gradient(eta, dx)
        pressure_term = GRAVIDADE * grad_eta

        # 3. Termo Híbrido de Injeção de Momentum (F_ext / (ρh))
        # O denominador é a massa por área, essencial para conservação.
        # Evita divisão por zero:
        rho_h = DENSIDADE_FLUIDO * h
        rho_h[rho_h <= 1e-6] = 1e-6  # Limite inferior para a estabilidade

        hybrid_term = force_ext / rho_h

        # Aceleração (du/dt)
        du_dt = -pressure_term + hybrid_term - advection_term

        return du_dt