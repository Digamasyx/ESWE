"""
AMR Grid - CORRIGIDO

CORREÇÕES IMPLEMENTADAS:
1. AMR "conceitual" - marca células mas mantém grade uniforme
2. Usa dx/dy efetivos baseados no nível de refinamento
3. Compatível com np.gradient e fluxos explícitos
4. Preparado para futura implementação Quadtree

NOTA: Para AMR verdadeiro com células de tamanhos diferentes,
      seria necessário Quadtree + fluxos nas interfaces.
      Esta versão é um meio-termo pragmático.
"""

import numpy as np

class AMRGrid:
    """
    Grade computacional com refinamento adaptativo SIMPLIFICADO
    
    Mantém grade uniforme mas marca células refinadas para:
    - Ajustar CFL localmente
    - Guiar decisões de discretização
    - Preparar para futura implementação Quadtree
    """
    
    def __init__(self, Lx, Ly, nx, ny, h0):
        """
        Inicializa grade AMR
        
        Parâmetros:
        -----------
        Lx, Ly : float
            Dimensões do domínio (m)
        nx, ny : int
            Número de células na malha base
        h0 : float
            Profundidade de repouso (m)
        """
        self.Lx = Lx
        self.Ly = Ly
        self.nx = nx
        self.ny = ny
        self.h0 = h0
        
        # Espaçamento da malha base (UNIFORME)
        self.dx = Lx / nx
        self.dy = Ly / ny
        
        # Campos principais
        self.eta = np.zeros((ny, nx))  # Elevação (m)
        self.d = h0 * np.ones((ny, nx))  # Batimetria (m)
        
        # Níveis de refinamento CONCEITUAIS
        # 0 = base, 1 = refinado 1x, 2 = refinado 2x, etc
        # NOTA: Não subdivide células fisicamente, apenas marca
        self.refinement_level = np.zeros((ny, nx), dtype=int)
        self.max_refinement_level = 2  # Máximo permitido
        
        # Parâmetros de refinamento
        self.grad_eta_refine = 0.5      # Refina se ||∇η|| ≥ 0.5
        self.grad_eta_coarsen = 0.4     # Desrefina se ||∇η|| < 0.4
        self.grad_eta_sph = 0.577       # SPH se ||∇η|| ≥ tan(30°)
        
        # Estatísticas
        self.refinement_history = []
    
    def get_coordinates(self):
        """Retorna coordenadas da malha (sempre uniforme)"""
        x = np.linspace(0, self.Lx, self.nx)
        y = np.linspace(0, self.Ly, self.ny)
        X, Y = np.meshgrid(x, y)
        return X, Y
    
    def get_effective_spacing(self, i, j):
        """
        Retorna espaçamento efetivo considerando nível de refinamento
        
        NOTA: Em AMR conceitual, isso é usado apenas para CFL.
              A grade física permanece uniforme.
        """
        level = self.refinement_level[i, j]
        factor = 2**level
        
        dx_eff = self.dx / factor
        dy_eff = self.dy / factor
        
        return dx_eff, dy_eff
    
    def compute_gradient_magnitude(self):
        """
        Calcula ||∇η|| usando diferenças centrais (grade uniforme)
        """
        # Gradientes centrais (compatível com grade uniforme)
        deta_dx = np.gradient(self.eta, self.dx, axis=1)
        deta_dy = np.gradient(self.eta, self.dy, axis=0)
        
        grad_mag = np.sqrt(deta_dx**2 + deta_dy**2)
        
        return grad_mag
    
    def compute_strain_rate(self, u, v):
        """
        Calcula |S̄| para critério de turbulência
        """
        du_dx = np.gradient(u, self.dx, axis=1)
        du_dy = np.gradient(u, self.dy, axis=0)
        dv_dx = np.gradient(v, self.dx, axis=1)
        dv_dy = np.gradient(v, self.dy, axis=0)
        
        # Tensor de deformação
        S11 = du_dx
        S22 = dv_dy
        S12 = 0.5 * (du_dy + dv_dx)
        
        strain_mag = np.sqrt(2 * (S11**2 + S22**2 + S12**2))
        
        return strain_mag
    
    def refine_adaptive(self, u, v, theta_turb=1.0):
        """
        Aplica refinamento adaptativo CONCEITUAL
        
        Marca células mas não subdivide fisicamente.
        
        Parâmetros:
        -----------
        u, v : ndarray
            Campos de velocidade
        theta_turb : float
            Limiar para turbulência
        """
        # Critério 1: Gradiente geométrico
        grad_mag = self.compute_gradient_magnitude()
        
        # Máscara de refinamento
        refine_mask = grad_mag >= self.grad_eta_refine
        
        # Critério 3: Turbulência
        strain_mag = self.compute_strain_rate(u, v)
        turbulence_mask = strain_mag > theta_turb
        
        refine_mask = refine_mask | turbulence_mask
        
        # Máscara de desrefinamento (com histerese)
        coarsen_mask = (grad_mag < self.grad_eta_coarsen) & (strain_mag < theta_turb*0.5)
        
        # Aplica refinamento (incrementa nível)
        self.refinement_level[refine_mask] = np.minimum(
            self.refinement_level[refine_mask] + 1,
            self.max_refinement_level
        )
        
        # Aplica desrefinamento (decrementa nível)
        self.refinement_level[coarsen_mask] = np.maximum(
            self.refinement_level[coarsen_mask] - 1,
            0
        )
        
        # Suaviza transições (evita saltos bruscos de nível)
        self._smooth_refinement_levels()
        
        # Estatísticas
        n_refined = np.sum(self.refinement_level > 0)
        self.refinement_history.append(n_refined)
    
    def _smooth_refinement_levels(self):
        """
        Suaviza níveis de refinamento para evitar saltos > 1
        
        Regra: Células vizinhas não podem diferir por mais de 1 nível
        """
        max_iterations = 3
        
        for _ in range(max_iterations):
            changed = False
            new_levels = self.refinement_level.copy()
            
            for i in range(1, self.ny-1):
                for j in range(1, self.nx-1):
                    # Nível máximo dos vizinhos
                    neighbors = [
                        self.refinement_level[i-1, j],
                        self.refinement_level[i+1, j],
                        self.refinement_level[i, j-1],
                        self.refinement_level[i, j+1]
                    ]
                    max_neighbor = max(neighbors)
                    
                    # Se diferença > 1, incrementa
                    if max_neighbor - self.refinement_level[i, j] > 1:
                        new_levels[i, j] = max_neighbor - 1
                        changed = True
            
            self.refinement_level = new_levels
            
            if not changed:
                break
    
    def detect_breaking_cells(self):
        """
        Detecta células que violam hipótese hidrostática
        
        Retorna:
        --------
        breaking_cells : list of tuple
            Lista de índices (i, j)
        """
        grad_mag = self.compute_gradient_magnitude()
        
        # Critério SPH (Eq. 32)
        breaking_mask = grad_mag >= self.grad_eta_sph
        
        breaking_cells = list(zip(*np.where(breaking_mask)))
        
        return breaking_cells
    
    def set_bathymetry_slope(self, x_start, x_end, depth_start, depth_end):
        """
        Define batimetria com rampa (útil para praia)
        """
        X, Y = self.get_coordinates()
        
        slope = (depth_end - depth_start) / (x_end - x_start)
        
        mask = (X >= x_start) & (X <= x_end)
        self.d[mask] = depth_start + slope * (X[mask] - x_start)
        
        self.d[X < x_start] = depth_start
        self.d[X > x_end] = depth_end
    
    def set_bathymetry_gaussian_bump(self, x0, y0, amplitude, width):
        """
        Adiciona obstáculo gaussiano
        """
        X, Y = self.get_coordinates()
        
        r_squared = (X - x0)**2 + (Y - y0)**2
        bump = amplitude * np.exp(-r_squared / (2 * width**2))
        
        self.d += bump
    
    def get_cell_area(self, i, j):
        """
        Retorna área da célula
        
        NOTA: Em AMR conceitual, área física é sempre dx*dy
              Nível de refinamento é apenas marcador
        """
        return self.dx * self.dy
    
    def get_cell_bounds(self, i, j):
        """Retorna limites espaciais da célula"""
        x_min = j * self.dx
        x_max = (j + 1) * self.dx
        y_min = i * self.dy
        y_max = (i + 1) * self.dy
        
        z_min = -self.d[i, j]
        z_max = self.eta[i, j]
        
        return {
            'x_min': x_min, 'x_max': x_max,
            'y_min': y_min, 'y_max': y_max,
            'z_min': z_min, 'z_max': z_max
        }
    
    def get_refinement_statistics(self):
        """Retorna estatísticas de refinamento"""
        total_cells = self.nx * self.ny
        
        stats = {
            'total_cells': total_cells,
            'level_0': np.sum(self.refinement_level == 0),
            'level_1': np.sum(self.refinement_level == 1),
            'level_2': np.sum(self.refinement_level == 2),
            'max_level': np.max(self.refinement_level),
            'refined_fraction': np.sum(self.refinement_level > 0) / total_cells
        }
        
        return stats


class QuadtreeAMRGrid(AMRGrid):
    """
    STUB para futura implementação de Quadtree verdadeiro
    
    Para AMR real com células de tamanhos diferentes:
    - Estrutura de dados baseada em árvore
    - Cálculo de fluxo nas interfaces com hanging nodes
    - Interpolação conservativa
    - Balanço de fluxo rigoroso
    
    Referência: LeVeque (2002), "Finite Volume Methods for Hyperbolic Problems"
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("[INFO] QuadtreeAMRGrid é um stub - usando AMR conceitual")
        print("       Para AMR verdadeiro, implemente estrutura de árvore")
    
    # TODO: Implementar
    # - Subdivisão recursiva de células
    # - Travessia da árvore
    # - Cálculo de fluxo conservativo em interfaces
    # - Interpolação de estado entre níveis