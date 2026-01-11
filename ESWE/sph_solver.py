"""
Solver SPH 3D - OTIMIZADO COM NUMBA + UNIFORM GRID

CORREÇÕES IMPLEMENTADAS:
1. Vetorização com NumPy arrays
2. Compilação JIT com Numba para loops críticos
3. Complexidade reduzida de O(N^2) para O(N)
"""

import numpy as np

# Configuração do Numba
try:
    from numba import jit, prange, int32, float64
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    print("[ERRO CRÍTICO] Numba não encontrado. A performance será inviável.")
    # Fallback dummy
    def jit(*args, **kwargs):
        def decorator(func): return func
        return decorator
    prange = range

# ==============================================================================
# KERNELS NUMBA (Low Level)
# ==============================================================================

@jit(nopython=True, fastmath=True)
def get_grid_index(x, y, z, grid_min, cell_size, grid_dim_x, grid_dim_y, grid_dim_z):
    """Calcula o índice linear da célula 1D a partir das coordenadas 3D"""
    idx_x = int((x - grid_min[0]) / cell_size)
    idx_y = int((y - grid_min[1]) / cell_size)
    idx_z = int((z - grid_min[2]) / cell_size)
    
    # Clamp para evitar segfault nas bordas
    idx_x = max(0, min(idx_x, grid_dim_x - 1))
    idx_y = max(0, min(idx_y, grid_dim_y - 1))
    idx_z = max(0, min(idx_z, grid_dim_z - 1))
    
    return idx_x + idx_y * grid_dim_x + idx_z * grid_dim_x * grid_dim_y

@jit(nopython=True, fastmath=True)
def build_neighbor_grid(positions, N, h, grid_min, grid_max):
    """
    Constrói a Linked-List Cell Grid
    Retorna: head (início da lista por célula), next (próxima partícula), cell_size, dims
    """
    cell_size = h  # Célula do tamanho do raio de suavização
    
    # Dimensões da grade
    grid_dim_x = int(np.ceil((grid_max[0] - grid_min[0]) / cell_size))
    grid_dim_y = int(np.ceil((grid_max[1] - grid_min[1]) / cell_size))
    grid_dim_z = int(np.ceil((grid_max[2] - grid_min[2]) / cell_size))
    
    num_cells = grid_dim_x * grid_dim_y * grid_dim_z
    
    # Inicializa arrays da Linked List
    # head: armazena o índice da PRIMEIRA partícula em cada célula (-1 se vazio)
    head = np.full(num_cells, -1, dtype=np.int32)
    # next_particle: armazena o índice da PRÓXIMA partícula na mesma célula
    next_particle = np.full(N, -1, dtype=np.int32)
    
    for i in range(N):
        cell_idx = get_grid_index(positions[i,0], positions[i,1], positions[i,2], 
                                  grid_min, cell_size, grid_dim_x, grid_dim_y, grid_dim_z)
        
        # Lógica de inserção na cabeça da lista (LIFO)
        next_particle[i] = head[cell_idx]
        head[cell_idx] = i
        
    return head, next_particle, cell_size, (grid_dim_x, grid_dim_y, grid_dim_z)

@jit(nopython=True, parallel=True, fastmath=True)
def compute_forces_optimized(N, positions, velocities, masses, densities, pressures, 
                             h, head, next_particle, grid_min, cell_size, grid_dims,
                             visc_alpha, c_s, g_const):
    """
    Calcula Densidade e Forças em loop único otimizado com busca espacial
    """
    # Constantes do Kernel
    poly6_const = 315.0 / (64.0 * np.pi * h**9)
    spiky_grad_const = -45.0 / (np.pi * h**6)
    h2 = h * h
    
    forces = np.zeros((N, 3), dtype=np.float64)
    new_densities = np.zeros(N, dtype=np.float64)
    
    gx, gy, gz = grid_dims
    
    # Loop paralelo sobre todas as partículas
    for i in prange(N):
        pos_i = positions[i]
        
        # Passo 1: Calcular Densidade
        rho = 0.0
        
        # Encontra célula atual
        idx_x = int((pos_i[0] - grid_min[0]) / cell_size)
        idx_y = int((pos_i[1] - grid_min[1]) / cell_size)
        idx_z = int((pos_i[2] - grid_min[2]) / cell_size)
        
        # Busca nas células vizinhas (3x3x3)
        for ka in range(max(0, idx_x-1), min(gx, idx_x+2)):
            for kb in range(max(0, idx_y-1), min(gy, idx_y+2)):
                for kc in range(max(0, idx_z-1), min(gz, idx_z+2)):
                    
                    cell_idx = ka + kb * gx + kc * gx * gy
                    j = head[cell_idx]
                    
                    # Percorre a lista encadeada na célula vizinha
                    while j != -1:
                        # Distância quadrada
                        dx = pos_i[0] - positions[j, 0]
                        dy = pos_i[1] - positions[j, 1]
                        dz = pos_i[2] - positions[j, 2]
                        r2 = dx*dx + dy*dy + dz*dz
                        
                        if r2 < h2:
                            # Kernel Poly6 para densidade
                            w = poly6_const * (h2 - r2)**3
                            rho += masses[j] * w
                        
                        j = next_particle[j]
        
        new_densities[i] = max(rho, 1.0) # Evita divisão por zero
        
    # Sincronização necessária para cálculo de pressão
    # Como pressures depende de densities, calculamos localmente ou passamos array pronto
    # Aqui vamos recalcular a pressão localmente para evitar outro loop global
    
    # Loop de Forças (requer densidade já calculada)
    for i in prange(N):
        pos_i = positions[i]
        rho_i = new_densities[i]
        pres_i = pressures[i] # Pressão vem do passo anterior ou atualizada externamente
        
        f_x = 0.0
        f_y = 0.0
        f_z = 0.0
        
        idx_x = int((pos_i[0] - grid_min[0]) / cell_size)
        idx_y = int((pos_i[1] - grid_min[1]) / cell_size)
        idx_z = int((pos_i[2] - grid_min[2]) / cell_size)

        for ka in range(max(0, idx_x-1), min(gx, idx_x+2)):
            for kb in range(max(0, idx_y-1), min(gy, idx_y+2)):
                for kc in range(max(0, idx_z-1), min(gz, idx_z+2)):
                    cell_idx = ka + kb * gx + kc * gx * gy
                    j = head[cell_idx]
                    
                    while j != -1:
                        if i != j:
                            dx = pos_i[0] - positions[j, 0]
                            dy = pos_i[1] - positions[j, 1]
                            dz = pos_i[2] - positions[j, 2]
                            r2 = dx*dx + dy*dy + dz*dz
                            
                            if r2 < h2 and r2 > 1e-12:
                                r = np.sqrt(r2)
                                rho_j = new_densities[j]
                                pres_j = pressures[j]
                                
                                # Kernel Spiky Gradiente
                                grad_w = spiky_grad_const * (h - r)**2 / r
                                
                                # Termo de Pressão Simétrico
                                f_press = -masses[j] * (pres_i/(rho_i**2) + pres_j/(rho_j**2)) * grad_w
                                
                                # Termo de Viscosidade Artificial
                                # Monaghan (1992) type
                                dvx = velocities[i,0] - velocities[j,0]
                                dvy = velocities[i,1] - velocities[j,1]
                                dvz = velocities[i,2] - velocities[j,2]
                                v_dot_r = dvx*dx + dvy*dy + dvz*dz
                                
                                visc = 0.0
                                if v_dot_r < 0:
                                    mu = (h * v_dot_r) / (r2 + 0.01*h2)
                                    visc = -masses[j] * (-visc_alpha * c_s * mu) / (0.5 * (rho_i + rho_j)) * grad_w
                                
                                f_total = f_press + visc
                                f_x += f_total * dx
                                f_y += f_total * dy
                                f_z += f_total * dz
                                
                        j = next_particle[j]
        
        # Adiciona gravidade
        forces[i, 0] = f_x
        forces[i, 1] = f_y
        forces[i, 2] = f_z - g_const
        
    return forces, new_densities

# ==============================================================================
# CLASSE SPH SOLVER
# ==============================================================================

class SPHSolver:
    """Solver SPH 3D Otimizado"""

    def __init__(self, g=9.81, rho0=1000.0, h_sph=0.5):
        self.g = g
        self.rho0 = rho0
        self.h = h_sph
        
        # Arrays NumPy (não mais lista de dicts para performance)
        self.num_particles = 0
        self.positions = np.zeros((0, 3))
        self.velocities = np.zeros((0, 3))
        self.masses = np.zeros(0)
        self.densities = np.zeros(0)
        self.pressures = np.zeros(0)
        
        # Parâmetros
        self.c_s = 30.0 # Speed of sound (precisa ser ~10x max vel)
        self.alpha = 0.1
        self.gamma = 7.0

    def add_particle(self, x, y, z, u, v, w, mass):
        """Adiciona partícula aos arrays"""
        if self.num_particles == 0:
            self.positions = np.array([[x, y, z]])
            self.velocities = np.array([[u, v, w]])
            self.masses = np.array([mass])
            self.densities = np.array([self.rho0])
            self.pressures = np.array([0.0])
        else:
            self.positions = np.vstack([self.positions, [x, y, z]])
            self.velocities = np.vstack([self.velocities, [u, v, w]])
            self.masses = np.append(self.masses, mass)
            self.densities = np.append(self.densities, self.rho0)
            self.pressures = np.append(self.pressures, 0.0)
        
        self.num_particles += 1

    def step(self, dt):
        """Avança simulação no tempo"""
        if self.num_particles == 0:
            return

        # 1. Preparar Grid de Pesquisa Espacial
        # Define limites dinâmicos da grade com uma margem
        grid_min = np.min(self.positions, axis=0) - 2*self.h
        grid_max = np.max(self.positions, axis=0) + 2*self.h
        
        head, next_particle, cell_size, dims = build_neighbor_grid(
            self.positions, self.num_particles, self.h, grid_min, grid_max
        )

        # 2. Atualizar Equação de Estado (Pressão) antes das forças
        # P = B * ((rho/rho0)^gamma - 1)
        B = (self.rho0 * self.c_s**2) / self.gamma
        self.pressures = B * ((self.densities / self.rho0)**self.gamma - 1.0)
        
        # 3. Calcular Forças e Densidades (Kernel Pesado)
        forces, new_densities = compute_forces_optimized(
            self.num_particles, self.positions, self.velocities, self.masses,
            self.densities, self.pressures, self.h,
            head, next_particle, grid_min, cell_size, dims,
            self.alpha, self.c_s, self.g
        )
        
        self.densities = new_densities

        # 4. Integração Temporal (Euler Simplético)
        self.velocities += dt * forces # F = ma, mas forças já estão divididas pela massa no kernel se rho for usado
        # Nota: O kernel SPH calcula f_press/rho, que é aceleração. 
        # Se forces for realmente força, divida por massa. 
        # No kernel acima, calculei como aceleração (termos divididos por rho*rho).
        
        self.positions += dt * self.velocities

    def get_particles_dict(self):
        """Compatibilidade com o visualizador anterior"""
        particles = []
        for i in range(self.num_particles):
            particles.append({
                'position': self.positions[i],
                'velocity': self.velocities[i],
                'mass': self.masses[i],
                'density': self.densities[i],
                'pressure': self.pressures[i]
            })
        return particles
    def clear_particles(self):
        """Remove todas as partículas"""
        self.num_particles = 0
        self.positions = np.zeros((0, 3))
        self.velocities = np.zeros((0, 3))
        self.masses = np.zeros(0)
        self.densities = np.zeros(0)
        self.pressures = np.zeros(0)

    def check_hydrostatic_recovery(self, threshold=0.05):
        """
        Verifica se o escoamento retornou à condição hidrostática
    
        Parâmetros:
        -----------
        threshold : float
            Limiar de aceleração vertical para considerar hidrostático
        
        Retorna:
        --------
        bool: True se hidrostático, False caso contrário
        """
        if self.num_particles == 0:
            return True
    
        # Calcular acelerações médias
        if hasattr(self, 'last_velocities'):
            dt = 0.01  # Supondo um dt fixo para estimativa
            acc_z = np.abs(self.velocities[:, 2] - self.last_velocities[:, 2]) / dt
            avg_acc_z = np.mean(acc_z)
            return avg_acc_z < threshold
        else:
            # Primeira iteração, não há histórico
            self.last_velocities = self.velocities.copy()
            return False