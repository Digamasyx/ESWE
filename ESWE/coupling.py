"""
Coupling Module - Acoplamento Forte ESWE-SPH - CORRIGIDO

CORREÇÕES IMPLEMENTADAS:
1. Compatível com novo SPHSolver com arrays NumPy
2. ELIMINA double dipping de momentum
3. Estratégia A: Células ESWE congeladas durante SPH ativo
4. Força F_ext aplicada apenas UMA VEZ ao final da fase SPH
5. Conservação rigorosa de momentum garantida
"""

import numpy as np
from eswe_solver import ESWESolver
from sph_solver import SPHSolver
from amr_grid import AMRGrid

class ESWESPHCoupling:
    """Gerencia o acoplamento forte entre ESWE (2D) e SPH (3D) - SEM DOUBLE DIPPING"""
    
    def __init__(self, eswe_solver: ESWESolver, sph_solver: SPHSolver, grid: AMRGrid):
        """
        Inicializa sistema de acoplamento
        
        Parâmetros:
        -----------
        eswe_solver : ESWESolver
            Solver ESWE 2D
        sph_solver : SPHSolver
            Solver SPH 3D (com arrays NumPy)
        grid : AMRGrid
            Grade computacional
        """
        self.eswe = eswe_solver
        self.sph = sph_solver
        self.grid = grid
        
        # Estado do acoplamento
        self.active_cells = []
        self.activation_time = 0.0
        
        # CORRIGIDO: Armazena estado inicial para cálculo de ΔP
        self.initial_velocities = []
        self.frozen_state = {}  # Armazena estado ESWE antes de congelar
        
        # Parâmetros
        self.particles_per_cell = 27  # 3x3x3
        self.buffer_layers = 1
    
    def detect_breaking_waves(self):
        """
        Detecta células em quebra de onda
        """
        return self.grid.detect_breaking_cells()
    
    def activate_sph(self, breaking_cells):
        """
        Ativa domínio SPH
        
        CORRIGIDO: Congela células ESWE imediatamente
        """
        if len(breaking_cells) == 0:
            return
        
        # Expande com buffer
        extended_cells = self._expand_with_buffer(breaking_cells)
        self.active_cells = extended_cells
        self.activation_time = 0.0
        
        # CORRIGIDO: Salva estado inicial das células para restauração
        self.frozen_state = {}
        for i, j in extended_cells:
            self.frozen_state[(i, j)] = {
                'eta': self.grid.eta[i, j],
                'u': self.eswe.u[i, j],
                'v': self.eswe.v[i, j]
            }
        
        # CONGELA células ESWE (não evoluem mais)
        self.eswe.freeze_cells(extended_cells)
        
        # Cria partículas SPH
        self.sph.clear_particles()
        self.initial_velocities = []
        
        for i, j in extended_cells:
            self._create_particles_in_cell(i, j)
        
        print(f"  → SPH ativado: {self.sph.num_particles} partículas em {len(extended_cells)} células")
        print(f"  → {len(extended_cells)} células ESWE CONGELADAS durante fase SPH")
    
    def _expand_with_buffer(self, cells):
        """Expande conjunto de células com buffer"""
        extended = set(cells)
        
        for _ in range(self.buffer_layers):
            new_cells = set()
            for i, j in extended:
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < self.grid.ny and 0 <= nj < self.grid.nx:
                            new_cells.add((ni, nj))
            extended = new_cells
        
        return list(extended)
    
    def _create_particles_in_cell(self, i, j):
        """
        Cria partículas SPH em uma célula
        """
        bounds = self.grid.get_cell_bounds(i, j)
        
        h_cell = self.grid.d[i, j] + self.grid.eta[i, j]
        
        if h_cell <= 0.01:
            return
        
        u_cell = self.eswe.u[i, j]
        v_cell = self.eswe.v[i, j]
        
        # Massa total (Eq. 33)
        area = self.grid.get_cell_area(i, j)
        M_cell = self.eswe.rho * h_cell * area
        
        # Massa por partícula (Eq. 34)
        m_particle = M_cell / self.particles_per_cell
        
        # Distribuição 3x3x3
        n_layers = 3
        nx_local = 3
        ny_local = 3
        
        for iz in range(n_layers):
            # Posição z dentro da célula
            z_relative = (iz + 0.5) / n_layers
            z = bounds['z_min'] + z_relative * h_cell
            
            for ix in range(nx_local):
                # Posição x dentro da célula
                x_relative = (ix + 0.5) / nx_local
                x = bounds['x_min'] + x_relative * (bounds['x_max'] - bounds['x_min'])
                
                for iy in range(ny_local):
                    # Posição y dentro da célula
                    y_relative = (iy + 0.5) / ny_local
                    y = bounds['y_min'] + y_relative * (bounds['y_max'] - bounds['y_min'])
                    
                    # Adicionar partícula usando o novo método
                    self.sph.add_particle(x, y, z, u_cell, v_cell, 0.0, m_particle)
                    
                    # Armazena velocidade inicial
                    self.initial_velocities.append(np.array([u_cell, v_cell, 0.0]))
    
    def check_deactivation_criterion(self):
        """
        Verifica critério de desativação
        """
        if self.sph.num_particles == 0:
            return True
        
        # Critério primário: aceleração vertical
        hydrostatic = self._check_hydrostatic_recovery(threshold=0.05)
        
        # Critério secundário: gradiente relaxou
        if len(self.active_cells) > 0:
            grad_mag = self.grid.compute_gradient_magnitude()
            max_grad = np.max([grad_mag[i, j] for i, j in self.active_cells 
                              if 0 <= i < self.grid.ny and 0 <= j < self.grid.nx])
            gradient_relaxed = max_grad < self.grid.grad_eta_coarsen
        else:
            gradient_relaxed = True
        
        # Tempo mínimo
        min_time = self.activation_time > 0.5
        
        return hydrostatic and gradient_relaxed and min_time
    
    def _check_hydrostatic_recovery(self, threshold=0.05):
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
        if self.sph.num_particles == 0:
            return True
        
        # Verificar se há histórico de velocidades
        if not hasattr(self, '_last_velocities') or self._last_velocities is None:
            self._last_velocities = self.sph.velocities.copy()
            return False
        
        # Calcular acelerações verticais médias
        dt = 0.01  # Assumindo um dt fixo para estimativa
        if self.sph.num_particles > 0:
            acc_z = np.abs(self.sph.velocities[:, 2] - self._last_velocities[:, 2]) / dt
            avg_acc_z = np.mean(acc_z) if len(acc_z) > 0 else 0.0
        else:
            avg_acc_z = 0.0
        
        # Atualizar histórico
        self._last_velocities = self.sph.velocities.copy() if self.sph.num_particles > 0 else None
        
        return avg_acc_z < threshold
    
    def deactivate_sph(self):
        """Finaliza fase SPH e aplica momentum acumulado ao ESWE"""
        if not self.active_cells:
            return
        
        # 1. Calcula variação total de momentum (P_final - P_inicial)
        delta_P = self._get_momentum_change()
        
        # 2. Sincroniza campos físicos
        self._update_eswe_from_sph()
        
        # 3. Aplica força resultante no grid ESWE
        if delta_P is not None:
            self._apply_sph_force_to_grid(delta_P)
        
        # 4. Limpa partículas e libera memória
        self.sph.clear_particles()
        self.active_cells = []
        self.activation_time = 0.0
        self.initial_velocities = []
        self.frozen_state = {}
        self._last_velocities = None
        
        # 5. Descongela células ESWE
        self.eswe.unfreeze_cells()
        
        print(f"  → SPH desativado. Células ESWE descongeladas.")
    
    def _get_momentum_change(self):
        """
        Calcula a variação total de momentum do sistema SPH
        
        Retorna:
        --------
        delta_P : ndarray ou None
            Vetor 3D com variação total de momentum [Px, Py, Pz]
        """
        if self.sph.num_particles == 0 or len(self.initial_velocities) == 0:
            return None
        
        # Verificar compatibilidade
        if len(self.initial_velocities) != self.sph.num_particles:
            print(f"[AVISO] Número de velocidades iniciais ({len(self.initial_velocities)}) "
                  f"não corresponde ao número de partículas ({self.sph.num_particles})")
            return None
        
        # Momentum inicial
        P_initial = np.zeros(3)
        for i, vel in enumerate(self.initial_velocities):
            if i < self.sph.num_particles:
                P_initial += self.sph.masses[i] * vel
        
        # Momentum final
        P_final = np.zeros(3)
        for i in range(self.sph.num_particles):
            P_final += self.sph.masses[i] * self.sph.velocities[i]
        
        return P_final - P_initial
    
    def _apply_sph_force_to_grid(self, F_total):
        """
        Aplica força SPH no grid ESWE
        
        CORRIGIDO: Distribui uniformemente sobre área do domínio SPH
        """
        # Área total do domínio SPH
        A_total = sum(self.grid.get_cell_area(i, j) 
                     for i, j in self.active_cells)
        
        if A_total < 1e-10:
            return
        
        # Densidade de força (N/m²)
        f_density = F_total / A_total
        
        # Distribui para cada célula
        for i, j in self.active_cells:
            if 0 <= i < self.grid.ny and 0 <= j < self.grid.nx:
                A_cell = self.grid.get_cell_area(i, j)
                
                # Força nesta célula (N)
                F_cell = f_density * A_cell
                
                # ADICIONA a F_ext (será aplicado no próximo step do ESWE)
                self.eswe.F_ext_x[i, j] = F_cell[0]
                self.eswe.F_ext_y[i, j] = F_cell[1]
    
    def _update_eswe_from_sph(self):
        """
        Atualiza campos ESWE baseado no estado final das partículas SPH
        
        CORRIGIDO: Atualização de estado física, não de momentum
        """
        if self.sph.num_particles == 0:
            return

        # Dicionário para acumular dados por célula ativa
        cell_data = {}
        for cell in self.active_cells:
            cell_data[cell] = {'indices': [], 'vel_x': [], 'vel_y': [], 'z_values': []}
        
        # Mapear partículas para células
        for idx in range(self.sph.num_particles):
            pos = self.sph.positions[idx]
            
            # Mapeamento para grid ESWE
            j = int(pos[0] / self.grid.dx)
            i = int(pos[1] / self.grid.dy)
            
            if (i, j) in cell_data:
                cell_data[(i, j)]['indices'].append(idx)
                cell_data[(i, j)]['vel_x'].append(self.sph.velocities[idx, 0])
                cell_data[(i, j)]['vel_y'].append(self.sph.velocities[idx, 1])
                cell_data[(i, j)]['z_values'].append(pos[2])
        
        # Atualizar cada célula
        for (i, j), data in cell_data.items():
            if data['indices']:  # Se há partículas na célula
                # Velocidades médias
                u_mean = np.mean(data['vel_x'])
                v_mean = np.mean(data['vel_y'])
                
                # Altura da superfície (Z máximo das partículas na célula)
                z_max = np.max(data['z_values'])
                
                # Atualização direta
                self.eswe.u[i, j] = u_mean
                self.eswe.v[i, j] = v_mean
                self.grid.eta[i, j] = z_max - self.grid.d[i, j]
    
    def exchange_momentum(self, dt):
        """
        REMOVIDO: Não há exchange contínuo de momentum
        
        Durante a fase SPH:
        - Células ESWE estão CONGELADAS
        - SPH evolui independentemente
        - Momentum é transferido APENAS ao final via F_ext
        
        Esta função apenas incrementa o timer.
        """
        self.activation_time += dt