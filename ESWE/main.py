
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
import time

from eswe_solver import ESWESolver
from sph_solver import SPHSolver
from amr_grid import AMRGrid
from coupling import ESWESPHCoupling

class HybridSimulation:
    """Classe principal para gerenciar a simulação híbrida ESWE-SPH"""
    
    def __init__(self, Lx=50.0, Ly=50.0, nx=100, ny=100, h0=2.0):
        """
        Inicializa a simulação
        
        Parâmetros:
        -----------
        Lx, Ly : float
            Dimensões do domínio em metros
        nx, ny : int
            Número de células na malha base
        h0 : float
            Profundidade de repouso em metros
        """
        # Parâmetros físicos
        self.g = 9.81  # m/s²
        self.rho = 1000.0  # kg/m³
        self.h0 = h0
        
        # Domínio computacional
        self.Lx = Lx
        self.Ly = Ly
        self.nx = nx
        self.ny = ny
        
        # Grade AMR
        self.grid = AMRGrid(Lx, Ly, nx, ny, h0)
        
        # Solver ESWE (2D)
        self.eswe = ESWESolver(self.grid, self.g, self.rho)
        
        # Solver SPH (3D) - inicialmente vazio
        self.sph = SPHSolver(self.g, self.rho)
        
        # Sistema de acoplamento
        self.coupling = ESWESPHCoupling(self.eswe, self.sph, self.grid)
        
        # Estatísticas
        self.time = 0.0
        self.step = 0
        self.sph_active = False
        
        # Histórico para análise
        self.energy_history = []
        self.sph_history = []
        
    def add_initial_disturbance(self, x0, y0, amplitude=0.5, width=3.0):
        """Adiciona perturbação inicial gaussiana"""
        X, Y = self.grid.get_coordinates()
        r_squared = (X - x0)**2 + (Y - y0)**2
        self.grid.eta += amplitude * np.exp(-r_squared / (2 * width**2))
        
    def add_wave_source(self, x0, y0, amplitude=0.2, frequency=1.0):
        """Adiciona fonte de onda periódica"""
        omega = 2 * np.pi * frequency
        disturbance = amplitude * np.sin(omega * self.time)
        
        X, Y = self.grid.get_coordinates()
        r_squared = (X - x0)**2 + (Y - y0)**2
        width = 2.0
        
        self.grid.eta += disturbance * np.exp(-r_squared / (2 * width**2))
        
    def step_simulation(self, dt=None):
        """Executa um passo de tempo da simulação"""
        
        # 1. Calcula dt adaptativo se não fornecido
        if dt is None:
            dt = self.eswe.compute_cfl_timestep()
        
        # 2. Refina malha adaptativamente
        self.grid.refine_adaptative(self.eswe.u, self.eswe.v)
        
        # 3. Detecta células que violam hipótese hidrostática
        breaking_cells = self.coupling.detect_breaking_waves()
        
        # 4. Se há quebra de onda, ativa SPH
        if len(breaking_cells) > 0 and not self.sph_active:
            print(f"[t={self.time:.3f}s] Ativando SPH em {len(breaking_cells)} células")
            self.coupling.activate_sph(breaking_cells)
            self.sph_active = True
        
        # 5. Integra ESWE
        self.eswe.step(dt)
        
        # 6. Se SPH ativo, integra partículas
        if self.sph_active:
            self.sph.step(dt)
            
            # Verifica critério de desativação
            if self.coupling.check_deactivation_criterion():
                print(f"[t={self.time:.3f}s] Desativando SPH")
                self.coupling.deactivate_sph()
                self.sph_active = False
        
        # 7. Troca momentum entre SPH e ESWE
        if self.sph_active:
            self.coupling.exchange_momentum(dt)
        
        # Atualiza tempo
        self.time += dt
        self.step += 1
        
        # Estatísticas
        self.energy_history.append(self.compute_total_energy())
        self.sph_history.append(len(self.sph.particles) if self.sph_active else 0)
        
        return dt
    
    def compute_total_energy(self):
        """Calcula energia total do sistema"""
        # Energia potencial
        E_pot = 0.5 * self.g * np.sum(self.grid.eta**2) * self.grid.dx * self.grid.dy
        
        # Energia cinética
        h = self.grid.d + self.grid.eta
        E_kin = 0.5 * self.rho * np.sum(h * (self.eswe.u**2 + self.eswe.v**2)) * self.grid.dx * self.grid.dy
        
        return E_pot + E_kin
    
    def run_realtime(self, duration=30.0, fps=30):
        """Executa simulação com visualização em tempo real"""
        
        # Configuração da visualização
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # Subplots
        ax_profile = fig.add_subplot(gs[0, :])  # Corte vertical
        ax_velocity = fig.add_subplot(gs[1, :])  # Campo de velocidades
        ax_top = fig.add_subplot(gs[2, 0:2])  # Vista de cima
        ax_sph = fig.add_subplot(gs[2, 2], projection='3d')  # SPH 3D
        
        # Dados para o corte vertical (no centro em y)
        slice_y = self.ny // 2
        
        def init():
            return []
        
        def update(frame):
            # Executa múltiplos passos por frame para tempo real
            steps_per_frame = 3
            for _ in range(steps_per_frame):
                if self.time >= duration:
                    break
                dt = self.step_simulation()
            
            # Limpa axes
            ax_profile.clear()
            ax_velocity.clear()
            ax_top.clear()
            ax_sph.clear()
            
            # 1. CORTE VERTICAL (altura vs x no centro do domínio)
            x_coords = np.linspace(0, self.Lx, self.nx)
            eta_slice = self.grid.eta[slice_y, :]
            h_slice = self.grid.d[slice_y, :] + eta_slice
            
            ax_profile.fill_between(x_coords, 0, -self.grid.d[slice_y, :], 
                                   color='saddlebrown', alpha=0.6, label='Fundo')
            ax_profile.fill_between(x_coords, -self.grid.d[slice_y, :], h_slice - self.h0,
                                   color='dodgerblue', alpha=0.7, label='Água')
            ax_profile.plot(x_coords, h_slice - self.h0, 'b-', linewidth=2, label='Superfície')
            
            # Marca células com SPH ativo
            if self.sph_active:
                for cell in self.coupling.active_cells:
                    ix, iy = cell
                    if iy == slice_y:
                        x_cell = ix * self.grid.dx
                        ax_profile.axvspan(x_cell, x_cell + self.grid.dx, 
                                         color='red', alpha=0.3)
            
            ax_profile.set_xlabel('x (m)', fontsize=10)
            ax_profile.set_ylabel('η (m)', fontsize=10)
            ax_profile.set_title(f'Corte Vertical (y={slice_y*self.grid.dy:.1f}m) - t={self.time:.2f}s', 
                               fontsize=11, fontweight='bold')
            ax_profile.legend(loc='upper right', fontsize=8)
            ax_profile.grid(True, alpha=0.3)
            ax_profile.set_ylim([-self.h0-0.5, 2.0])
            
            # 2. CAMPO DE VELOCIDADES (quiver plot no corte)
            x_vel = x_coords[::5]
            u_slice = self.eswe.u[slice_y, ::5]
            v_magnitude = np.abs(self.eswe.v[slice_y, ::5])
            
            colors = plt.cm.viridis(v_magnitude / (np.max(v_magnitude) + 1e-6))
            
            for i, (x, u_val, color) in enumerate(zip(x_vel, u_slice, colors)):
                ax_velocity.arrow(x, 0, u_val*2, 0, head_width=0.1, 
                                head_length=0.2, fc=color, ec=color, alpha=0.7)
            
            ax_velocity.axhline(y=0, color='k', linestyle='-', linewidth=1)
            ax_velocity.set_xlabel('x (m)', fontsize=10)
            ax_velocity.set_ylabel('u (m/s)', fontsize=10)
            ax_velocity.set_title('Campo de Velocidades Horizontal', fontsize=11, fontweight='bold')
            ax_velocity.grid(True, alpha=0.3)
            ax_velocity.set_ylim([-2, 2])
            
            # 3. VISTA DE CIMA (contorno de eta)
            X, Y = self.grid.get_coordinates()
            contour = ax_top.contourf(X, Y, self.grid.eta, levels=20, cmap='RdBu_r')
            ax_top.contour(X, Y, self.grid.eta, levels=10, colors='k', 
                          linewidths=0.5, alpha=0.3)
            
            # Marca células SPH
            if self.sph_active:
                for cell in self.coupling.active_cells:
                    ix, iy = cell
                    rect = plt.Rectangle((ix*self.grid.dx, iy*self.grid.dy), 
                                        self.grid.dx, self.grid.dy,
                                        fill=False, edgecolor='red', linewidth=2)
                    ax_top.add_patch(rect)
            
            ax_top.set_xlabel('x (m)', fontsize=10)
            ax_top.set_ylabel('y (m)', fontsize=10)
            ax_top.set_title('Vista Superior - Elevação η(x,y)', fontsize=11, fontweight='bold')
            ax_top.set_aspect('equal')
            plt.colorbar(contour, ax=ax_top, label='η (m)')
            
            # 4. VISUALIZAÇÃO SPH 3D
            if self.sph_active and len(self.sph.particles) > 0:
                positions = np.array([p['position'] for p in self.sph.particles])
                velocities = np.array([p['velocity'] for p in self.sph.particles])
                v_mag = np.linalg.norm(velocities, axis=1)
                
                scatter = ax_sph.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                                        c=v_mag, cmap='hot', s=10, alpha=0.6)
                
                ax_sph.set_xlabel('x (m)', fontsize=8)
                ax_sph.set_ylabel('y (m)', fontsize=8)
                ax_sph.set_zlabel('z (m)', fontsize=8)
                ax_sph.set_title(f'SPH 3D ({len(self.sph.particles)} partículas)', 
                               fontsize=10, fontweight='bold')
                ax_sph.set_xlim([0, self.Lx])
                ax_sph.set_ylim([0, self.Ly])
                ax_sph.set_zlim([0, self.h0 + 2])
                
                plt.colorbar(scatter, ax=ax_sph, label='|v| (m/s)', shrink=0.5)
            else:
                ax_sph.text(0.5, 0.5, 0.5, 'SPH Inativo', 
                          horizontalalignment='center',
                          transform=ax_sph.transAxes, fontsize=12)
                ax_sph.set_xlim([0, 1])
                ax_sph.set_ylim([0, 1])
                ax_sph.set_zlim([0, 1])
            
            # Info
            info_text = f'Passo: {self.step} | SPH: {"ATIVO" if self.sph_active else "inativo"} | E={self.energy_history[-1]:.2e} J'
            fig.suptitle(info_text, fontsize=12, fontweight='bold')
            
            return []
        
        # Animação
        frames = int(duration * fps / 3)  # 3 steps per frame
        anim = FuncAnimation(fig, update, init_func=init, 
                           frames=frames, interval=1000/fps, blit=False)
        
        plt.tight_layout()
        plt.show()
        
        return anim


def main():
    """Função principal de execução"""
    print("="*60)
    print("SIMULAÇÃO HÍBRIDA ESWE-SPH")
    print("Sistema de Conservação de Momentum 2D/3D")
    print("="*60)
    
    # Cria simulação
    sim = HybridSimulation(Lx=50.0, Ly=50.0, nx=80, ny=80, h0=2.0)
    
    # Adiciona perturbação inicial (onda gaussiana)
    print("\n[INFO] Adicionando perturbação inicial...")
    sim.add_initial_disturbance(x0=25.0, y0=25.0, amplitude=1.0, width=3.0)
    
    # Executa com visualização em tempo real
    print("[INFO] Iniciando simulação em tempo real...")
    print("[INFO] Pressione Ctrl+C para interromper\n")
    
    try:
        sim.run_realtime(duration=30.0, fps=30)
    except KeyboardInterrupt:
        print("\n[INFO] Simulação interrompida pelo usuário")
    
    print("\n" + "="*60)
    print(f"Simulação concluída: {sim.step} passos, t={sim.time:.2f}s")
    print("="*60)


if __name__ == "__main__":
    main()