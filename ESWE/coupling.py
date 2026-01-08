import numpy as np
from eswe_solver import ESWESolver
from sph_solver import SPHSolver
from amr_grid import AMRGrid

class ESWESPHCoupling:
	"""Gerencia o acoplamento forte entre ESWE (2D) e SPH (3D)"""

	def __init__(self, eswe_solver: ESWESolver, sph_solver: SPHSolver, grid: AMRGrid):
		"""
		Inicializa sistema de acoplamento

		Parametros:
		-----------
		eswe_solver: ESWESolver
			Solver ESWE 2D
		sph_solver: SPHSolver
			Solver SPH 3D
		grid: AMRGrid
			Grade computacional
		"""
		self.eswe = eswe_solver
		self.sph = sph_solver
		self.grid = grid

		# Estado do acoplamento
		self.active_cells = [] # Celulas com SPH ativo
		self.activation_time = 0
		self.initial_velocities = [] # Para cálculo de dP

		# Parametros
		self.particles_per_cell = 27 # 3x3x3 por célula
		self.buffer_layers = 1 # Camadas de ghost particles

	def detect_breaking_waves(self):
		"""
        Detecta células em quebra de onda
        
        Retorna:
        --------
        breaking_cells : list of tuple
            Lista de índices (i, j) das células em quebra
        """
		return self.grid.detect_breaking_cells()
	def activate_sph(self, breaking_cells):
		"""
		Ativa dominio SPH nas celulas especificadas
		
		Parametros:
		-----------
		breaking_cells: list of tuple
			Células onde ativar SPH
		"""
		if len(breaking_cells) == 0:
			return

		# Expande dominio com buffer
		extended_cells = self._expand_with_buffer(breaking_cells)

		self.active_cells = extended_cells
		self.activation_time = 0.0
		self.initial_velocities = []

		# Cria particulas SPH
		self.sph.clear_particles()

		for i,j in extended_cells:
			self._create_particles_in_cell(i, j)

		print(f"  → SPH ativado: {len(self.sph.particles)} partículas em {len(extended_cells)} células")

	def _expand_with_buffer(self, cells):
		"""Expande conjunto de celulas com camadas de buffer"""
		extended = set(cells)

		for _ in range(self.buffer_layers):
			new_cells = set()
			for i, j in extended:
				# Adiciona vizinhos
				for di in [-1, 0, 1]:
					for dj in [-1, 0, 1]:
						ni, nj = i + di, j + dj
						if 0 <= ni < self.grid.ny and 0 <= nj < self.grid.nx:
							new_cells.add((ni, nj))
			extended = new_cells
		return list(extended)

	def _create_particles_in_cell(self, i, j):
		"""
		Cria particulas SPH em uma célula

		Parametros:
		-----------
		i, j: int
			Índices da celula
		"""
		# Limites da célula
		bounds = self.grid.get_cell_bounds(i, j)

		# Profundidade total
		h_cell = self.grid.d[i, j] + self.grid.eta[i, j]

		if h_cell <= 0.01: # Célula seca
			return


		# Velocidade da célula ESWE
		u_cell = self.eswe.u[i, j]
		v_cell = self.eswe.v[i, j]

		# Massa total da célula
		area = self.grid.get_cell_area(i, j)
		M_cell = self.eswe.rho * h_cell * area

		# Massa por particula
		m_particle = M_cell / self.particles_per_cell

		# Distribuição espacial (3x3x3)
		n_layers = 3
		nx_local = 3
		ny_local = 3

		for iz in range(n_layers):
			z = bounds['z_min'] + (iz + 0.5) * h_cell / n_layers

			for ix in range(nx_local):
				x = bounds['x_min'] + (ix + 0.5) * (bounds['x_max'] - bounds['x_min']) / nx_local

				for iy in range(ny_local):
					y = bounds['y_min'] + (iy + 0.5) * (bounds['y_max'] - bounds['y_min']) / ny_local

					# Posição da particula
					position = np.array([x, y, z])

					# Velocidade inicial - componente vertical zero
					velocity = np.array([u_cell, v_cell, 0.0])
					
					# Adiciona particula
					self.sph.add_particle(position, velocity, m_particle)

					# Armazena velocidade inicial para cálculo de dP
					self.initial_velocities.append(velocity.copy())

	def check_deactivation_criterion(self):
		"""
		Verifica se SPH pode ser desativado

		retorna:
		---------
		should_deactivate: bool
			True se criterios de desativação forem satisfeitos
		"""
		if len(self.sph.particles) == 0:
			return True

		# Criterio primario - aceleração vertical
		hydrostatic = self.sph.check_hydrostatic_recovery(0.05) # 5% de g

		# Criterio secundario - histerese do gradiente
		grad_mag = self.grid.compute_gradient_magnitude()
		max_grad = np.max([grad_mag[i, j] for i, j in self.active_cells if 0 <= i < self.grid.ny and 0 <= j < self.grid.nx])
		gradient_relaxed = max_grad < self.grid.grad_eta_coarsen

		# Tempo minimo de ativação (evita oscilações)
		min_time = self.activation_time > 0.5

		return hydrostatic and gradient_relaxed and min_time

	def deactivate_sph(self):
		"""Desativa SPH e reintegra estado no ESWE"""
		if len(self.sph.particles) == 0:
			return

		# Calcula variação de momentum horizontal
		delta_P = self.sph.get_momentum_change(self.initial_velocities)

		# Tempo ativo
		T_active = self.activation_time

		if T_active > 0:
			# Força media
			F_total = delta_P / T_active

			# Distribui para celulas ativas
			self._distribute_sph_force(F_total)

		# Limpa SPH
		self.sph.clear_particles()
		self.active_cells = []
		self.initial_velocities = []
		self.activation_time = 0.0
	def _distribute_sph_force(self, F_total):
		"""
		Distribui força SPH para grade ESWE

		Parametros:
		-----------
		F_total: ndarray
			Força total horizontal [Fx, Fy] (N)
		"""
		# Área total do dominio SPH
		A_total = sum(self.grid.get_cell_area(i, j) for i ,j in self.active_cells)

		if A_total < 1e-6: # Tolerancia
			return

		# Densidade de força por u.a (unidade de area)
		f_density = F_total / A_total

		# Distribui para cada célula
		for i, j in self.active_cells:
			A_cell = self.grid.get_cell_area(i, j)
			
			# Força na celula
			F_cell = f_density * A_cell

			# Injeta em F_ext
			self.eswe.F_ext_x[i, j] += F_cell[0]
			self.eswe.F_ext_y[i, j] += F_cell[1]

	def exchange_momentum(self, dt):
		"""
		Troca continua de momentum durante fase SPH ativa

		Parametros:
		-----------
		dt: float
			Passo de tempo (s)
		"""
		self.activation_time += dt

		# Atualiza campos ESWE baseado em posição média das particulas
		# (implementação simplificada - poderia ser mais sofisticada)
		if len(self.sph.particles) == 0:
			return

		# Agrupa particulas por celula
		cell_particles = {cell: [] for cell in self.active_cells}

		for particle in self.sph.particles:
			pos = particle['position']

			# Encontra celula
			i = int(pos[1] / self.grid.dy)
			j = int(pos[0] / self.grid.dx)

			if (i, j) in cell_particles:
				cell_particles[(i, j)].append(particle)

		# Atualiza velocidades ESWE baseado em particulas
		for (i, j), particles in cell_particles.items():
			if len(particles) == 0:
				continue

			# Velocidade media horizontal das particulas
			u_mean = np.mean([p['velocity'][0] for p in particles])
			v_mean = np.mean([p['velocity'][1] for p in particles])

			# Atualiza ESWE (com blend suave)
			alpha = 0.3 # Fator de blend
			self.eswe.u[i, j] = (1 - alpha) * self.eswe.u[i, j] + alpha * u_mean
			self.eswe.v[i, j] = (1 - alpha) * self.eswe.v[i, j] + alpha * v_mean

			# Atualiza superficie livre baseado em distribuição vertical
			z_max = max(p['position'][2] for p in particles)
			eta_sph = z_max

			self.grid.eta[i, j] = (1 - alpha) * self.grid.eta[i, j] + alpha * eta_sph
