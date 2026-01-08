import numpy as np

class AMRGrid:
	"""Grade computacional com refinamento adaptativo (AMR)"""
	
	def __init__(self, Lx, Ly, nx, ny, h0):
		"""
		Inicializa grade AMR

		Parametros:
		------------
		Lx, Ly: float
			Dimensões do dominio (m)
		nx, ny: float
			Número de células na malha base
		h0: float
			Profundidade de repouso (m)
		"""
		self.Lx = Lx
		self.Ly = Ly
		self.nx = nx
		self.ny = ny
		self.h0 = h0

		# Espaçamento da malha base
		self.dx = Lx / nx
		self.dy = Ly / ny

		# Campos principais
		self.eta = np.zeros((ny, nx)) # Elevação da superficie livre (m)
		self.d = h0 * np.ones((nx, ny)) # Batimetria (m)

		# Niveis de refinamento (0 = base, 1 = refinado 1x, etc)
		self.refinement_level = np.zeros((nx, ny), dtype=int)

		# Parametros de refinamento
		self.grad_eta_refine = 0.5
		self.grad_eta_coarsen = 0.4
		self.grad_eta_sph = 0.577 # tan(30º)

	def get_coordinates(self):
		"""
		Retorna coordenadas da malha

		Retorna:
		---------
		X, Y: ndarray
			Matrizes de coordenadas
		"""
		x = np.linspace(0, self.Lx, self.nx)
		y = np.linspace(0, self.Ly, self.ny)

		X,Y = np.meshgrid(x, y)

		return X, Y

	def compute_gradient_magnitude(self):
		"""
		Calcula magnitude do gradiente de eta

		Retorna:
		---------
		grad_mag: ndarray
			||∇η|| em cada célula
		"""

		# Gradientes centrais
		deta_dx = np.gradient(self.eta, self.dx, axis=1)
		deta_dy = np.gradient(self.eta, self.dy, axis=0)

		# Magnitude
		grad_mag = np.sqrt(deta_dx**2 + deta_dy**2)

		return grad_mag

	def compute_temporal_variation(self, eta_old, dt):
		"""
		Calcula variação temporal de eta

		Parametros:
		------------
		eta_old: ndarray
			Elevação no passo anterior
		dt: float
			Passo de tempo (s)

		Retorna:
		---------
		temporal_var: ndarray
			|∂η/∂t| / √(gh) em cada célula
		"""

		h = self.d + self.eta
		c_wave = np.sqrt(9.81 * h)

		deta_dt = np.abs(self.eta - eta_old) / dt

		# Normalizado pela velocidade de onda
		temporal_var = deta_dt / (c_wave + 1e-6)

		return temporal_var

	def compute_strain_rate(self, u, v):
		"""
        Calcula taxa de deformação
        
        Parâmetros:
        ------------
        u, v : ndarray
            Componentes de velocidade horizontal
            
        Retorna:
        ---------
        strain_mag : ndarray
            |S̄| em cada célula
        """
		# Derivadas das velocidades
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

	def refine_adaptative(self, u, v, theta_dyn=0.2, theta_turb=1.0):
		"""
		Aplica refinamento adaptativo baseado em múltiplos critérios

		Parametros:
		-----------
		u, v: ndarray
			Campos de velocidade
		theta_dyn: float
			Limiar para critério dinâmico
		theta_turb: float
			Limiar para critério de turbulência
		"""
		# 1º Critério: Gradiente geometrico
		grad_mag = self.compute_gradient_magnitude()

		refine_mask = grad_mag >= self.grad_eta_refine
		coarsen_mask = grad_mag < self.grad_eta_coarsen

		# 3º Critério: Turbulência/cisalhamento
		strain_mag = self.compute_strain_rate(u, v)
		turbulence_mask = strain_mag > theta_turb

		refine_mask = refine_mask | turbulence_mask

		# Aplica refinamento (modo simplificado)
		self.refinement_level[refine_mask] = np.minimum(
			self.refinement_level[refine_mask] + 1, 2	
		)

		# Aplica desrefinamento
		self.refinement_level[coarsen_mask] = np.maximum(
			self.refinement_level[coarsen_mask] - 1, 0	
		)

	def detect_breaking_cells(self):
		"""
		Detecta células que violam a hipótese hidrostática

		Retorna:
		---------
		breaking_cells: list of tuple
			Lista de indices (i, j) das células em quebra
		"""
		grad_mag = self.compute_gradient_magnitude()

		# Critério de transição SPH
		breaking_mask = grad_mag >= self.grad_eta_sph

		# Converte para lista de indices
		breaking_cells = list(zip(*np.where(breaking_mask)))

		return breaking_cells
	
	def set_bathymetry_slope(self, x_start, x_end, depth_start, depth_end):
		"""
		Define a batimatria da rampa (útil em casos de praia)

		Parametros:
		-----------
		x_start, x_end: float
			Posições inicial e final da rampa (m)
		depth_start, depth_end: float
			Profundidades inicial e final da rampa (m)
		"""	
		X, Y = self.get_coordinates()

		# Rampa linear
		slope = (depth_end - depth_start) / (x_end - x_start)

		mask = (X >= x_start) & (X <= x_end)
		self.d[mask] = depth_start + slope * (X[mask] - x_start)

		# Antes da rampa
		self.d[X < x_start] = depth_start

		# Depois da rampa
		self.d[X > x_end] = depth_end

	def set_bathymetry_gaussian_bump(self, x0, y0, amplitude, width):
		"""
		Adiciona obstaculo gaussiano à batimetria

		Parametros:
		-----------
		x0, y0: float
			Centro do obstaculo (m)
		amplitude: float
			Altura do obstaculo (m, positivo = eleva fundo)
		width: float
			Largura caracteristica (m)
		"""
		X, Y = self.get_coordinates()

		r_squared = (X - x0)**2 + (Y - y0)**2
		bump = amplitude * np.sqrt(-r_squared / (2 * width**2))

		self.d += bump

	def get_cell_area(self, i, j):
		"""
		Retorna àrea de uma célula (considerando refinamento)

		Parametros:
		-----------
		i, j: int
			Índices da célula

		Retorna:
		---------
		area: float
			Área da célula (m²)
		"""
		level = self.refinement_level[i, j]
		factor = 2**level

		dx_cell = self.dx / factor
		dy_cell = self.dy / factor

		return dx_cell * dy_cell

	def get_cell_bounds(self, i, j):
		"""
		Retorna os limites espaciais de uma célula

		Parametros:
		-----------
		i, j: int
			Índices da célula

		Retorna:
		---------
		bounds: dict
			Dicionário com 'x_min', 'x_max', 'y_min', 'y_max', 'z_min', 'z_max'
		"""
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