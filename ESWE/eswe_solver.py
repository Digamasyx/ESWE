import numpy as np
from scipy.ndimage import laplace

class ESWESolver:
	"""Solver para as equações de Águas Rasas Extendidas"""
	def __init__(self, grid, g=9.81, rho=1000.0):
		"""
		Inicializa o solver ESWE

		Parametros:
		------------
		grid: AMRGrid
			Grade Computacional
		g: float
			Aceleração Gravitacional (m/s²)
		rho: float
			Densidade da Água (kg/m³)
		"""
		self.grid = grid
		self.g = g
		self.rho = rho

		# Campos de velocidades (m/s)
		self.u = np.zeros_like(grid.eta) # Velocidade horizontal x
		self.v = np.zeros_like(grid.eta) # Velocidade horizontal y

		# Parametros Físicos
		self.C_f = 0.003 # Coef. de atrito de fundo
		self.C_s = 0.12 # Coef. de Smagorinsky
		self.gamma = 0.072 # Tensão Superficial (N/m)

		# Parametros Numéricos
		self.C_CFL = 0.5 # Número de Courant

		# Acoplamento SPH
		self.F_ext_x = np.zeros_like(grid.eta)
		self.F_ext_y = np.zeros_like(grid.eta)
	def compute_cfl_timestep(self):
		"""
		Calcula o passo de tempo adaptativo pela condição CFL
		
		Retorna:
		----------
		dt: float
			Passo de tempo máximo permitido (s)
		"""
		h = self.grid.d + self.grid.eta

		# Velocidade máxima de propagação
		c_wave = np.sqrt(self.g * h)
		v_max = np.sqrt(self.u**2 + self.v**2) + c_wave

		# Evita divisão por zero
		v_max = np.maximum(v_max, 1e-6)

		# Condição CFL
		dt = self.C_CFL * np.min([self.grid.dx, self.grid.dy]) / np.max(v_max)

		return min(dt, 0.01) # Limita dt máximo
	def compute_smagorinsky_viscosity(self):
		"""
		Calcula viscosidade turbulenta SGS usando modelo de Smagorinsky

		Retorna:
		---------
		nu_sgs: ndarray
			Campo de viscosidade SGS (m²/s)
		"""
		# Largura do filtro
		Delta = np.sqrt(self.grid.dx * self.grid.dy)

		# Derivadas para tensor de deformação
		du_dx = np.gradient(self.u, self.grid.dx, axis=1)
		du_dy = np.gradient(self.u, self.grid.dy, axis=0)
		dv_dx = np.gradient(self.v, self.grid.dx, axis=1)
		dv_dy = np.gradient(self.v, self.grid.dy, axis=0)

		# Módulo do tensor de taxa de deformação
		S_mag = np.sqrt(2 * (du_dx**2 + dv_dy**2 + 0.5*(du_dy + dv_dx) ** 2))

		# Viscosidade SGS
		nu_sgs = (self.C_s * Delta) **2 * S_mag

		return nu_sgs

	def compute_bed_friction(self):
		"""
		Calcula dissipação por atrito de fundo

		Retorna:
		---------
		D_fundo_x, D_fundo_y: tuple of ndarray
			Componentes de arrasto de fundo
		"""
		h = self.grid.d + self.grid.eta

		# Magnitude da velocidade
		u_mag = np.sqrt(self.u**2 + self.v**2)

		# Evita divisão por zero
		h_safe = np.maximum(h, 0.01)

		# Arrasto Quadrático
		D_fundo_x = -self.C_f * self.u * u_mag / h_safe
		D_fundo_y = -self.C_f * self.v * u_mag / h_safe

		return D_fundo_x, D_fundo_y

	def compute_advection(self):
		"""
		Calcula o termo de advecção

		Retorna:
		adv_x, adv_y: tuple of ndarray
			Componentes de advecção
		"""
		h = self.grid.d + self.grid.eta

		# Produtos tensoriais
		hu2 = h * self.u * self.u
		hv2 = h * self.v * self.v
		huv = h * self.v * self.u

		# Divergencias 
		adv_x = (np.gradient(hu2, self.grid.dx, axis=1) +
				 np.gradient(huv, self.grid.dy, axis=0))
		adv_y = (np.gradient(huv, self.grid.dx, axis=1) +
				 np.gradient(hv2, self.grid.dy, axis=0))

		return adv_x, adv_y

	def compute_pressure_gradient(self):
		"""
		Calcula gradiente de pressão gravitacional

		Retorna:
		----------
		G_x, G_y: tuple of ndarray
			Componentes do gradiente de pressão
		"""
		h = self.grid.d + self.grid.eta

		# Gradiente de eta
		deta_dx = np.gradient(self.grid.eta, self.grid.dx, axis=1)
		deta_dy = np.gradient(self.grid.eta, self.grid.dx, axis=0)

		# Força de pressão
		G_x = -self.g * h * deta_dx
		G_y = -self.g * h * deta_dy

		return G_x, G_y

	def compute_turbulent_diffusion(self):
		"""
		Calcula difusão turbulenta

		Retorna:
		----------
		D_turb_x, D_turb_y: tuple of ndarray
			Componentes da difusão turbulenta
		"""
		nu_sgs = self.compute_smagorinsky_viscosity()

		# Laplacianos
		lap_u = laplace(self.u) / (self.grid.dx**2)
		lap_v = laplace(self.v) / (self.grid.dy**2)

		# Difusão
		D_turb_x = nu_sgs * lap_u
		D_turb_y = nu_sgs * lap_v

		return D_turb_x, D_turb_y

	def step(self, dt):
		"""
		Executa um passo de tempo usando Runge-Kutta de 2ª ordem

		Parametros:
		dt: float
			Passo de tempo (s)
		"""

		# Estado inicial
		eta0 = self.grid.eta.copy()
		u0 = self.u.copy()
		v0 = self.v.copy()

		# Euler explicito
		self._rk_stage(dt)

		eta1 = self.grid.eta.copy()
		u1 = self.u.copy()
		v1 = self.v.copy()

		# Restaura estado original
		self.grid.eta = eta0
		self.u = u0
		self.v = v0
		        
		# Ponto médio
		self._rk_stage(dt)
        
        # Média RK2
		self.grid.eta = 0.5 * (eta0 + self.grid.eta)
		self.u = 0.5 * (u0 + self.u)
		self.v = 0.5 * (v0 + self.v)

		# Aplica condições de contorno
		self._apply_boundary_conditions()

	def _rk_stage(self, dt):
		"""Executa um estágio do método Runge-Kutta"""

		h = self.grid.d + self.grid.eta

		div_hu = (np.gradient(h * self.u, self.grid.dx, axis=1) +
                  np.gradient(h * self.v, self.grid.dy, axis=0))
		deta_dt = -div_hu

		adv_x, adv_y = self.compute_advection()

		G_x, G_y = self.compute_pressure_gradient()

		D_turb_x, D_turb_y = self.compute_turbulent_diffusion() 
		
		D_fundo_x, D_fundo_y = self.compute_bed_friction()

		# Taxa de variação do momentum
		d_hu_dt_x = -adv_x + G_x + D_turb_x + D_fundo_x + self.F_ext_x
		d_hu_dt_y = -adv_y + G_y + D_turb_y + D_fundo_y + self.F_ext_y

		# Converter para taxa de velocidade
		h_safe = np.maximum(h, 0.01)
		du_dt = d_hu_dt_x / h_safe - self.u * deta_dt / h_safe
		dv_dt = d_hu_dt_y / h_safe - self.v * deta_dt / h_safe

		# Integra no tempo
		self.grid.eta += dt * deta_dt
		self.u += dt * du_dt
		self.v += dt * dv_dt

	def _apply_boundary_conditions(self):
		"""Aplica condições de contorno"""

		# Condições u * n = 0 nas bordas
		self.u[0, :] = 0
		self.u[-1, :] = 0
		self.v[:, 0] = 0
		self.v[:, -1] = 0

		# Superficie livre nas bordas
		self.grid.eta[0, :] = self.grid.eta[1, :]
		self.grid.eta[-1, :] = self.grid.eta[-2, :]
		self.grid.eta[:, 0] = self.grid.eta[:, 1]
		self.grid.eta[:, -1] = self.grid.eta[:, -2]
