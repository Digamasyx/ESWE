"""
Solver ESWE

CORREÇÕES IMPLEMENTADAS:
1. Esquema upwind de 1ª ordem para advecção (estável)
2. Limitadores de fluxo TVD (Minmod) para 2ª ordem
3. Cálculo explícito de fluxos nas interfaces
4. Compatível com AMR simplificado
"""

import numpy as np
from scipy.ndimage import laplace
from amr_grid import AMRGrid

class ESWESolver:
	"""Solver para as equacoes de Aguas Rasas Extendidas"""
	def __init__(self, grid: AMRGrid, g=9.81, rho=1000.0):
		"""
		Inicializa o solver ESWE

		Parametros:
		------------
		grid: AMRGrid
			Grade Computacional
		g: float
			Aceleracao Gravitacional (m/s²)
		rho: float
			Densidade da Agua (kg/m³)
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
		self.flux_limiter = 'minmod'  # 'minmod', 'superbee', 'vanleer', ou 'none'

		# Acoplamento SPH
		self.F_ext_x = np.zeros_like(grid.eta)
		self.F_ext_y = np.zeros_like(grid.eta)

		# Células onde SPH está ativo (congeladas)
		self.frozen_cells = set()

		# Runge-Kutta
		self._eta0_rk = None
		self._u0_rk = None
		self._v0_rk = None

	def compute_cfl_timestep(self):
		"""
		Calcula o passo de tempo adaptativo pela condição CFL
		
		Retorna:
		----------
		dt: float
			Passo de tempo máximo permitido (s)
		"""
		h = self.grid.d + self.grid.eta
		h = np.maximum(h, 0.01)

		# Velocidade máxima de propagação
		c_wave = np.sqrt(self.g * h)
		v_max = np.sqrt(self.u**2 + self.v**2) + c_wave

		# Considera o refinamento da malha
		dx_min = self.grid.dx / (2**np.max(self.grid.refinement_level))
		dy_min = self.grid.dy / (2**np.max(self.grid.refinement_level))

		# Condição CFL
		dt = self.C_CFL * min(dx_min, dy_min) / (np.max(v_max) + 1e-6)

		return min(dt, 0.01) # Limita dt máximo

	def minmod(self, a, b):
		"""Limitador de fluxo Minmod (TVD)"""
		return np.where(a * b > 0, np.where(np.abs(a) < np.abs(b), a, b), 0.0)

	def superbee(self, a, b):
		"""Limitador de fluxo Superbee (TVD)"""
		s1 = self.minmod(a, 2*b)
		s2 = self.minmod(2*a, b)
		return np.where(np.abs(s1) > np.abs(s2), s1, s2)
	def vanleer(self, a, b):
		"""Limitador de fluxo Van Leer (TVD)"""
		return np.where(a * b > 0, 2 * a * b / (a + b + 1e-10), 0.0)

	def compute_limited_slopes(self, phi):
		"""
		Calcula slopes limitados para reconstrução de 2ª ordem (MUSCL)
		
		Parâmetros:
		-----------
		phi : ndarray
			Campo escalar (η, u, ou v)
			
		Retorna:
		--------
		slope_x, slope_y : tuple of ndarray
			Slopes limitados nas direções x e y
		"""
		# Diferenças foward e backward
		dphi_foward_x = np.diff(phi, axis=1, append=phi[:, -1:])
		dphi_backward_x = np.diff(phi, axis=1, append=phi[:, :1])

		dphi_foward_y = np.diff(phi, axis=0, append=phi[-1:, :])
		dphi_backward_y = np.diff(phi, axis=0, append=phi[:1, :])

		# Aplica limitador
		match self.flux_limiter:
			case 'minmod':
				slope_x = self.minmod(dphi_foward_x, dphi_backward_x)
				slope_y = self.minmod(dphi_foward_y, dphi_backward_y)
			case 'superbee':
				slope_x = self.superbee(dphi_foward_x, dphi_backward_x)
				slope_y = self.superbee(dphi_foward_y, dphi_backward_y)
			case 'vanleer':
				slope_x = self.vanleer(dphi_foward_x, dphi_backward_x)
				slope_y = self.vanleer(dphi_foward_y, dphi_backward_y)
			case _: # Case default ('none')
				slope_x = np.zeros_like(phi)
				slope_y = np.zeros_like(phi)

		return slope_x, slope_y

	def compute_flux_x(self, h, u, v):
		"""
		Calcula fluxos na direção X usando reconstrução MUSCL (2ª Ordem)
		"""
		# 1. Calcular Slopes (Limitadores) para reconstrução
		# Nota: slope_u_x tem o mesmo shape de u
		slope_h_x, _ = self.compute_limited_slopes(h)
		slope_u_x, _ = self.compute_limited_slopes(u)
		slope_v_x, _ = self.compute_limited_slopes(v)

		# 2. Reconstrução dos estados nas interfaces (i + 1/2)
		# Estado Esquerdo (L): vem da célula i, extrapola meia célula à frente
		# Indices [:, :-1] representam a célula i
		h_L = h[:, :-1] + 0.5 * slope_h_x[:, :-1]
		u_L = u[:, :-1] + 0.5 * slope_u_x[:, :-1]
		v_L = v[:, :-1] + 0.5 * slope_v_x[:, :-1]

		# Estado Direito (R): vem da célula i+1, extrapola meia célula para trás
		# Indices [:, 1:] representam a célula i+1
		h_R = h[:, 1:] - 0.5 * slope_h_x[:, 1:]
		u_R = u[:, 1:] - 0.5 * slope_u_x[:, 1:]
		v_R = v[:, 1:] - 0.5 * slope_v_x[:, 1:]

		# 3. Solver de Riemann (HLL ou Upwind baseado na velocidade média)
		# Velocidade de transporte na interface
		u_avg = 0.5 * (u_L + u_R)
		
		# Upwind: Se u > 0, o fluxo vem da esquerda (L). Se u < 0, vem da direita (R).
		# Fluxo de Massa: F = h * u
		flux_mass = np.where(u_avg > 0, h_L * u_L, h_R * u_R)
		
		# Fluxo de Momentum X: F = h * u^2 + 0.5 * g * h^2
		# Obs: Mantendo sua abordagem original, a pressão (0.5gh^2) é tratada no termo fonte,
		# então aqui transportamos apenas o momento convectivo (h*u*u).
		flux_mom_x = np.where(u_avg > 0, h_L * u_L * u_L, h_R * u_R * u_R)
		
		# Fluxo de Momentum Y: F = h * u * v
		flux_mom_y = np.where(u_avg > 0, h_L * u_L * v_L, h_R * u_R * v_R)

		return flux_mass, flux_mom_x, flux_mom_y

	def compute_flux_y(self, h, u, v):
		"""
		Calcula fluxos na direção Y usando reconstrução MUSCL (2ª Ordem)
		"""
		# 1. Calcular Slopes na direção Y
		_, slope_h_y = self.compute_limited_slopes(h)
		_, slope_u_y = self.compute_limited_slopes(u)
		_, slope_v_y = self.compute_limited_slopes(v)

		# 2. Reconstrução (Preciso ter cuidado com os eixos: axis=0 é Y)
		# Estado "Abaixo" (L na lógica 1D): célula j
		h_L = h[:-1, :] + 0.5 * slope_h_y[:-1, :]
		u_L = u[:-1, :] + 0.5 * slope_u_y[:-1, :]
		v_L = v[:-1, :] + 0.5 * slope_v_y[:-1, :]

		# Estado "Acima" (R na lógica 1D): célula j+1
		h_R = h[1:, :] - 0.5 * slope_h_y[1:, :]
		u_R = u[1:, :] - 0.5 * slope_u_y[1:, :]
		v_R = v[1:, :] - 0.5 * slope_v_y[1:, :]

		# 3. Solver de Riemann
		v_avg = 0.5 * (v_L + v_R)

		flux_mass = np.where(v_avg > 0, h_L * v_L, h_R * v_R)
		flux_mom_x = np.where(v_avg > 0, h_L * v_L * u_L, h_R * v_R * u_R)
		flux_mom_y = np.where(v_avg > 0, h_L * v_L * v_L, h_R * v_R * v_R)

		return flux_mass, flux_mom_x, flux_mom_y
	
	def compute_advection(self):
		"""
		Calcula termo de advecção usando esquema upwind (CORRIGIDO)
	
		Retorna divergência dos fluxos: ∇·F
		"""
		h = self.grid.d + self.grid.eta
		h = np.maximum(h, 0.01)
	
		# Fluxos nas interfaces
		hu_x, hu2_x, huv_x = self.compute_flux_x(h, self.u, self.v)
		hv_y, huv_y, hv2_y = self.compute_flux_y(h, self.u, self.v)
	
		# CORREÇÃO: Ajustar as dimensões dos fluxos para calcular divergência
		# Precisamos garantir que todos os arrays tenham a mesma dimensão
	
		# Para fluxos x: duplicar a primeira e última coluna para ter dimensão (ny, nx)
		hu_x_full = np.zeros_like(h)
		hu_x_full[:, :-1] = hu_x  # Fluxos entre células
		hu_x_full[:, -1] = hu_x[:, -1]  # Extrapolar última interface
	
		hu2_x_full = np.zeros_like(h)
		hu2_x_full[:, :-1] = hu2_x
		hu2_x_full[:, -1] = hu2_x[:, -1]
	
		# Para fluxos y: duplicar a primeira e última linha para ter dimensão (ny, nx)
		hv_y_full = np.zeros_like(h)
		hv_y_full[:-1, :] = hv_y  # Fluxos entre células
		hv_y_full[-1, :] = hv_y[-1, :]  # Extrapolar última interface
	
		hv2_y_full = np.zeros_like(h)
		hv2_y_full[:-1, :] = hv2_y
		hv2_y_full[-1, :] = hv2_y[-1, :]
	
		# Para os fluxos cruzados, fazer o mesmo
		huv_x_full = np.zeros_like(h)
		huv_x_full[:, :-1] = huv_x
		huv_x_full[:, -1] = huv_x[:, -1]
	
		huv_y_full = np.zeros_like(h)
		huv_y_full[:-1, :] = huv_y
		huv_y_full[-1, :] = huv_y[-1, :]
	
		# Agora calcular divergência com arrays de mesma dimensão
		div_mass = (np.diff(hu_x_full, axis=1, prepend=hu_x_full[:, :1]) / self.grid.dx +
					np.diff(hv_y_full, axis=0, prepend=hv_y_full[:1, :]) / self.grid.dy)
	
		div_mom_x = (np.diff(hu2_x_full, axis=1, prepend=hu2_x_full[:, :1]) / self.grid.dx +
					 np.diff(huv_y_full, axis=0, prepend=huv_y_full[:1, :]) / self.grid.dy)
	
		div_mom_y = (np.diff(huv_x_full, axis=1, prepend=huv_x_full[:, :1]) / self.grid.dx +
					 np.diff(hv2_y_full, axis=0, prepend=hv2_y_full[:1, :]) / self.grid.dy)
	
		return div_mass, div_mom_x, div_mom_y
	
	def compute_smagorinsky_viscosity(self):
		"""Calcula viscosidade turbulenta SGS"""
		Delta = np.sqrt(self.grid.dx * self.grid.dy)
		
		# Derivadas com diferenças centrais (ok para difusão)
		du_dx = np.gradient(self.u, self.grid.dx, axis=1)
		du_dy = np.gradient(self.u, self.grid.dy, axis=0)
		dv_dx = np.gradient(self.v, self.grid.dx, axis=1)
		dv_dy = np.gradient(self.v, self.grid.dy, axis=0)
		
		# Módulo do tensor de deformação
		S_mag = np.sqrt(2 * (du_dx**2 + dv_dy**2 + 0.5*(du_dy + dv_dx)**2))
		
		nu_sgs = (self.C_s * Delta)**2 * S_mag
		
		return nu_sgs
	
	def compute_bed_friction(self):
		"""Calcula dissipação por atrito de fundo"""
		h = self.grid.d + self.grid.eta
		h_safe = np.maximum(h, 0.01)
		
		u_mag = np.sqrt(self.u**2 + self.v**2)
		
		D_fundo_x = -self.C_f * self.u * u_mag / h_safe
		D_fundo_y = -self.C_f * self.v * u_mag / h_safe
		
		return D_fundo_x, D_fundo_y
	
	def compute_pressure_gradient(self):
		"""Calcula gradiente de pressão -gh∇η"""
		h = self.grid.d + self.grid.eta
		
		deta_dx = np.gradient(self.grid.eta, self.grid.dx, axis=1)
		deta_dy = np.gradient(self.grid.eta, self.grid.dy, axis=0)
		
		G_x = -self.g * h * deta_dx
		G_y = -self.g * h * deta_dy
		
		return G_x, G_y
	
	def compute_turbulent_diffusion(self):
		"""Calcula difusão turbulenta ∇·(ν_SGS∇u)"""
		nu_sgs = self.compute_smagorinsky_viscosity()
		
		lap_u = laplace(self.u) / (self.grid.dx**2)
		lap_v = laplace(self.v) / (self.grid.dy**2)
		
		D_turb_x = nu_sgs * lap_u
		D_turb_y = nu_sgs * lap_v
		
		return D_turb_x, D_turb_y
	
	def freeze_cells(self, cells):
		"""Congela células onde SPH está ativo"""
		self.frozen_cells = set(cells)
	
	def unfreeze_cells(self):
		"""Descongela todas as células"""
		self.frozen_cells = set()
	
	def step(self, dt):
		"""
		Executa um passo de tempo usando Runge-Kutta de 2ª ordem
		"""
		# Estado inicial
		eta0 = self.grid.eta.copy()
		u0 = self.u.copy()
		v0 = self.v.copy()
		
		# === ESTÁGIO 1 ===
		self._rk_stage(dt)
		
		eta1 = self.grid.eta.copy()
		u1 = self.u.copy()
		v1 = self.v.copy()
		
		# Restaura estado inicial
		self.grid.eta = eta0
		self.u = u0
		self.v = v0

		# Armazena estado inicial para congelamento
		self._eta0_rk = eta0
		self._u0_rk = u0
		self._v0_rk = v0
		
		# === ESTÁGIO 2 ===
		self._rk_stage(dt)
		
		# Média RK2
		self.grid.eta = 0.5 * (eta0 + self.grid.eta)
		self.u = 0.5 * (u0 + self.u)
		self.v = 0.5 * (v0 + self.v)
		
		# Aplica condições de contorno
		self._apply_boundary_conditions()
		
		# Limpa forças externas após aplicação
		self.F_ext_x.fill(0.0)
		self.F_ext_y.fill(0.0)
	
	def _rk_stage(self, dt):
		"""Executa um estágio do método Runge-Kutta"""
		
		h = self.grid.d + self.grid.eta
		h = np.maximum(h, 0.01)
		
		# === EQUAÇÃO DE CONTINUIDADE ===
		div_mass, div_mom_x, div_mom_y = self.compute_advection()
		deta_dt = -div_mass
		
		# === EQUAÇÃO DE MOMENTUM ===
		# Gradiente de pressão
		G_x, G_y = self.compute_pressure_gradient()
		
		# Difusão turbulenta
		D_turb_x, D_turb_y = self.compute_turbulent_diffusion()
		
		# Atrito de fundo
		D_fundo_x, D_fundo_y = self.compute_bed_friction()
		
		# Taxa de variação do momentum
		d_hu_dt_x = -div_mom_x + G_x + D_turb_x + D_fundo_x + self.F_ext_x
		d_hu_dt_y = -div_mom_y + G_y + D_turb_y + D_fundo_y + self.F_ext_y
		
		# Converte para taxa de velocidade
		du_dt = d_hu_dt_x / h - self.u * deta_dt / h
		dv_dt = d_hu_dt_y / h - self.v * deta_dt / h
		
		# Integra no tempo
		self.grid.eta += dt * deta_dt
		self.u += dt * du_dt
		self.v += dt * dv_dt
		
		# CONGELA células com SPH ativo (CORRIGIDO)
		for (i, j) in self.frozen_cells:
			if 0 <= i < self.grid.ny and 0 <= j < self.grid.nx:
				self.grid.eta[i, j] = self._eta0_rk[i, j]  # Mantém estado original
				self.u[i, j] = self._u0_rk[i, j]
				self.v[i, j] = self._v0_rk[i, j]
	
	def _apply_boundary_conditions(self):
		"""Aplica condições de contorno"""
		
		# u·n = 0 nas bordas
		self.u[0, :] = 0
		self.u[-1, :] = 0
		self.v[:, 0] = 0
		self.v[:, -1] = 0
		
		# Superfície livre nas bordas
		self.grid.eta[0, :] = self.grid.eta[1, :]
		self.grid.eta[-1, :] = self.grid.eta[-2, :]
		self.grid.eta[:, 0] = self.grid.eta[:, 1]
		self.grid.eta[:, -1] = self.grid.eta[:, -2]