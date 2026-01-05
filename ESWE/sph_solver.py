from mimetypes import init
import numpy as np
from scipy.spatial import ckdtree

class SPHSolver:
	"""Solver SPH 3D para dinâmica de quebra de onda"""

	def __init__(self, g=9.81, rho0=1000.0, h_sph=0.5):
		"""
		Inicializa solver SPH

		Parametros:
		-----------
		g: float
			Aceleração gravitacional (m/s²)
		rho0: float
			Densidade de referencia (kg/m³)
		h_sph: float
			Smoothing length do kernel SPH (m)
		"""
		self.g = g
		self.rho0 = rho0
		self.h = h_sph # Smoothing length

		# Lista de particulas (cada uma é um dict)
		self.particles = []

		# Parametros SPH
		self.c_s = 10.0 # Velocidade do som artificial (m/s)
		self.alpha = 0.1 # Viscosidade artificial
		self.gamma = 7.0 # Expoente equação do estado

		# Constantes do kernel
		self._setup_kernel_constants()
		
	def _setup_kernel_constants(self):
		"""Define constantes do kernel cúbico spline"""
		self.kernel_const = 8.0 / (np.pi * self.h**3)

	def cubic_spline_kernel(self, r, h):
		"""
		Kernel cúbico spline 3D (Wendland C2)

		Parametros:
		-----------
		r: float ou ndarray
			Distancia entre particulas
		h: float
			Smoothing length

		Retorna:
		---------
		W: float ou ndarray
			Valor do kernel
		"""
		q = r / h
		W = np.zeros_like(q)

		# Suporte compacto [0, 2h]
		mask1 = q < 1.0
		mask2 = (q >= 1.0) & (q < 2.0)

		W[mask1] = self.kernel_const * (1.0 - 1.5*q[mask1]**2 + 0.75*q[mask1]**3)
		W[mask2] = self.kernel_const * 0.25 * (2.0 - q[mask2]) ** 3

		return W
	
	def cubic_spline_gradient(self, r_vec, r, h):
		"""
		Gradiente do kernel cúbico spline

		Parametros:
		-----------
		r_vec: ndarray
			Vetor distância (3D)
		r: float
			Magnitude da distância
		h: float
			Smoothing length

		Retorna:
		---------
		grad_W: ndarray
			Gradiente do kernel (vetor 3D)
		"""
		if r < 1e-6:
			return np.zeros(3)

		q = r / h

		if q < 1.0:
			dW_dq = self.kernel_const * (-3.0 * q + 2.25 * q **2)
		elif q < 2.0:
			dW_dq = self.kernel_const * 0.75 * (2.0 - q) ** 2
		else:
			return np.zeros(3)

		return (dW_dq / h) * (r_vec / r)

	def add_particle(self, position, velocity, mass):
		"""
		Adiciona particula SPH 

		Parametros:
		------------
		position: ndarray
			Posição inicial [x, y, z] (m)
		velocity: ndarray
			Velocidade inicial [u, v, w] (m/s)
		mass: float
			Massa da particula (kg)
		"""
		particle = {
			'position': np.array(position, dtype=float),
			'velocity': np.array(velocity, dtype=float),
			'acceleration': np.zeros(3),
			'mass': mass,
			'density': self.rho0,
			'pressure': 0.0
		}
		self.particles.append(particle)

	def compute_density(self):
		"""Calcula densidade de cada particula usando somatorio SPH"""
		if len(self.particles) == 0:
			return

		# Constroi arvore KD para busca eficinete de vizinhos
		positions = np.array([p['position'] for p in self.particles])
		tree = ckdtree(positions)


		for i, particle in enumerate(self.particles):
			# Busca vizinhos dentro de 2h
			neighbors = tree.query_ball_point(particle['position'], 2*self.h)

			rho = 0.0
			for j in neighbors:
				if i == j:
					continue

				r_vec = particle['position'] - self.particles[j]['position']
				r = np.linalg.norm(r_vec)

				W = self.cubic_spline_kernel(r, self.h)
				rho += self.particles[j]['mass'] * W


			# Atualiza densidade
			particle['density'] = max(rho, self.rho0 * 0.1)

	def compute_pressure(self):
		"""Calcula pressão usando equação de estado de Tait"""
		for particle in self.particles:
			# Equação de Tait
			B = self.c_s**2 * self.rho0 / self.gamma
			particle['pressure'] = B * ((particle['density'] / self.rho0) ** self.gamma - 1.0)
			
			# Pressão não pode ser negativa
			particle['pressure'] = max(particle['pressure'], 0.0)

	def compute_forces(self):
		"""Calcula forças SPH"""
		if len(self.particles) == 0:
			return

		positions = np.array([p['position'] for p in self.particles])
		tree = ckdtree(positions)

		for i, particle_i in enumerate(self.particles):
			particle_i['acceleration'] = np.array([0.0, 0.0, -self.g])

			neighbors = tree.query_ball_point(particle_i['position'], 2*self.h)

			for j in neighbors:
				if i == j:
					continue

				particle_j = self.particles[j]
				r_vec = particle_i['position'] - particle_j['position']
				r = np.linalg.norm(r_vec)

				if r < 1e-6:
					continue


				# Gradiente do kernel
				grad_W = self.cubic_spline_gradient(r_vec, r, self.h)


				# Força simetrica
				pressure_term = (particle_i['pressure'] / particle_i['density']**2 +
								 particle_j['pressure'] / particle_j['density']**2)
				F_pressure = -particle_j['mass'] * pressure_term * grad_W

				# Viscosidade
				v_ij = particle_i['velocity'] - particle_j['velocity']
				rho_ij = 0.5 * (particle_i['density'] + particle_j['density'])

				if np.dot(v_ij, r_vec) < 0:
					mu_ij = self.h * np.dot(v_ij, r_vec) / (r**2 + 0.01*self.h**2)
					pi_ij = -self.alpha * self.c_s * mu_ij / rho_ij

					F_viscosity = -particle_j['mass'] * pi_ij * grad_W
				else:
					F_viscosity = np.zeros(3)

				particle_i['acceleration'] += F_pressure + F_viscosity
	def step(self, dt):
		"""
		Integra particulas SPH no tempo (Leap-Frog)

		Parametros:
		------------
		dt: float
			Passo de tempo (s)
		"""
		if len(self.particles) == 0:
			return


		self.compute_density()
		self.compute_pressure()
		self.compute_forces()

		for particle in self.particles:
			particle['velocity'] += dt * particle['acceleration']

			particle['position'] += dt * particle['velocity']

	def get_momentum_change(self, initial_velocity):
		"""
		Calcula a variação de momentum horizontal

		Parametros:
		------------
		initial_velocity: list of ndarray
			Velocidades iniciais das particulas
		Retorna:
		---------
		delta_P: ndarray
			Variação de momentum horizontal [dPx, dPy] (kg * m/s)
		"""
		if len(self.particles) == 0:
			return np.zeros(2)

		delta_P = np.zeros(2)

		for i, particle in enumerate(self.particles):
			if i < len(initial_velocity):
				v_initial = initial_velocity[i]
				v_final = particle['velocity']

				delta_v = v_final[:2] - v_initial[:2]
				delta_P += particle['mass'] * delta_v

		return delta_P

	def check_hydrostatic_recovery(self, threshold=0.05):
		"""
		Verifica criterio de desativação


		Parametros:
		-----------
		threshold: float
			Limiar de aceleração vertical (fração de g)

		Retorna:
		---------
		is_hydrostatic: bool
			True se sistema retornou ao regime hidrostatico
		"""
		if len(self.particles) == 0:
			return True

		max_az = max(abs(p['acceleration'][2] + self.g) for p in self.particles)

		return max_az < threshold * self.g

	def clear_particles(self):
		self.particles = []

