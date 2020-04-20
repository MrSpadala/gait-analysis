
import numpy as np

def waveletter(v):
	u = np.zeros(int(len(v)/2))
	d = np.zeros(int(len(v)/2))
	for i in range(0,len(u)):
		u[i] = (v[2*i]+v[2*i+1]) / 2
		d[i] = v[2*i]-v[2*i+1]
	return u, d


def de_waveletter(u, d):
	#assert(abs(len(u)-len(d))<=1)
	min_len = min(len(u), len(d))
	v = np.zeros(2*min_len)
	for i in range(min_len):
		v[2*i+1] = u[i] - d[i]/2
		v[2*i] = d[i] + v[2*i+1]
	return v


def wavelet_smooth(v, iterations=2):
	"""
	Smooth the input v using wavelet decomposition:
	1. performs the wavelet decomposition using the number of iterations specified
	2. sets the last vector of differences to 0
	3. reconstruct and return the signal
	"""
	u_list = []
	d_list = []
	prev = v
	# Wavelet decompose for the requested iterations
	for _ in range(iterations):
		u, d = waveletter(prev)
		u_list.append(u)
		d_list.append(d)
		prev = u
	# Set the last d component to 0
	d_list[-1] *= 0
	# Recompose the signal
	u_prev = u_list[-1]
	for it in reversed(range(iterations)):
		d = d_list[it]
		u_prev = de_waveletter(u_prev, d)
	return u_prev


