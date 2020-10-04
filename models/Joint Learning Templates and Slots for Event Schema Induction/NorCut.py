
import numpy as np
import scipy.linalg as la
import scipy.sparse.linalg as sla
import random

def SpectralClustering(W, D, K):
	"""
	W: np.array: word vector?
	D: np.array: degree matrix?
	K: 

	main program of multiclass spectral clustering
	"""
	N = len(W) # number of templates
	# compute degree matrix
	D = np.diag(np.dot(W,[1]*N))
	W = W + 0.1
	print("Step 1 completed")

	# find optimal eigensolution Z^ast by:
	kernel = (D**(-0.5)) @ W @ (D**(-0.5))
	V, S = sla.eigs(kernel) # tol = 1e-3
	Z = np.dot(D**(-0.5), V[:,:K])
	print("Step 2 completed")

	# normalize Z^ast by
	ZZ = np.dot(Z, np.transpose(Z))
	X_tilde_ast = np.dot(np.multiply(ZZ, np.eye(N))**(-0.5), Z)
	print("Step 3 completed")

	return X_tilde_ast



def trace_function(X, J):
	"""
	to calcualte X_t or X_s in eq. 8
	X: numpy array: |E|X|T| entity-template matrix
	J: numpy array: |E|X|S| entity-sentence matrix
	"""
	A = np.dot(np.transpose(J),X)
	return np.trace(np.dot(A, np.transpose(A)))


def discretization(X_tilde_ast, K, N):
	"""
	"""
	R = np.zeros(N, N)
	R[:,0] = np.transpose(X_tilde_ast[random.randint(0,N-1),:])

	c = np.zeros(N, 1)
	for k in range(1, K):
		c = c + np.absolute( np.dot(X_tilde_ast, R[:,k-1]))
		cf = np.argwhere(c == np.minimum(c))
		R[:,k] = X_tilde_ast[cf[0],:]

	print("step 4 completed")
	
	# Initialize convergence monitoring parameter phi
	phi_bar_ast = 0
	# Find the optimal discrete solution $X^\ast$ by
	step = 1
	eps = 1e-10
	while True:
		X_tilde = np.dot(X_tilde_ast, R)
		X_ast = np.zeros(N, K)
		for i in range(N):
			X_ast[i,:] = X_tilde[i,:] == np.maximum(X_tilde[i,:])
		# find the optimal orthonormal matrix R^ast by:
		U, omega, U_tilde = np.linalg.svd(np.dot(np.transpose(X_ast), X_tilde_ast))
		phi_bar = np.trace(omega)
		print("Iteration %d | Difference = %f\n").format(step, phi_bar - phi_bar_ast)
		if (np.absolute(phi_bar - phi_bar_ast) < eps):
			break
		phi_bar_ast = phi_bar
		R = np.dot(U_tilde, np.transpose(U))
		step += 1

	return X_ast