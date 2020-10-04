from PIL import Image
import numpy as np

with open('1.bmp', 'rb') as f:
    I = bytearray(f.read())
I = I.rgb2gray(I)
I.reshape(I, 93*93, 1)
W = np.ones(93*93)

for i in range(93*93):
	for j in range(i+1, 93*93):
		W[i,j] = I[i] == I[j]
		W[j,i] = W[i,j]

W = W + 0.1

print("Step 0 completed")

N = len(W)
K = 5

# degree matrix
#D = np.diag(np.dot(W, np.ones(N)))
D = np.diag(np.dot(W,[1]*N))
print("Step 1 completed")

kernel = (D**(-0.5)) @ W @ (D**(-0.5))
V, S = sla.eigs(kernel) # tol = 1e-3
Z = np.dot(D**(-0.5), V[:,:K])
print("Step 2 completed")

ZZ = np.dot(Z, np.transpose(Z))
X_tilde_ast = np.dot(np.multiply(ZZ, np.eye(N,N))**(-0.5), Z)
print("Step 3 completed")

R = np.zeros(K, K)
R[:,0] = np.transpose(X_tilde_ast[random.randint(1,N),:])

c = np.zeros(N, 1)
for k in range(1, K):
	c = c + np.absolute( np.dot(X_tilde_ast, R[:,k-1]))
	cf = np.argwhere(c == np.minimum(c))
	R[:,k] = X_tilde_ast[cf[0],:]
print("step 4 completed")


phi_bar_ast = 0

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

I = np.dot(X_ast, np.transpose(np.array([0, 64, 128, 192, 256])))
I = np.reshape(I, (93,93))
I = np.uint8(I)
Image.imshow(I)
