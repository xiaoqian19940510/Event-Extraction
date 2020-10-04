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


#time0 = time.time()

print("Loading J, WT, WS, ...")

## to be modified according to file
with open("J.txt", "r") as j:
	J = j.readlines()

# classify template
with open("TS.txt", "r") as ts:
	WT = ts.readlines()

WT = WT + 0.1
N = len(WT)
DT = np.diag(np.dot(WT, np.ones(N,1)))
KT = 4

# classify slot
with open("SS.txt", "r") as ss:
	WS = ss.readlines()

WS = WS + 1.1
DS = np.diag(np.diag(WS, np.ones(N,1)))
KS = 4

# save data? wat?
WT = 4 - WT
WS = 4 - WS
DT = np.diag(np.dot(WT, np.ones(N,1)))
DS = np.diag(np.dot(WS, np.ones(N,1)))
print("loading J, WT, WS done!")

# initialize XT, XS
XT = np.zeros(N, KT)
for i in range(N):
	XT[i, random.randint(0,KS-1)] == 1
print("initializing XT, XS done")

# start to iterate
print("start iteration")
XT_old = XT
XS_old = XS
Totaliter = 1
lambda1 = 2 #/KT
lambda2 = 210000000 #/KT

# normalized cut? NC
XT = SpectralClustering(WT, DT, KT)
XS = SpectralClustering(WS, DS, KS)
# normalized cut + spectral clustering
while True:
    JT = ((1.0*1/2)/trace_function(XS,J))/lambda1*np.dot(DT,np.dot(J,np.transpose(J)))
    WT_ast = np.transpose(JT) + WT + JT
    XT = SpectralClustering(WT_ast,DT,KT)
    print('\n .............WT iteration completed\n');
    
    JS = 0.5 * trace_function(XT,J) / lambda2*np.dot((np.ones(N,N)-np.dot(J,np.transpose(J))),DS)
    WS_ast = np.transpose(JS) + WS + JS
  
    XS = SpectralClustering(WS_ast,DS,KS);
    print('\n .............WS iteration completed\n');
    
    R1,resid,rank,s = np.linalg.lstsq(np.dot(np.transpose(XT),XT), np.dot(np.transpose(XT), XT_old))
    R2, resid,rank,s = np.linalg.lstsq(np.dot(np.transpose(XS),XS), np.dot(np.transpose(XS),XS_old))
    residual2 = np.linalg.norm(np.dot(np.transpose(R1),R1) - np.eye(KT))+ np.linalg.norm(np.dot(np.transpose(R2),R2) - np.eye(KS));
    if residual2<0.001:
        break
        
    XT_old = XT
    XS_old = XS
    print('\n **************************************************************\n')
    print(('             Total iteration %d completed | Cost = %f').format(Totaliter,residual2))
    print('\n **************************************************************\n')
    Totaliter = Totaliter + 1

print('Iteration Completed, Staring to discretize...\n')
XT = discretization(XT,KT,N)
XS = discretization(XS,KS,N)
print('Discretize completed, start to write to file...\n')
# output the result(template,slot)
# save slot XT, XS
with open("NCSC.txt", "w") as fp:
	for t in range(KT):
		ct = XT[:,t] == 1
		for i in range(KS):
			c = XS[:,i] == 1
			ts = np.where(np.add(ct,c)>0, 1, 0) #ct & c ## element wise logical - 0 or not
			ts = np.argwhere(ts == 1)
			fp.write(str(len(ts))+"\n")
			for j in range(len(ts)):
				fp.write(str(ts[j]))
			fp.write("\n\n")
# for stats
with open("NCSC_stat.txt","w") as fp:
	for e in range(N):
		fp.write(np.argwhere(J[e,:]==1))
		fp.write(np.argwhere(XT[e,:]==1))		
		fp.write(np.argwhere(XS[e,:]==1))		
		fp.write("\n")
print("Everything is done")

# output image
I = np.dot(X_ast, np.array([0,64,128,192,256]))
I = np.reshape(I, (93,93))
I = np.uint8(I)
Image.imshow(I)

# output result(template) and save
with open("templates.txt","w") as fp:
	for i in range(K):
		c = np.argwhere(X_ast[:,i]==1)
		fp.write(str(len(c)))
		for j in range(len(c)):
			fp.write(str(len(c[j])))
		fp.write("\n\n")
