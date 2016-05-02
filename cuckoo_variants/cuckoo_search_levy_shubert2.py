import sys
import numpy as np
import scipy.special as sp

Tol = 1e-6

def fobj(u):
	""" This function calculates the objective function: Bivariate Michaelwicz function """
	x=u[0]
	y=u[1]
	#z=-np.sin(x)*np.sin(x**2/np.pi)**20-np.sin(y)*np.sin(2*y**2/np.pi)**20
	#z=-np.cos(x)*np.cos(y)*np.exp(-(x-np.pi)**2-(y-np.pi)**2);
	z = 0
	for i in range(5):
		z = z + (i+1) * np.cos((i+2)*x+1)
	t = 0	
	for i in range(5):
		t = t + (i+1) * np.cos((i+2)*y+1)

	return z*t

def simple_bounds(s, Lb, Ub):

	""" If a bird goes out of search space, it kept at the boundaries. """

	for i in range(len(s)):
		if s[i] > Ub[i]:
			s[i] = Ub[i]
		elif s[i] < Lb[i]:
			s[i] = Lb[i] 
	return s

def get_best_nest(nest, newnest, fitness):

	""" Checking that the new nests have better fitness. If not it will keep the previous solution. Also, 
	Returns the best nest and the best fitness function. """

	for i in range(nest.shape[0]):
		fnew=fobj(newnest[i,:])
		if fnew < fitness[i]:
			fitness[i]=fnew
			nest[i,:]=newnest[i,:]

	fmin = min(fitness)
	K = np.argmin(fitness)
	best = nest[K,:]
	return (fmin, best, nest, fitness)

def get_cuckoos(nest, best, Lb, Ub):

	""" Move a randomly selected cuckoo with Levy flight. The direction of flight is random and the length is 
	sampled from Cauchy distribution. For sampling Mantegna method is used. """
	n = nest.shape[0]
	beta=3/2;
	sigma=(sp.gamma(1+beta)*np.sin(np.pi*beta/2)/(sp.gamma((1+beta)/2)*beta*2**((beta-1)/2)))**(1/beta);
	for j in range(n):
		s = nest[j,:]
		u=np.random.randn(s.shape[0])*sigma
		v=np.random.randn(s.shape[0])
		step=u/abs(v)**(1/beta)
		stepsize=0.01*step*(s-best);
		s=s+stepsize*np.random.randn(s.shape[0])
		nest[j,:]=simple_bounds(s, Lb, Ub)
	return nest

def empty_nests(nest, Lb, Ub, pa):
	""" This function move birds from the nests with low fitness functions to alternative locations """
	n = nest.shape[0]
	K = np.random.random(nest.shape) > pa
	stepsize = np.random.rand()*(nest[np.random.permutation(n),:]-nest[np.random.permutation(n),:])
	new_nest = nest + stepsize * K
	for j in range(new_nest.shape[0]):
		s = new_nest[j,:]
		new_nest[j,:] = simple_bounds(s, Lb, Ub)
	return new_nest

def cuckoo_search(n=None, nd=None, Lb=None, Ub=None, pa=None):
	""" Main function for cuckoo determination """
	if n is None:
		n =25

	if nd is None:
		nd=2

	if Lb is None:
		Lb = np.ones(nd)*-10
	if Ub is None:
		Ub = np.ones(nd)*10

	if pa is None:
		pa = 0.25

	# creation of the list for parameter pairs 
	
	step = 1

    # initialization of the nests
	nests = np.zeros((n,nd))
	for i in range(n):
		nests[i,:] = Lb + (Ub-Lb)*np.random.rand(len(Lb))

	fitness = 10**10 * np.ones((n,1))
	best_nest, fmin, nest, fitness, N_iter = single_cuckoo_search(nests,fitness,Lb,Ub,pa,step) 

	return best_nest, fmin, nest, fitness, N_iter


def single_cuckoo_search(nest,fitness,Lb,Ub,pa,step):

	""" This function creates a group of cuckoos and optimizes fitness function. """

	n = nest.shape[0]
	fmin, best, nest, fitness = get_best_nest(nest, nest, fitness)
	new_best = fmin
	old_best = np.inf
	bestnest = best

	N_iter=0
	k = 0

	while N_iter < 100000:
		new_nest = get_cuckoos(nest, best, Lb, Ub)
		fnew, best, nest, fitness = get_best_nest(nest, new_nest, fitness)
		N_iter = N_iter + n
		new_nest = empty_nests(nest, Lb, Ub, pa)
		fnew, best, nest, fitness = get_best_nest(nest, new_nest, fitness)
		N_iter = N_iter + n 
		if fnew < fmin:
			if abs(fnew - fmin) < Tol:
				break
			fmin=fnew
			bestnest=best

	return bestnest, fmin, nest, fitness, N_iter

if __name__ == '__main__':
	true_fmin = -186.73
	pa = np.linspace(0.1,0.9,9)
	correct_ratio = np.zeros(9)
	iters = np.zeros(9)
	std_iters = np.zeros(9)
	all_iters = np.zeros((9,100))
	all_corrects = np.zeros((9,100))
	for j in range(9):
		print pa[j]
		for i in range(100):
			best_nest, fmin, nest, fitness, N_iter = cuckoo_search(pa = pa[j])
			if abs(true_fmin-fmin)<1:
				all_corrects[j,i]=1
			all_iters[j,i]=N_iter
		correct_ratio[j] = np.mean(all_corrects[j,:])
		iters[j] = np.mean(all_iters[j,:])
		std_iters[j] = np.std(all_iters[j,:])






