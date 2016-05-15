import sys
import numpy as np
import scipy.special as sp
import scipy.stats as ss 
import multiprocessing as mp
from joblib import Parallel, delayed

Tol = 1e-6

def fobj(u):
	""" This function calculates the objective function: Bivariate Michaelwicz function """
	#x=u[0]
	#y=u[1]
	#z=-np.sin(x)*np.sin(x**2/np.pi)**20-np.sin(y)*np.sin(2*y**2/np.pi)**20
	#z=-np.cos(x)*np.cos(y)*np.exp(-(x-np.pi)**2-(y-np.pi)**2);
	z = np.linalg.norm(u)
	return z

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
	#beta=3/2;
	#sigma=(sp.gamma(1+beta)*np.sin(np.pi*beta/2)/(sp.gamma((1+beta)/2)*beta*2**((beta-1)/2)))**(1/beta);
	for j in range(n):
		s = nest[j,:]
		#u=np.random.randn(s.shape[0])*sigma
		#v=np.random.randn(s.shape[0])
		#step=u/abs(v)**(1/beta)
		#stepsize=0.01*step*(s-best);
		s=s+ss.levy.rvs(size=256)*1e-20
		nest[j,:]=simple_bounds(s, Lb, Ub)
	return nest

def get_cuckoo(nest, indice, Lb, Ub):

	s = nest[indice,:]
	s=s+ss.levy.rvs(size=16)*1e-100
	nest[indice,:]=simple_bounds(s, Lb, Ub)
	return nest[indice,:]

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
		nd=16

	if Lb is None:
		Lb = np.ones(nd)*-5.12
	if Ub is None:
		Ub = np.ones(nd)*5.12

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
	ini_best = fmin
	new_best = fmin
	old_best = np.inf
	bestnest = best

	N_iter=0
	k = 0
	global Tol

	while N_iter < 100000:
		cuckoo = np.random.randint(n)
		new_nest = get_cuckoo(nest, cuckoo, Lb, Ub)
		new_obj = fobj(new_nest)
		comp_cuckoo = np.random.randint(n)
		if new_obj < fitness[comp_cuckoo]:
			nest[comp_cuckoo,:] = new_nest

		new_nest = empty_nests(nest, Lb, Ub, pa)
		fnew, best, nest, fitness = get_best_nest(nest, new_nest, fitness)
		if fnew < fmin:
			if abs(fnew - fmin) < Tol:
				break
			fmin=fnew
			bestnest=best
		N_iter = N_iter + 1 + int(n * pa)

	return bestnest, fmin, nest, fitness, N_iter

def para_analyze(j):

	true_best_nest = np.zeros(16)
	pa = np.linspace(0.1,0.9,9)
	all_corrects = np.zeros(10)
	all_iters = np.zeros(10)
	std_iters = np.zeros(10)
	dist = np.zeros(10)
	
	for i in range(10):
		if i%10==0:
			print j,i
		best_nest, fmin, nest, fitness, N_iter = cuckoo_search(pa = pa[j])
		dist[i]=np.linalg.norm(true_best_nest-best_nest)
		if np.linalg.norm(true_best_nest-best_nest)<1:
			all_corrects[i]=1
		all_iters[i]=N_iter
	correct_ratio = np.mean(all_corrects)
	iters = np.mean(all_iters)
	std_iters = np.std(all_iters)
	return j,correct_ratio, iters, std_iters, dist

if __name__ == '__main__':
	n_jobs = mp.cpu_count()
	a = Parallel(n_jobs=n_jobs)(delayed(para_analyze)(j) for j in range(9))	
	j, correct_ratio, iters, std_iters, dist = zip(*a)







