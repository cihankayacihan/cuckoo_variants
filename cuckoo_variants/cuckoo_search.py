import sys
import numpy as np
import multiprocessing as mp
from joblib import Parallel, delayed
import itertools

def fobj(u):
	x=u[0]
	y=u[1]
	z=-np.sin(x)*np.sin(x**2/np.pi)**20-np.sin(y)*np.sin(2*y**2/np.pi)**20
	return z

def simple_bounds(s, Lb, Ub):

	for i in range(len(s)):
		if s[i] > Ub[i]:
			s[i] = Ub[i]
		elif s[i] < Lb[i]:
			s[i] = Lb[i] 
	return s

def get_best_nest(nest, newnest, fitness):

	for i in range(nest.shape[0]):
		fnew=fobj(newnest[i,:])
		if fnew < fitness[i]:
			fitness[i]=fnew
			nest[i,:]=newnest[i,:]

	fmin = min(fitness)
	K = np.argmin(fitness)
	best = nest[K,:]
	return (fmin, best, nest, fitness)

def get_cuckoos(nest, best, Lb, Ub, step):
	n = nest.shape[0]
	for j in range(n):
		s = nest[j,:]
		if np.linalg.norm(s-best)>1e-6:
			s = s + step * np.random.randn(s.shape[0])
		nest[j,:]=simple_bounds(s, Lb, Ub)
	return nest

def empty_nests(nest, Lb, Ub, pa):
	n = nest.shape[0]
	K = np.random.random(nest.shape) > pa
	stepsize = np.random.rand()*(nest[np.random.permutation(n),:]-nest[np.random.permutation(n),:])
	new_nest = nest + stepsize * K
	for j in range(new_nest.shape[0]):
		s = new_nest[j,:]
		new_nest[j,:] = simple_bounds(s, Lb, Ub)
	return new_nest

def cuckoo_search(n=None, nd=None, Lb=None, Ub=None, pa=None):
	if n is None:
		n =25

	if nd is None:
		nd=2

	if Lb is None:
		Lb = np.ones(nd)*-100
	if Ub is None:
		Ub = np.ones(nd)*100

	# creation of the list for parameter pairs 
	pa = np.linspace(0.1,9.,64)
	step = np.linspace(0.01,Lb[0] + Ub[0], 64)
	all_para = list(itertools.product(pa,step))
	threads = len(all_para)

    # initialization of the nests for each system
	Tol = 1.e-12
	nests = np.zeros((threads,n,nd))
	for j in range(threads):
		for i in range(n):
			nests[j,i] = Lb + (Ub-Lb)*np.random.rand(len(Lb))

	fitness = 10**10 * np.ones((threads,n,1))

	n_jobs = mp.cpu_count()
	for i in range(100):
		a = Parallel(n_jobs=n_jobs)(delayed(single_cuckoo_search)(nests[i],fitness[i],Lb,Ub,all_para[i][0],all_para[i][1]) for i in range(64))
		best_nest, fmin, nest, fitness = zip(*a)
		np.random.shuffle(all_para)

	ind = np.argmin(fmin)
	total_best_min = fmin[ind]
	total_best_nest = best_nest[ind]
	return best_nest, fmin, nest, fitness, total_best_min, total_best_nest


def single_cuckoo_search(nest,fitness,Lb,Ub,pa,step):

	n = nest.shape[0]
	fmin, best, nest, fitness = get_best_nest(nest, nest, fitness)
	new_best = fmin
	old_best = np.inf
	bestnest = best

	N_iter=0

	while N_iter < 1000:
		new_nest = get_cuckoos(nest, best, Lb, Ub, step)
		fnew, best, nest, fitness = get_best_nest(nest, new_nest, fitness)
		N_iter = N_iter + n
		new_nest = empty_nests(nest, Lb, Ub, pa)
		fnew, best, nest, fitness = get_best_nest(nest, new_nest, fitness)
		N_iter = N_iter + n 
		if fnew < fmin:
			fmin=fnew
			bestnest=best

	return bestnest, fmin, nest, fitness

best_nest, fmin, nest, fitness, total_best_min, total_best_nest = cuckoo_search()
