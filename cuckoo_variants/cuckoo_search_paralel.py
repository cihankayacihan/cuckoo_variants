import sys
import numpy as np
import multiprocessing as mp
from joblib import Parallel, delayed
import itertools

def fobj(u):
	x=u[0]
	y=u[1]
	z=-np.sin(x)*np.sin(x**2/np.pi)**20-np.sin(y)*np.sin(2*y**2/np.pi)**20
	#z=-np.cos(x)*np.cos(y)*np.exp(-(x-np.pi)**2-(y-np.pi)**2);
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

def swapElements(arr, id1, id2):

	arr[id1], arr[id2] = arr[id2], arr[id1]
	return arr

def parameter_swap(all_para, fmin):

	id1 = np.argmin(fmin)
	id2 = np.random.randint(len(all_para))
	all_para = swapElements(all_para, id1, id2)
	return all_para

def cuckoo_search(n=None, nd=None, Lb=None, Ub=None, pa=None):
	if n is None:
		n =25

	if nd is None:
		nd=2

	if Lb is None:
		Lb = np.ones(nd)*0
	if Ub is None:
		Ub = np.ones(nd)*5

	# creation of the list for parameter pairs 
	pa = 0.25
	step = np.linspace(0.01,10,8)
	all_para = list(itertools.product(step))
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
		a = Parallel(n_jobs=n_jobs)(delayed(single_cuckoo_search)(nests[j],fitness[j],Lb,Ub,pa,all_para[j]) for j in range(8))
		best_nest, fmin, nest, fitness = zip(*a)
		all_para = parameter_swap(all_para, fmin)
		if i != 0:
			ind = np.argmin(fmin)
			total_best_min = fmin[ind]
			total_best_nest = best_nest[ind]
			print total_best_min, total_best_nest, all_para[ind]


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
if __name__ == '__main__':
	best_nest, fmin, nest, fitness, total_best_min, total_best_nest = cuckoo_search()
