import sys
import numpy as np
import multiprocessing as mp
from joblib import Parallel, delayed
import scipy.special as sp
import itertools

Tol = 1e-6

def fobj(u):
	z = np.linalg.norm(u)
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

def get_cuckoos(nest, best, Lb, Ub, stepx):
	n = nest.shape[0]
	beta=3/2;
	sigma=(sp.gamma(1+beta)*np.sin(np.pi*beta/2)/(sp.gamma((1+beta)/2)*beta*2**((beta-1)/2)))**(1/beta);
	for j in range(n):
		s = nest[j,:]
		u=np.random.randn(s.shape[0])*sigma
		v=np.random.randn(s.shape[0])
		step=u/abs(v)**(1/beta)
		stepsize=stepx*step*(s-best);
		s=s+stepsize*np.random.randn(s.shape[0])
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

def parameter_swap(all_para):

	id1 = np.random.randint(len(all_para))
	id2 = np.random.randint(len(all_para))
	all_para = swapElements(all_para, id1, id2)
	return all_para

def cuckoo_search(n=None, nd=None, Lb=None, Ub=None, pa=None):
	if n is None:
		n =25

	if nd is None:
		nd=256

	if Lb is None:
		Lb = np.ones(nd)*-5.12
	if Ub is None:
		Ub = np.ones(nd)*5.12

	# creation of the list for parameter pairs 
	pa = np.linspace(0.1,0.9,9)
	step = 1
	all_para = list(itertools.product(pa))
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
		a = Parallel(n_jobs=n_jobs)(delayed(single_cuckoo_search)(nests[j],fitness[j],Lb,Ub,all_para[j],step) for j in range(9))
		best_nest, fmin, nests, fitness, N_iter, done = zip(*a)
		if True in done:
			ind = np.argmin(fmin)
			total_best_min = fmin[ind]
			total_best_nest = best_nest[ind]
			total_iteration = (i+1)*1000+N_iter[ind]
			break
		k = 0
		while k<5:
			all_para = parameter_swap(all_para) 
			k += 1
		if i != 0:
			ind = np.argmin(fmin)
			total_best_min = fmin[ind]
			total_best_nest = best_nest[ind]

	ind = np.argmin(fmin)
	total_best_min = fmin[ind]
	total_best_nest = best_nest[ind]
	return best_nest, fmin, nests, fitness, total_best_min, total_best_nest, total_iteration


def single_cuckoo_search(nest,fitness,Lb,Ub,pa,step):

	global Tol
	n = nest.shape[0]
	fmin, best, nest, fitness = get_best_nest(nest, nest, fitness)
	new_best = fmin
	old_best = np.inf
	bestnest = best
	done = 0

	N_iter=0

	while N_iter < 1000:
		new_nest = get_cuckoos(nest, best, Lb, Ub, step)
		fnew, best, nest, fitness = get_best_nest(nest, new_nest, fitness)
		N_iter = N_iter + n
		new_nest = empty_nests(nest, Lb, Ub, pa)
		fnew, best, nest, fitness = get_best_nest(nest, new_nest, fitness)
		N_iter = N_iter + n 
		if fnew < fmin:
			if abs(fnew - fmin) < Tol:
				done = 1
				break
			fmin=fnew
			bestnest=best

	return bestnest, fmin, nest, fitness, N_iter, done
if __name__ == '__main__':
	true_best_nest = np.zeros(256)
	all_iters = np.zeros(100)
	all_corrects = np.zeros(100)
	for i in range(100):
		best_nest, fmin, nest, fitness, total_best_min, total_best_nest, total_iteration = cuckoo_search()
		if np.linalg.norm(true_best_nest-total_best_nest)<1:
			all_corrects[i]=1
		all_iters[i]=total_iteration
	print np.mean(all_corrects), np.mean(all_iters), np.std(all_iters)


