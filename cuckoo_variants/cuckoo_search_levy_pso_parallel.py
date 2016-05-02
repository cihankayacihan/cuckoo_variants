import sys
import numpy as np
import multiprocessing as mp
from joblib import Parallel, delayed
import itertools
import scipy.special as sp

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

def all_bounds(nest, Lb, Ub):

	for i in range(nest.shape[0]):
		nest[i,:]=simple_bounds(nest[i,:],Lb,Ub)

	return nest


def get_best_nest(nest, newnest, fitness, pbest):

	for i in range(nest.shape[0]):
		fnew=fobj(newnest[i,:])
		if fnew < fitness[i]:
			fitness[i]=fnew
			nest[i,:]=newnest[i,:]
		if fnew < fobj(pbest[i,:]):
			pbest[i,:]=newnest[i,:]

	fmin = min(fitness)
	K = np.argmin(fitness)
	best = nest[K,:]
	return (fmin, best, nest, fitness, pbest)

def get_cuckoos(nest, best, Lb, Ub, step):
	n = nest.shape[0]
	beta=3/2;
	sigma=(sp.gamma(1+beta)*np.sin(np.pi*beta/2)/(sp.gamma((1+beta)/2)*beta*2**((beta-1)/2)))**(1/beta);
	for j in range(n):
		s = nest[j,:]
		u=np.random.randn(s.shape[0])*sigma
		v=np.random.randn(s.shape[0])
		steps=u/abs(v)**(1/beta)
		stepsize=step*steps*(s-best);
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

def get_gbest(pbest):
	n = pbest.shape[0]
	fitness = np.zeros(n)
	for i in range(n):
		fitness[i]=fobj(pbest[i])
	best = np.argmin(fitness)
	return pbest[best,:]	

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
	pa = np.linspace(0.2,0.3,8)
	step = np.linspace(0.5,1.5,8)
	c1 = 0.1
	c2 = 0.6
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
	for i in range(10):
		a = Parallel(n_jobs=n_jobs)(delayed(single_cuckoo_search)(nests[i],fitness[i],Lb,Ub,all_para[i][0],all_para[i][1],c1,c2) for i in range(64))
		best_nest, fmin, nest, fitness = zip(*a)
		np.random.shuffle(all_para)

	ind = np.argmin(fmin)
	total_best_min = fmin[ind]
	total_best_nest = best_nest[ind]
	return best_nest, fmin, nest, fitness, total_best_min, total_best_nest


def single_cuckoo_search(nest,fitness,Lb,Ub,pa,step,c1,c2):

	n = nest.shape[0]
	d = nest.shape[1]
	pbest = nest
	fmin, best, nest, fitness, pbest = get_best_nest(nest, nest, fitness, pbest)
	new_best = fmin
	old_best = np.inf
	bestnest = best
	velocity = np.zeros((n,d))

	N_iter=0

	while N_iter < 1000:
		new_nest = get_cuckoos(nest, best, Lb, Ub, step)
		fnew, best, nest, fitness, pbest = get_best_nest(nest, new_nest, fitness, pbest)
		N_iter = N_iter + n
		gbest = get_gbest(pbest)
		velocity = velocity + c1 * (pbest - nest) + c2 * (gbest - nest)
		nest = nest + velocity
		nest = all_bounds(nest, Lb, Ub)

		new_nest = empty_nests(nest, Lb, Ub, pa)
		fnew, best, nest, fitness, pbest = get_best_nest(nest, new_nest, fitness, pbest)
		N_iter = N_iter + n 
		if fnew < fmin:
			fmin=fnew
			bestnest=best

	return bestnest, fmin, nest, fitness

best_nest, fmin, nest, fitness, total_best_min, total_best_nest = cuckoo_search()
