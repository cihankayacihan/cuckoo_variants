import sys
import numpy as np
import multiprocessing as mp
from joblib import Parallel, delayed
import scipy.special as sp
import itertools
import scipy.stats as ss 
from subprocess import call

Tol = 1e-6

data = np.loadtxt('datafile')

def run_sim(parameters,indice):

	global data

	f = open('prepara','w')
	f.write('begin parameters\n' +
    	'egf_tot       1.2e6    # molecule counts\n' +
    	'egfr_tot      1.8e5    # molecule counts\n' +
    	'Grb2_tot      1.0e5    # molecule counts\n' +
    	'Shc_tot       2.7e5    # molecule counts\n' +
    	'Sos_tot       1.3e4    # molecule counts\n' +
    	'Grb2_Sos_tot  4.9e4    # molecule counts\n' + 

    	'kp1      1.667e-06 # ligand-monomer binding (scaled), units: /molecule/s\n' +
    	'km1           0.06 # ligand-monomer dissociation, units: /s\n' +

    	'kp2      5.556e-06 # aggregation of bound monomers (scaled), units: /molecule/s\n' +
    	'km2            ' + str(parameters[0]) + ' # dissociation of bound monomers, units: /s\n' +

    	'kp3            ' + str(parameters[1]) + ' # dimer transphosphorylation, units: /s\n' +
    	'km3          4.505 # dimer dephosphorylation, units: /s\n' +

    	'kp14             ' + str(parameters[2]) + ' # Shc transphosphorylation, units: /s\n' +
    	'km14          ' + str(parameters[3]) + ' # Shc dephosphorylation, units: /s\n' +
    
    	'km16         0.005 # Shc cytosolic dephosphorylation, units: /s\n' +

    	'kp9      8.333e-07 # binding of Grb2 to receptor (scaled), units: /molecule/s\n' + 
    	'km9           ' + str(parameters[4]) + ' # dissociation of Grb2 from receptor, units: /s\n' +

    	'kp10     5.556e-06 # binding of Sos to receptor (scaled), units: /molecule/s\n' +
    	'km10          ' + str(parameters[5]) + ' # dissociation of Sos from receptor, units: /s\n' +

    	'kp11      1.25e-06 # binding of Grb2-Sos to receptor (scaled), units: /molecule/s\n' +
    	'km11          ' + str(parameters[6]) + ' # diss. of Grb2-Sos from receptor, units: /s\n' +
    
	    'kp13       2.5e-05 # binding of Shc to receptor (scaled), units: /molecule/s\n' +
    	'km13           ' + str(parameters[7]) + ' # diss. of Shc from receptor, units: /s\n' +

    	'kp15       2.5e-07 # binding of ShcP to receptor (scaled), units: /molecule/s\n' +
    	'km15           ' + str(parameters[8]) + ' # diss. of ShcP from receptor, units: /s\n' +

    	'kp17     1.667e-06 # binding of Grb2 to RP-ShcP (scaled), units: /molecule/s\n' +
    	'km17           ' + str(parameters[9]) + ' # diss. of Grb2 from RP-ShcP, units: /s\n' +

    	'kp18       2.5e-07 # binding of ShcP-Grb2 to receptor (scaled), units: /molecule/s\n' +
    	'km18           ' + str(parameters[10]) + ' # diss. of ShcP-Grb2 from receptor, units: /s\n' +

    	'kp19     5.556e-06 # binding of Sos to RP-ShcP-Grb2 (scaled), units: /molecule/s\n' +
    	'km19        ' + str(parameters[11]) + ' # diss. of Sos from RP-ShcP-Grb2, units: /s\n' +

    	'kp20     6.667e-08 # binding of ShcP-Grb2-Sos to receptor (scaled), units: /molecule/s\n' +
    	'km20          ' + str(parameters[12]) + ' # diss. of ShcP-Grb2-Sos from receptor, units: /s\n' +

    	'kp24         5e-06 # binding of Grb2-Sos to RP-ShcP (scaled), units: /molecule/s\n' +
    	'km24        ' + str(parameters[13]) + ' # diss. of Grb2-Sos from RP-ShcP, units: /s\n' +

    	'kp21     1.667e-06 # binding of ShcP to Grb2 in cytosol (scaled), units: /molecule/s\n' +
    	'km21          0.01 # diss. of Grb2 and SchP in cytosol, units: /s\n' + 

    	'kp23     1.167e-05 # binding of ShcP to Grb2-Sos in cytosol (scaled), units: /molecule/s\n' + 
    	'km23           ' + str(parameters[14]) + ' # diss. of Grb2-Sos and SchP in cytosol, units: /s\n' +

    	'kp12     5.556e-08 # binding of Grb2 to Sos in cytosol (scaled), units: /molecule/s\n' +
    	'km12        ' + str(parameters[15]) + ' # diss. of Grb2 and Sos in cytosol, units: /s\n' +

    	'kp22     1.667e-05 # binding of ShcP-Grb2 to Sos in cytosol (scaled), units: /molecule/s\n' +
    	'km22         ' + str(parameters[16]) + ' # diss. of ShcP-Grb2 and Sos in cytosol, units: /s\n' + 

    'loop1 = (kp9/km9)*(kp10/km10)/((kp11/km11)*(kp12/km12))\n' +
    'loop2 = (kp15/km15)*(kp17/km17)/((kp21/km21)*(kp18/km18))\n' +
    'loop3 = (kp18/km18)*(kp19/km19)/((kp22/km22)*(kp20/km20))\n' +
    'loop4 = (kp12/km12)*(kp23/km23)/((kp22/km22)*(kp21/km21))\n' +
    'loop5 = (kp15/km15)*(kp24/km24)/((kp20/km20)*(kp23/km23))\n' +
	'end parameters\n') 
	f.close()

	call("cat prepara" + str(indice) + " afterpara > network" + str(indice) + ".net", shell=True)

	f = open('sim' + str(indice) + '.bngl','w')
	f.write('readFile({file=>"network' + str(indice) + '.net"})\n' +
	'simulate({method=>"ode",t_end=>40,n_steps=>50,atol=>1e-8,rtol=>1e-8,sparse=>1})\n')
	f.close()
	call("BNG2.pl --log sim" + str(indice) + ".bngl", shell=True)

	sim = np.loadtxt('sim' + str(indice) + '.gdat')

	error = 0
	for i in range(sim.shape[0]):
		for j in range(1,16):
			error += ((data[i,j]-sim[i,j])/np.max(data[:,j]))**2

	return error

def fobj(u, indice):
	""" This function calculates the objective function: Bivariate Michaelwicz function """
	#x=u[0]
	#y=u[1]
	#z=-np.sin(x)*np.sin(x**2/np.pi)**20-np.sin(y)*np.sin(2*y**2/np.pi)**20
	#z=-np.cos(x)*np.cos(y)*np.exp(-(x-np.pi)**2-(y-np.pi)**2);
	z = run_sim(u, indice)
	return z

def simple_bounds(s, Lb, Ub):

	for i in range(len(s)):
		if s[i] > Ub[i]:
			s[i] = Ub[i]
		elif s[i] < Lb[i]:
			s[i] = Lb[i] 
	return s

def get_best_nest(nest, newnest, fitness, indice):

	for i in range(nest.shape[0]):
		fnew=fobj(newnest[i,:], indice)
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
		# u=np.random.randn(s.shape[0])*sigma
		# v=np.random.randn(s.shape[0])
		# step=u/abs(v)**(1/beta)
		# stepsize=stepx*step*(s-best);
		# s=s+stepsize*np.random.randn(s.shape[0])
		s=s+ss.levy.rvs(size=2)*1e-10
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

def get_cuckoo(nest, indice, Lb, Ub):

	""" Move a randomly selected cuckoo with Levy flight. The direction of flight is random and the length is 
	sampled from Cauchy distribution. For sampling Mantegna method is used. """
	s = nest[indice,:]
	s=s+ss.levy.rvs(size=17)*1e-10
	nest[indice,:]=simple_bounds(s, Lb, Ub)
	return nest[indice,:]

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
		nd=17

	if Lb is None:
		Lb = np.ones(nd)*0
	if Ub is None:
		Ub = np.ones(nd)*5

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
		a = Parallel(n_jobs=n_jobs)(delayed(single_cuckoo_search)(nests[j],fitness[j],Lb,Ub,all_para[j],step,j) for j in range(9))
		best_nest, fmin, nests, fitness, N_iter, done = zip(*a)
		if True in done:
			ind = np.argmin(fmin)
			total_best_min = fmin[ind]
			total_best_nest = best_nest[ind]
			total_iteration = (i+1)*100+N_iter[ind]
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


def single_cuckoo_search(nest,fitness,Lb,Ub,pa,step,indice):

	global Tol
	n = nest.shape[0]
	fmin, best, nest, fitness = get_best_nest(nest, nest, fitness, indice)
	new_best = fmin
	old_best = np.inf
	bestnest = best
	done = 0
	N_iter=0

	while N_iter < 100:
		cuckoo = np.random.randint(n)
		new_nest = get_cuckoo(nest, cuckoo, Lb, Ub)
		new_obj = fobj(new_nest, indice)
		comp_cuckoo = np.random.randint(n)
		if new_obj < fitness[comp_cuckoo]:
			nest[comp_cuckoo,:] = new_nest

		new_nest = empty_nests(nest, Lb, Ub, pa)
		fnew, best, nest, fitness = get_best_nest(nest, new_nest, fitness, indice)
		N_iter = N_iter + 1 + int(n * pa[0])
		if fnew < fmin:
			print fnew, best, N_iter, indice, pa
			if abs(fnew - fmin) < Tol:
				done = 1
				break
			fmin=fnew
			bestnest=best

	return bestnest, fmin, nest, fitness, N_iter, done
if __name__ == '__main__':
	true_best_nest = np.array([0.1,0.5,3,0.03,0.05,0.06,0.03,0.6,0.3,0.1,0.3,0.0214,0.12,0.0429,0.1,0.0015,0.064])
	all_iters = np.zeros(5)
	all_corrects = np.zeros(5)
	for i in range(5):
		best_nest, fmin, nest, fitness, total_best_min, total_best_nest, total_iteration = cuckoo_search()
		if np.linalg.norm(true_best_nest-total_best_nest)<1e-1:
			all_corrects[i]=1
		all_iters[i]=total_iteration
	print np.mean(all_corrects), np.mean(all_iters), np.std(all_iters)


