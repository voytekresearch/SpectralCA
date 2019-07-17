import numpy as np

def white_noise(t, mu=0, sigma=1):
	return np.random.normal(loc=mu, scale=sigma, size=(len(t)))

def conv_color_noise(t, kernel, mu=0, sigma=1):
	# convolved (filtered) white noise
	x = np.random.normal(loc=mu, scale=sigma, size=(len(t)+len(kernel)))
	return np.convolve(x,kernel,mode='same')[:len(t)]

def OU_process(t, theta=1., mu=0., sigma=5.):
	# discretized Ornstein-Uhlenbeck process: 
	#	dx = theta*(x-mu)*dt + sigma*dWt, where
	# dWt: increments of Wiener process, i.e. white noise
	# theta: memory scale (higher = faster fluc)
	# mu: mean
	# sigma: std
	# see: https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process#Solution
	# for integral solution
		
	x0 = mu
	dt = t[1]-t[0]
	Ws = np.random.normal(size=len(t))
	ex = np.exp(-theta*t);
	Ws[0]=0.
	return x0*ex+mu*(1.-ex)+sigma*ex*np.cumsum(np.exp(theta*t)*np.sqrt(dt)*Ws)

def sim_noise(t, num_trials = 10, color='white', params=None):
	"""
	Simulates many trials of white noise, colored noise (filtered white), 
	or an Ornstein-Uhlenbeck process, with the specified parameters.
	
	 --- Args ---
    t : array, 1d
        time indices to simulate over, at dt resolution

    num_trials : int
        number of trials to simulate
        default: 10

	color : str {'white', 'colored', 'OU'}
		type of noise to simulate

    params : dict
        dictionary of simulation parameters, including
        'mu' - mean
        'sigma' - std dev
        'theta' - "memory" in OU process
        'tau' - time constant (seconds) of filter in colored noise
        
        default: 
        params = {
			'mu': 0.,
			'sigma': 1.,
			'theta': 1.,
			'tau': 0.01
		}

    --- Return ---
    x : 2D float array
    	matrix of simulated data, time x trial
	"""
	if params is None:
		params = {
			'mu': 0.,
			'sigma': 1.,
			'theta': 1.,
			'tau': 0.01
		}

	dt = t[1]-t[0]
	mu=params['mu']
	sigma=params['sigma']
	x = np.zeros((len(t), num_trials))
	if color is 'white':
		for tr in range(num_trials):
			x[:,tr] = white_noise(t, mu=mu, sigma=sigma)

	elif color is 'colored':
		tau = params['tau']
		ker = np.exp(-(1./tau)*np.arange(0,1,dt))
		for tr in range(num_trials):
			x[:,tr] = conv_color_noise(t, kernel=ker, mu=mu, sigma=sigma)

	elif color is 'OU':
		theta = params['theta']
		for tr in range(num_trials):
			x[:,tr] = OU_process(t, theta=theta, mu=mu, sigma=sigma)

	return x



# def noise_sim(color, numtrs, with_osc = False, oFreq = 20.22, oAmp = 20):
# 	fs = 1000.
# 	t = np.arange(0.,600.,1./fs)
# 	ker = np.exp(-50*np.arange(0,500)/1000.)

# 	# steady sinusoidal oscillation
# 	osc = np.sin(2*np.pi*oFreq*np.arange(0,len(t))/fs)*1.

# 	# initialize array
# 	SCVnoise = np.zeros((numtrs,501))
# 	if color == 'white':
# 		for tr in range(0,numtrs):
# 	    	# white noise
# 	    	x = np.random.normal(size=(len(t)))
# 	    	if with_osc == True:
# 	    		f, SCV = spectral.scv(x+osc, fs, noverlap=0)
# 	    	else:
# 	    		f, SCV = spectral.scv(x, fs, noverlap=0)
# 	    	SCVnoise[tr,:] = SCV
# 	elif color == 'colored':
# 			for tr in range(0,numtrs):
# 		    # colored noise
# 		    x = np.random.normal(size=(len(t)))
# 		    y = np.convolve(x,ker,mode='same')
# 		    if with_osc == True:
# 	    		if with_osc == True:
# 	    		f, SCV = spectral.scv(y+osc, fs, noverlap=0)
# 	    	else:
# 	    		f, SCV = spectral.scv(y, fs, noverlap=0)
# 	    	SCVnoise[tr,:] = SCV   

# 	return SCVnoise
