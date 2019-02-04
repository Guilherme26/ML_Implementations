import numpy as np, math, copy

class LogisticRegression():
	x = None
	y = None
	theta = None
	alpha = 0.0005
	num_feat = 0
	num_input = 0
	_noChange = 0

	def __init__(self, alpha=0.0005):
		self.alpha = alpha
		self._noChange = 0
		self.x = []

	def fit(self, x, y):
		"""
			The input must be modified in order to generalize the 
			gradient descent formula. That is, add a constant 
			factor to multiply the error value theta_0
		"""
		for item in x:
			self.x.append(np.concatenate(([1], item)))
		self.x = np.array(self.x)
		self.y = np.array(y)
		self.num_input = len(self.x)
		self.num_feat = len(self.x[0])

		# Creates an array with the dimensions equals the number of features
		self.theta = np.ones(self.num_feat)
		
		while True:
			# If the algorithm has no changes through three iterations, it converged
			if self._noChange > 3:
				break

			self._updateWeight()

	def predict(self, xi):
		# Equal probability for either positive or negative
		if self._hypothesis(np.concatenate(([1], xi))) > 0.5:
			return True
		return False

	def _updateWeight(self):
		err = self._calc_error()
		theta_tmp = copy.copy(self.theta)

		for j in range(self.num_feat):
			theta_tmp[j] = theta_tmp[j] - self.alpha * err[j]

		if self._compare(theta_tmp):
			self._noChange += 1
		else:
			self._noChange = 0

		self.theta = theta_tmp

	def _compare(self, theta_tmp):
		for i, j in zip(self.theta, theta_tmp):
			if(np.round(i,4) != np.round(j,4)):
				return False
		return True

	def _hypothesis(self, xi):
		return 1 / (1 + math.pow(math.e, np.sum(self.theta * xi)))

	# xi[j] is the value of a feature from a record
	def _calc_error(self):
		err = np.zeros(self.num_feat)
		for xi, yi in zip(self.x, self.y):
			for j in range(self.num_feat):
				err[j] = err[j] + (self._hypothesis(xi) - yi) * xi[j]

		return (-1 / self.num_input) * err