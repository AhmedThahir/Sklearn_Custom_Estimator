## Estimator

```python
import numpy as np
from scipy import optimize as o
from inspect import getfullargspec

from sklearn.base import BaseEstimator
#from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.pipeline import make_pipeline

import numpy as np
from scipy import optimize as o
from inspect import getfullargspec

from sklearn.base import BaseEstimator
#from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
```

```python
class CustomRegressionModel(BaseEstimator):
	"""
	All variables inside the Class should end with underscore
	"""
	def __str__(self):
		return str(self.model)
	def __repr__(self):
		return str(self)
	
	def mse(self, pred, true, sample_weight):
		error = pred - true
		
		loss = error**2

		# median is robust to outliers than mean
		cost = np.mean(
			sample_weight * loss
		)

		return cost

	def loss(self, pred, true):
		return self.error(pred, true, self.sample_weight)
	
	def l1(self, params):
		return np.sum(np.abs(params-self.model.initial_guess))
	def l2(self, params):
		return np.sum((params-self.model.initial_guess) ** 2)
	def l3(self, params):
		return self.l1(params) + self.l2(params)

	def reg(self, params, penalty_type="l3", lambda_reg_weight = 1.0):
		"""
		lambda_reg_weight = Coefficient of regularization penalty
		"""

		if penalty_type == "l1":
			penalty = self.l1(params)
		elif penalty_type == "l2":
			penalty = self.l2(params)
		elif penalty_type == "l3":
			penalty = self.l3(params)
		else:
			raise Exception

		return lambda_reg_weight * penalty/self.sample_size

	def cost(self, params, X, y):
		pred = self.model.equation(X, *params)
		return self.loss(pred, true=y) + self.reg(params)

	def fit(self, X, y, model, method="L-BFGS-B", error = None, sample_weight=None, alpha=0.95):
		check_X_y(X, y) #Using self.X,self.y = check_X_y(self.X,self.y) removes column names

		self.X = X
		self.y = y

		self.n_features_in_ = self.X.shape[1]

		if sample_weight is None or len(sample_weight) <= 1: # sometimes we can give scalar sample weight same for all
			self.sample_size = self.X.shape[0]
		else:
			self.sample_size = sample_weight[sample_weight > 0].shape[0]

		self.sample_weight = (
			sample_weight
			if sample_weight is not None
			else np.full(self.sample_size, 1) # set Sample_Weight as 1 by default
		)

		self.error = (
			error
			if error is not None
			else self.mse
		)

		self.model = model

		params = getfullargspec(self.model.equation).args
		params = [param for param in params if param not in ['self', "x"]]
		
		self.optimization = o.minimize(
			self.cost,
			x0 = self.model.initial_guess,
			args = (self.X, self.y),
			method = method, # "L-BFGS-B", "Nelder-Mead", "SLSQP",
			constraints = [

			],
			bounds = [
				(-1, None) for param in params # variables must be positive
			]
		)

		success = self.optimization.success
		if success is False:
			st.warning("Did not converge!")
			st.stop()
			return False

		self.popt = (
			self.optimization.x
		)

		self.rmse = mse(
			self.output(self.X),
			self.y,
			sample_weight = self.sample_weight,
			squared=False
		)

		self.dof = self.sample_size - len(params) - 1 # n-k-1

		if "hess_inv" in self.optimization:
			self.covx = (
				self.optimization
				.hess_inv
				.todense()
			)

			self.pcov = list(
				np.diag(
					self.rmse *
					np.sqrt(self.covx)
				)
			)
		
			self.popt_with_uncertainty = [
				f"""{{ \\small (
					{round_s(popt, 2)}
					±
					{round_s(stats.t.ppf(alpha, self.dof) * pcov.round(2), 2)}
				)}}""" for popt, pcov in zip(self.popt, self.pcov)
			]
		else:
			self.popt_with_uncertainty = [
				f"""{{ \\small (
					{round_s(popt, 2)}
					±
					{round_s(stats.t.ppf(alpha, self.dof) * self.rmse, 2)}
				)}}""" for popt, pcov in zip(self.popt, self.pcov)
			]

		self.model.set_fitted_coeff(*self.popt_with_uncertainty)
		
		return self
	
	def output(self, X):
		return (
			self.model
			.equation(X, *self.popt)
		)
	
	def get_se_x_cent(self, X_cent):
		return self.rmse * np.sqrt(
			(1/self.sample_size) + (X_cent.T).dot(self.covx).dot(X_cent)
		)
	def get_pred_se(self, X):
		if False: # self.covx is not None: # this seems to be abnormal. check this
			X_cent = X - self.X.mean()
			se = X_cent.apply(self.get_se_x_cent, axis = 1)
		else:
			se = self.rmse
		return se

	def predict(self, X, alpha=0.95):
		check_is_fitted(self) # Check to verify if .fit() has been called
		check_array(X) #X = check_array(X) # removes column names

		pred = (
			self.output(X)
			.astype(np.float32)
		)

		se = self.get_pred_se(X)

		ci =  stats.t.ppf(alpha, self.dof) * se

		return pd.concat([pred, pred+ci, pred-ci], axis=1)
```

```python
model = CustomRegressionModel()
print(model) # prints latex

model.fit(
  X_train,
  y_train,
  model = Arrhenius(),
  method = "Nelder-Mead"
)
model.predict(X_test)

print(model) # prints latex with coefficent values
```

## Model

```python
class Math:
	def __str__(self):
		return self.latex()
	def __repr__(self):
		return str(self)
	def equation(self):
		return ""
	def latex(self):
		return ""

class Huber(Math):
	def cost(self, pred, true, sample_weight, delta=None):
		error = pred - true
		error_abs = np.abs(error)

		if delta is None:
			# delta = 0.1
			delta = 1.345 * np.std(error)

		# huber loss
		loss = np.where(
			error_abs > delta,
			(error**2 / 2),
			(
				delta * error_abs -
				delta**2 / 2
			)
		)

		cost = np.mean(
			sample_weight * loss
		)
		return cost
      
	def latex(self):
		return r"""
		$$
		\Large
		\text{mean}
		\begin{cases}
		\dfrac{u_i^2}{2}, & \vert u_i \vert > \delta \\
		\delta \vert u_i \vert - \dfrac{\delta^2}{2}, & \text{otherwise}
		\end{cases} \\
		\delta_\text{recommended} = 1.345 \sigma_u
		$$
		"""

class Model(Math):
	def __init__(self):
		args = getfullargspec(self.equation).args
		self.args = tuple([arg for arg in args if arg not in ["self", "x"]])
		
		self.defaults = getfullargspec(self.equation).defaults

		self.initial_guess = (
			[0 for arg in self.args]
			if (self.defaults is None) or (len(self.args) != len(self.defaults))
			else self.defaults
		)

		self.fitted_coeff = None

	def set_fitted_coeff(self, *fitted_coeff):
		self.fitted_coeff = fitted_coeff
	def __str__(self):
		fillers = (
			self.args
			if self.fitted_coeff is None
			else self.fitted_coeff
		)
		
		return self.latex() % fillers

class LinearRegression(Model):
	def equation(self, x, m, c):  # hypothesis
		x = x["Input"]
		return (m * x) + c
	def latex(self):
		return r"""
		$$
		\begin{aligned}
    y &= \textcolor{hotpink}{%s} x + \textcolor{hotpink}{%s} \\
    \text{where }
    x &= \text{Input}
    \end{aligned}
		$$
		"""
```

