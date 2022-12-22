import numpy as np

# def errfunc_with_friction(xb, Q, phi):
#     return Q.flatten() - phi.dot(xb)


class errfunc():
    def __init__(self, Q, phi, bounds, rescale=False, scalar=False):
        self.Q = Q.flatten()
        self.phi = phi
        self.bounds = np.asarray(bounds)
        self.rescale = rescale
        self.scalar = scalar

    def __call__(self, xb):
        if self.rescale:
            x = self.unscale_pars(xb)
        else:
            x = np.copy(xb)

        if self.scalar:
            loss = np.sum(np.abs(self.Q - self.phi.dot(x)))
        else:
            loss = self.Q - self.phi.dot(x)
        return loss

    def unscale_pars(self, pars):
        """
        [0, 1] -> par
        """
        p = np.copy(pars)
        p = self.bounds[0] + p*(self.bounds[1]-self.bounds[0])
        return p

    def rescale_pars(self, pars):
        """
        par -> [0, 1]
        """
        p = np.copy(pars)
        p = (p - self.bounds[0]) / (self.bounds[1]-self.bounds[0])
        return p


class errfunc_nl():
    def __init__(self, dyn_fun, bounds, X, ACC, U, rescale=False, scalar=False):
        self.dyn_fun = dyn_fun
        self.X = X
        self.ACC = ACC
        self.U = U
        self.bounds = np.asarray(bounds)
        self.rescale = rescale
        self.scalar = scalar

    def __call__(self, xb):
        if self.rescale:
            x = self.unscale_pars(xb)
        else:
            x = np.copy(xb)

        loss = self.dyn_fun(
                self.X.T[0],
                self.X.T[1],
                self.X.T[2],
                self.X.T[3],
                self.ACC.T[0],
                self.ACC.T[1],
                #self.U.T[0],
                #self.U.T[1],
                *x)[:, 0, :] - self.U.T

        if self.scalar:
            loss = np.sum(np.abs(loss))
        return loss

    def unscale_pars(self, pars):
        """
        [0, 1] -> par
        """
        p = np.copy(pars)
        p = self.bounds[0] + p*(self.bounds[1]-self.bounds[0])
        return p

    def rescale_pars(self, pars):
        """
        par -> [0, 1]
        """
        p = np.copy(pars)
        p = (p - self.bounds[0]) / (self.bounds[1]-self.bounds[0])
        return p
