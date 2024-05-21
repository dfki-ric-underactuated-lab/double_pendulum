import numpy as np
from scipy.optimize import curve_fit as cf


def poly1(t, A, B):
    return A * pow(t, 1) + B


def poly2(t, A, B, C):
    return A * pow(t, 2) + B * pow(t, 1) + C


def poly3(t, A, B, C, D):
    return A * pow(t, 3) + B * pow(t, 2) + C * pow(t, 1) + D


class FitPiecewisePolynomial:
    """
    Gets data and number of break points and
    fit cubic segment polynomials to each section of data.
    data_x: a numpy array x data, usually time, that we want to fit polynomial to it
    data_y: y data that we want to fit polynomial to it
    """

    def __init__(self, data_x, data_y, num_break, poly_degree):
        self.data_x = data_x
        self.data_y = data_y
        self.num_break = num_break
        self.poly_degree = poly_degree
        (
            self.x_sec_data,
            self.y_sec_data,
            self.coeff_sec_data,
        ) = self.create_section_poly()

    def determin_poly(self):
        if self.poly_degree == 1:
            return poly1
        elif self.poly_degree == 2:
            return poly2
        elif self.poly_degree == 3:
            return poly3
        else:
            print('Choose between "1,2,3" for the degree of the polynomial')
            return None

    def end_time(self):
        return self.data_x[-1]

    def start_time(self):
        return self.data_x[0]

    def split_data(self, data):
        """
        Takes the original data and return a list of splitted arrays
        """
        l = len(data)
        sec_len = int(np.ceil(l / self.num_break))
        #         if l % self.num_break != 0 :
        #             print(f'Uneven division.len={l},section_len={sec_len}, last_sec_len={l - (sec_len * (self.num_break - 1))}')
        return np.array_split(data, self.num_break)

    def create_section_poly(self):
        """
        This function takes the splitted data(x, y) and return 2 lists
        - list of the x-data to be fitted to the setion data
        - list of the fitted value
        """
        splitted_data_x = self.split_data(self.data_x)
        splitted_data_y = self.split_data(self.data_y)
        x_sec_data = []
        y_sec_data = []
        coeff_sec_data = []
        index = 0
        for sec in splitted_data_x:
            x_sec_data.append(np.linspace(sec[0], sec[-1], len(sec)))
            func = self.determin_poly()  # self.poly3
            p_coeff, p_cov = cf(
                func, splitted_data_x[index], splitted_data_y[index], maxfev=2000
            )
            fit = func(x_sec_data[index], *p_coeff)
            y_sec_data.append(fit)
            coeff_sec_data.append(p_coeff)
            index += 1
        self.x_sec_data = x_sec_data
        self.y_sec_data = y_sec_data
        self.coeff_sec_data = coeff_sec_data
        return x_sec_data, y_sec_data, coeff_sec_data

    def get_value(self, value):
        poly_index = min(
            [
                index
                for index, element in enumerate(
                    [any(poly >= value) for poly in self.x_sec_data]
                )
                if element == True
            ]
        )
        p_coeff = self.coeff_sec_data[poly_index]
        func = self.determin_poly()  # self.poly3
        return func(value, *p_coeff)


class InterpolateVector:
    def __init__(self, T, X, num_break=40, poly_degree=3):
        self.dim = np.shape(X)[1]
        self.X = []
        for d in range(self.dim):
            if np.count_nonzero(X[:, d]) == 0:
                self.X.append(None)
            else:
                pol = FitPiecewisePolynomial(T, X[:, d], num_break, poly_degree)
                self.X.append(pol)

    def get_value(self, value):
        x = np.empty(self.dim)
        for d in range(self.dim):
            if self.X[d] is not None:
                val = self.X[d].get_value(value)
                x[d] = val
            else:
                x[d] = 0.0
        return x


class InterpolateMatrix:
    def __init__(self, T, X, num_break=40, poly_degree=3):
        self.dim1 = np.shape(X)[1]
        self.dim2 = np.shape(X)[2]
        self.X = []
        for d1 in range(self.dim1):
            Xd1 = []
            for d2 in range(self.dim2):
                if np.count_nonzero(X[:, d1, d2]) == 0:
                    Xd1.append(None)
                else:
                    pol = FitPiecewisePolynomial(
                        T, X[:, d1, d2], num_break, poly_degree
                    )
                    Xd1.append(pol)
            self.X.append(Xd1)

    def get_value(self, value):
        x = np.empty((self.dim1, self.dim2))
        for d1 in range(self.dim1):
            for d2 in range(self.dim2):
                if self.X[d1][d2] is not None:
                    val = self.X[d1][d2].get_value(value)
                    x[d1, d2] = val
                else:
                    x[d1, d2] = 0.0
        return x


def ResampleTrajectory(T, X, U, dt, num_break=40, poly_degree=3):
    n = int(T[-1] / dt)

    X_interp = InterpolateVector(T=T, X=X, num_break=num_break, poly_degree=poly_degree)

    U_interp = InterpolateVector(T=T, X=U, num_break=num_break, poly_degree=poly_degree)

    T_resamp = np.linspace(0, T[-1], n)
    X_resamp = []
    U_resamp = []
    for t in T_resamp:
        X_resamp.append(X_interp.get_value(t))
        U_resamp.append(U_interp.get_value(t))
    X_resamp = np.asarray(X_resamp)
    U_resamp = np.asarray(U_resamp)
    return T_resamp, X_resamp, U_resamp
