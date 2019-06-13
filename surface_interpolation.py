import numpy as np
from scipy import interpolate, optimize


class SurfInterp:
    """Creates a parametric multi-variate interpolation function.

    Once the SurfInterp object is created, it can be called like a
    function where psi and eta points are passed in and interpolated
    x, y, z points are returned.

    The interpolation is performed with the scipy.interpolate.griddata
    function and the cubic setting."""
    def __init__(self, surf_data, x_bounds, y_bounds):
        self._points = np.array([surf_data[0], surf_data[1]]).T
        self._values = surf_data[2]
        self._x_l = x_bounds[0]
        self._x_u = x_bounds[1]
        self._y_l = y_bounds[0]
        self._y_u = y_bounds[1]

    def __call__(self, psi, eta):
        x = self._x_l(eta)[0]+psi*(self._x_u(eta)[0]-self._x_l(eta)[0])
        # y = self._eta_l(psi)[1]+eta*(self._eta_u(psi)[1]-self._eta_l(psi)[1])
        y = self._y_l(psi)[1]+eta*(self._y_u(psi)[1]-self._y_l(psi)[1])

        z = self._z_interp(x, y)
        # plt.scatter(x, y)
        # plt.show()
        # ****verify that ordering below is correct
        # plt.scatter(x[0, :], y[0, :])
        # plt.scatter(x[:, 0], y[:, 0])
        # plt.show()
        z[0, :] = self._x_l(eta[0, :])[2]
        z[-1, :] = self._x_u(eta[-1, :])[2]
        z[:, -1] = self._y_u(psi[:, -1])[2]
        z[:, 0] = self._y_l(psi[:, 0])[2]
        # plt.scatter(x, y, c=z)
        # plt.show()
        # plt.scatter(y[:, 0], z[:, 0])
        # plt.show()
        # z[-1, :] = 0.
        # z[:, 0] = 0.
        # z[:, -1] = 0.

        return x, y, z

    def inverse(self, x, y):
        psi = np.full_like(x, 0.)
        eta = np.full_like(x, 0.)

        # for psi_i, eta_i, x_i, y_i in np.nditer([x, y]):
        for i in range(len(x)):
            sol = optimize.root(self._distance, (0.5, 0.5),
                                args=(x[i], y[i]))
            psi[i], eta[i] = sol.x
            # print(sol.fun)
                # psi[i, j] = psi_c
        # (x - self._x_l(eta)[0])+psi*(self._x_u(eta)[0]-self._x_l(eta)[0])
        return psi, eta

    def _z_interp(self, x, y):
        z_interp = interpolate.griddata(self._points, self._values,
                                        (x, y), method='cubic')

        return z_interp

    def _distance(self, psieta, x0, y0):
        psi, eta = psieta

        # x, y, z = self.__call__(psi, eta)
        x = self._x_l(eta)[0]+psi*(self._x_u(eta)[0]-self._x_l(eta)[0])
        # y = self._eta_l(psi)[1]+eta*(self._eta_u(psi)[1]-self._eta_l(psi)[1])
        y = self._y_l(psi)[1]+eta*(self._y_u(psi)[1]-self._y_l(psi)[1])
        dist = np.sqrt((x-x0)**2+(y-y0)**2)

        return x-x0, y-y0


class Spline1D:
    """Provides parametric function interface for scipy.interpolate.splprep"""
    def __init__(self, x, y, z, smooth=0.):  # , i_return=None):
#        print(x, y, z)
        self._tck = interpolate.splprep([x, y, z], s=smooth)[0]
        # self._i_return = i_return

    def __call__(self, p):
        return interpolate.splev(p, self._tck)
        # if self._i_return is not None:
        #     return interpolate.splev(p, self._tck)[self._i_return]
        # else:


class Linear1D:
    def __init__(self, x_range, y_range, z_range):
        self._x_range = x_range
        self._y_range = y_range
        self._z_range = z_range

    def __call__(self, p):
        x = self._interp(self._x_range, p)
        y = self._interp(self._y_range, p)
        z = self._interp(self._z_range, p)

        return x, y, z

    @staticmethod
    def _interp(value_range, p):
        first, last = value_range
        v = first + p*(last-first)

        return v


def gen_wing_surface(dp_wing, dp_inner_edge, dp_outer_edge):
    """A wrapper around SurfInterp that simplifies wing interpolation
    surface generation"""
    y_lower = Spline1D(dp_inner_edge[:, 0],
                       dp_inner_edge[:, 1],
                       dp_inner_edge[:, 2])
    y_upper = Spline1D(dp_outer_edge[:, 0],
                       dp_outer_edge[:, 1],
                       dp_outer_edge[:, 2])
    x_lower = Linear1D([dp_inner_edge[0, 0], dp_outer_edge[0, 0]],
                       [dp_inner_edge[0, 1], dp_outer_edge[0, 1]],
                       [dp_inner_edge[0, 2], dp_outer_edge[0, 2]])
    x_upper = Linear1D([dp_inner_edge[-1, 0], dp_outer_edge[-1, 0]],
                       [dp_inner_edge[-1, 1], dp_outer_edge[-1, 1]],
                       [dp_inner_edge[-1, 2], dp_outer_edge[-1, 2]])

    wing_surf = SurfInterp(dp_wing.T, (x_lower, x_upper), (y_lower, y_upper))

    return wing_surf


class CartToCyl:
    # This is not a cylindrical coordinate system in the strict sense
    # because the fuselage is not straight and picking a single straight
    # axis would not work. Instead the midway point between the upper and
    # lower edges of the fuselage is used and polar coordinates are
    # generated about that.
    # z_l and z_u are parameterized splines where the parameter
    # that is passed in ranges from zero to one. The x_length is
    # used in the CartToCyl to obtain x/x_length to be passed in as the
    # parameter
    def __init__(self, z_bounds, x_max):
        self._z_l = z_bounds[0]
        self._z_u = z_bounds[1]
        self._x_max = x_max

    def __call__(self, x, y, z):
        psi = x/self._x_max
        z_mid = (self._z_l(psi)[2]+self._z_u(psi)[2])/2.
        r = np.sqrt(y*y+(z-z_mid)**2)
        with np.errstate(divide='ignore'):
            ratio = (z-z_mid)/y
        theta = np.arctan(ratio)

        return x, theta, r

    def inverse(self, x, theta, r):
        psi = x/self._x_max
        z_mid = (self._z_l(psi)[2]+self._z_u(psi)[2])/2.
        y = r*np.cos(theta)
        z = r*np.sin(theta)+z_mid

        return x, y, z

# def gen_fuselage_surface(dp_fuselage, dp_lower_edge, dp_upper_edge,
#                          dp_front_edge, dp_back_edge):
