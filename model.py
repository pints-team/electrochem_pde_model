import pybamm
import numpy as np
import matplotlib.pylab as plt
import pints
import time
from data import ECTimeData


class SingleReactionSolution(pints.ForwardModel):
    def __init__(self, n=100, max_x=10, param=None):
        # Set fixed parameters here
        if param is None:
            param = pybamm.ParameterValues({
                "Far-field concentration of A [mol cm-3]": 1e-6,
                "Diffusion Constant [cm2 s-1]": 7.2e-6,
                "Faraday Constant [C mol-1]": 96485.3328959,
                "Gas constant [J K-1 mol-1]": 8.314459848,
                "Electrode Area [cm2]": 0.07,
                "Temperature [K]": 297.0,
                "Voltage frequency [rad s-1]": 9.0152,
                "Voltage start [V]": 0.5,
                "Voltage reverse [V]": -0.1,
                "Voltage amplitude [V]": 0.08,
                "Scan Rate [V s-1]": 0.08941,
            })

        # Create dimensional fixed parameters
        c_inf = pybamm.Parameter("Far-field concentration of A [mol cm-3]")
        D = pybamm.Parameter("Diffusion Constant [cm2 s-1]")
        F = pybamm.Parameter("Faraday Constant [C mol-1]")
        R = pybamm.Parameter("Gas constant [J K-1 mol-1]")
        S = pybamm.Parameter("Electrode Area [cm2]")
        T = pybamm.Parameter("Temperature [K]")

        E_start_d = pybamm.Parameter("Voltage start [V]")
        E_reverse_d = pybamm.Parameter("Voltage reverse [V]")
        deltaE_d = pybamm.Parameter("Voltage amplitude [V]")
        v = pybamm.Parameter("Scan Rate [V s-1]")

        # Create dimensional input parameters
        E0 = pybamm.InputParameter("Reversible Potential [non-dim]")
        k0 = pybamm.InputParameter("Reaction Rate [non-dim]")
        alpha = pybamm.InputParameter("Symmetry factor [non-dim]")
        Cdl = pybamm.InputParameter("Capacitance [non-dim]")
        Ru = pybamm.InputParameter("Uncompensated Resistance [non-dim]")
        omega_d = pybamm.InputParameter("Voltage frequency [rad s-1]")


        E0_d = pybamm.InputParameter("Reversible Potential [V]")
        k0_d = pybamm.InputParameter("Reaction Rate [s-1]")
        alpha = pybamm.InputParameter("Symmetry factor [non-dim]")
        Cdl_d = pybamm.InputParameter("Capacitance [F]")
        Ru_d = pybamm.InputParameter("Uncompensated Resistance [Ohm]")

        # Create scaling factors for non-dimensionalisation
        E_0 = R * T / F
        T_0 = E_0 / v
        L_0 = pybamm.sqrt(D * T_0)
        I_0 = D * F * S * c_inf / L_0

        # Non-dimensionalise parameters
        E0 = E0_d / E_0
        k0 = k0_d * L_0 / D
        Cdl = Cdl_d * S * E_0 / (I_0 * T_0)
        Ru = Ru_d * I_0 / E_0
        omega = 2 * np.pi * omega_d * T_0

        E_start = E_start_d / E_0
        E_reverse = E_reverse_d / E_0
        t_reverse = E_start - E_reverse
        deltaE = deltaE_d / E_0

        # Input voltage protocol
        Edc_forward = -pybamm.t
        Edc_backwards = pybamm.t - 2*t_reverse
        Eapp = E_start + \
            (pybamm.t <= t_reverse) * Edc_forward + \
            (pybamm.t > t_reverse) * Edc_backwards + \
            deltaE * pybamm.sin(omega * pybamm.t)

        # create PyBaMM model object
        model = pybamm.BaseModel()

        # Create state variables for model
        theta = pybamm.Variable("ratio_A", domain="solution")
        i = pybamm.Variable("Current")

        # Effective potential
        Eeff = Eapp - i * Ru

        # Faradaic current
        i_f = pybamm.BoundaryGradient(theta, "left")

        # ODE equations
        model.rhs = {
            theta: pybamm.div(pybamm.grad(theta)),
            i: 1/(Cdl * Ru) * (-i_f + Cdl * Eapp.diff(pybamm.t) - i),
        }

        # algebraic equations (none)
        model.algebraic = {
        }

        # Butler-volmer boundary condition at electrode
        theta_at_electrode = pybamm.BoundaryValue(theta, "left")
        butler_volmer = k0 * (
            theta_at_electrode * pybamm.exp(-alpha * (Eeff - E0))
            - (1 - theta_at_electrode) * pybamm.exp((1-alpha) * (Eeff - E0))
        )

        # Boundary and initial conditions
        model.boundary_conditions = {
            theta: {
                "right": (pybamm.Scalar(1), "Dirichlet"),
                "left": (butler_volmer, "Neumann"),
            }
        }

        model.initial_conditions = {
            theta: pybamm.Scalar(1),
            i: Cdl * (-1.0 + deltaE * omega),
        }

        # set spatial variables and solution domain geometry
        x = pybamm.SpatialVariable('x', domain="solution")
        default_geometry = pybamm.Geometry({
            "solution": {
                x: {"min": pybamm.Scalar(0), "max": pybamm.Scalar(max_x)}
            }
        })

        default_var_pts = {
            x: n
        }

        # Using Finite Volume discretisation on an expanding 1D grid for solution
        default_submesh_types = {
            "solution": pybamm.MeshGenerator(pybamm.Exponential1DSubMesh, {'side': 'left'})
        }
        default_spatial_methods = {
            "solution": pybamm.FiniteVolume()
        }

        # model variables
        model.variables = {
            "Current [non-dim]": i,
        }

        #--------------------------------

        # Set model parameters
        param.process_model(model)
        geometry = default_geometry
        param.process_geometry(geometry)

        # Create mesh and discretise model
        mesh = pybamm.Mesh(geometry, default_submesh_types, default_var_pts)
        disc = pybamm.Discretisation(mesh, default_spatial_methods)
        disc.process_model(model)

        # Create solver
        solver = pybamm.CasadiSolver(mode="fast",
                                     rtol=1e-9,
                                     atol=1e-9,
                                     extra_options_setup={'print_stats': False})
        #model.convert_to_format = 'jax'
        #solver = pybamm.JaxSolver(method='BDF')
        #model.convert_to_format = 'python'
        #solver = pybamm.ScipySolver(method='BDF')

        # Store discretised model and solver
        self._model = model
        self._solver = solver
        self._fast_solver = None
        self._omega_d = param["Voltage frequency [rad s-1]"]

        self._I_0 = param.process_symbol(I_0).evaluate()
        self._T_0 = param.process_symbol(T_0).evaluate()
        self._E_0 = param.process_symbol(E_0).evaluate()
        self._L_0 = param.process_symbol(L_0).evaluate()
        self._S = param.process_symbol(S).evaluate()
        self._D = param.process_symbol(D).evaluate()
        self._default_var_points = default_var_pts

    def non_dim(self, x):
        if len(x) > 6:
            return np.array([
                x[0] * self._L_0 / self._D,
                x[1] / self._E_0,
                x[2],
                x[3] * self._I_0 / self._E_0,
                x[4] * self._S * self._E_0 / (self._I_0 * self._T_0),
                x[5] * 2 * np.pi * self._T_0,
                x[6]
            ])
        else:
            return np.array([
                x[0] * self._L_0 / self._D,
                x[1]/self._E_0,
                x[2],
                x[3] * self._I_0 / self._E_0,
                x[4] * self._S * self._E_0 / (self._I_0 * self._T_0),
                x[5] * 2 * np.pi * self._T_0
            ])

    def simulate(self, parameters, times):
        input_parameters = {
            "Reaction Rate [s-1]": parameters[0],
            "Reversible Potential [V]": parameters[1],
            "Symmetry factor [non-dim]": parameters[2],
            "Uncompensated Resistance [Ohm]": parameters[3],
            "Capacitance [F]": parameters[4],
            "Voltage frequency [rad s-1]": parameters[5],
        }
        # if self.fast_solver is None:
        #    solution = self._solver.solve(self._model, times, inputs=input_parameters)

        index = list(self._default_var_points.values())[0]

        try:
            solution = self._solver.solve(self._model, times, inputs=input_parameters)
            return solution.y[index:index+1, :].reshape(-1)
        except pybamm.SolverError:
            print('solver errored for params',parameters)
            return np.zeros_like(times)

    def n_parameters(self):
        return 6


if __name__ == '__main__':

    # pybamm.set_logging_level('INFO')
    model = SingleReactionSolution()
    data = ECTimeData('GC02_FeIII-1mM_1M-KCl_02a_009Hz.txt', model,
                      ignore_begin_samples=5, ignore_end_samples=0, samples_per_period=200)

    x = np.array([0.0101, 0.214, 0.53, 8.0, 20.0e-6, 9.0152, 0.01])

    n = 2000
    t_eval = np.linspace(0, 50, n)

    t0 = time.perf_counter()
    y1 = model.simulate(x, t_eval)
    t1 = time.perf_counter()
    y2 = model.simulate(x, t_eval)
    t2 = time.perf_counter()
    print('times', t1-t0, t2-t1)
    plt.plot(data.times, data.current)
    plt.plot(t_eval, y2)
    plt.ylabel("current [non-dim]")
    plt.xlabel("time [non-dim]")
    plt.show()
