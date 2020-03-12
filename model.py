import pybamm
import numpy as np
import matplotlib.pylab as plt
import pints


class SingleReactionSolution(pints.ForwardModel):
    def __init__(self, param=None):
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
                "Voltage amplitude [V]": 0.03,
                "Scan Rate [V s-1]": 0.08941,
            })



        # Create dimensional fixed parameters
        c_inf = pybamm.Parameter("Far-field concentration of A [mol cm-3]")
        D = pybamm.Parameter("Diffusion Constant [cm2 s-1]")
        F = pybamm.Parameter("Faraday Constant [C mol-1]")
        R = pybamm.Parameter("Gas constant [J K-1 mol-1]")
        S = pybamm.Parameter("Electrode Area [cm2]")
        T = pybamm.Parameter("Temperature [K]")
        omega_d = pybamm.Parameter("Voltage frequency [rad s-1]")
        E_start_d = pybamm.Parameter("Voltage start [V]")
        E_reverse_d = pybamm.Parameter("Voltage reverse [V]")
        deltaE_d = pybamm.Parameter("Voltage amplitude [V]")
        v = pybamm.Parameter("Scan Rate [V s-1]")

        # Create dimensional input parameters
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
            i: 1/(Cdl * Ru) * (i_f + Cdl * Eapp.diff(pybamm.t) - i),
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
            i: pybamm.Scalar(0),
        }

        # set spatial variables and solution domain geometry
        x = pybamm.SpatialVariable('x', domain="solution")
        model.default_geometry = pybamm.Geometry({
            "solution": {
                "primary": {x: {"min": pybamm.Scalar(0), "max": pybamm.Scalar(20)}}
            }
        })
        model.default_var_pts = {
            x: 100
        }

        # Using Finite Volume discretisation on an expanding 1D grid for solution
        model.default_submesh_types = {
            "solution": pybamm.MeshGenerator(pybamm.Exponential1DSubMesh, {'side': 'left'})
        }
        model.default_spatial_methods = {
            "solution": pybamm.FiniteVolume()
        }

        # model variables
        model.variables = {
            "Current [non-dim]": i,
        }

        #--------------------------------

        # Set model parameters
        param.process_model(model)
        geometry = model.default_geometry
        param.process_geometry(geometry)

        # Create mesh and discretise model
        mesh = pybamm.Mesh(geometry, model.default_submesh_types, model.default_var_pts)
        disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
        disc.process_model(model)

        # Create solver
        solver = pybamm.CasadiSolver(mode="fast")

        # Store discretised model and solver
        self._model = model
        self._solver = solver
        self._omega_d = param.process_symbol(omega_d).evaluate()
        self._I_0 = param.process_symbol(I_0).evaluate()
        self._T_0 = param.process_symbol(T_0).evaluate()

    def simulate(self, parameters, times):
        input_parameters = {
            "Reaction Rate [s-1]": parameters[0],
            "Reversible Potential [V]": parameters[1],
            "Symmetry factor [non-dim]": parameters[2],
            "Uncompensated Resistance [Ohm]": parameters[3],
            "Capacitance [F]": parameters[4],
        }

        solution = self._solver.solve(self._model, times, inputs=input_parameters)
        return solution["Current [non-dim]"](times)

    def n_parameters(self):
        return 5


if __name__ == '__main__':
    model = SingleReactionSolution()
    x = [0.0101, 0.214, 0.53, 8.0, 20.0e-6]

    n = 2000
    t_eval = np.linspace(0, 50, n)

    plt.plot(t_eval, model.simulate(x, t_eval))
    plt.ylabel("current [non-dim]")
    plt.xlabel("time [non-dim]")
    plt.show()
