import numpy as np
import openmdao.api as om
from run_simulation import simulate_falcon


class falcon_simulation(om.ExplicitComponent):

    def setup(self):
        self.add_input('ctrl_gains', val=np.array([20,30,20,60]), desc='INDI ctrl gains')

        self.add_output('x_final', val=10., desc='x at the final of the simulation time')
        self.add_output('z_final', val=10., desc='z at the final of the simulation time')
        self.add_output('y_error', val=0., desc='how much y deviates from zero')

        self.declare_partials(of='*', wrt='*', method='fd')

    def compute(self, inputs, outputs):
        ctrl_gains = inputs['ctrl_gains']

        x_final,z_final,y_error = simulate_falcon(ctrl_gains)

        outputs['x_final'] = x_final
        outputs['z_final'] = z_final
        outputs['y_error'] = y_error**2 # to minimize if positive or negative


class MDA(om.Group):
    def setup(self):
        cycle = self.add_subsystem('cycle', om.Group(), promotes=['*'])
        cycle.add_subsystem('gains', falcon_simulation(), promotes=['*'])

if __name__ == "__main__":
    MDO = om.Problem()
    MDO.model = MDA()


    MDO.driver = om.ScipyOptimizeDriver()
    MDO.driver.options['optimizer'] = 'COBYLA'
    #MDO.driver = om.pyOptSparseDriver()
    #MDO.driver.options['optimizer'] = 'IPOPT'
    #MDO.driver.opt_settings['tol'] = 5.0E-5

    # Design variables
    MDO.model.add_design_var('ctrl_gains', lower=0.01, upper=60, scaler= 100)

    # Constraints
    MDO.model.add_constraint('x_final', lower=3)
    MDO.model.add_constraint('z_final', lower=20.)

    # Objective function
    MDO.model.add_objective('y_error')

    # Ask OpenMDAO to finite-difference across the model to compute the gradients for the optimizer
    #MDO.model.approx_totals()
    MDO.setup()
    MDO.run_driver()
