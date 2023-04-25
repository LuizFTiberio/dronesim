"""
I took this as is from Beard and McLain code and their awesome book
https://github.com/randybeard/uavbook
-------------------------------------------
Class to determine wind velocity at any given moment,
calculates a steady wind speed and uses a stochastic
process to represent wind gusts. (Follows section 4.4 in uav book)
"""
import sys
from dronesim.utils.transfer_function import transferFunction
import numpy as np


class WindSimulation:
    def __init__(self, Ts):
        # steady state wind defined in the inertial frame
        self._steady_state = np.array([[0, 0, 0]]).T #5, 1.5, 0.

        #   Dryden gust model parameters (section 4.4 UAV book)
        Va = 20
        Lu = 200
        Lv = 200
        Lw = 50
        gust_flag = False
        if gust_flag==True:
            sigma_u = 1.06
            sigma_v = 1.06
            sigma_w = 0.7
        else:
            sigma_u = 0.
            sigma_v = 0.
            sigma_w = 0.

        # Dryden transfer functions (section 4.4 UAV book)
        u_num = sigma_u * np.sqrt(2*Va)* np.array([[1]])
        u_den = np.sqrt(np.pi*Lu) * np.array([[1,Va/Lu]])
        self.u_w = transferFunction(num=u_num, den=u_den, Ts=Ts)

        v_num = sigma_v * np.sqrt(3*Va) * np.array([[1, Va/(Lv*np.sqrt(3))]])
        v_den = np.sqrt(np.pi*Lv) * np.array([[1, 2*Va/Lv, (Va/Lv)**2]])
        self.v_w = transferFunction(num=v_num, den=v_den, Ts=Ts)

        w_num = sigma_w * np.sqrt(3 * Va) * np.array([[1, Va / (Lw * np.sqrt(3))]])
        w_den = np.sqrt(np.pi * Lw) * np.array([[1, 2 * Va / Lw, (Va / Lw) ** 2]])

        self.w_w = transferFunction(num=w_num, den=w_den, Ts=Ts)
        self._Ts = Ts

    def update(self):
        # returns a six vector.
        #   The first three elements are the steady state wind in the inertial frame
        #   The second three elements are the gust in the body frame
        gust = 2*np.array([[self.u_w.update(np.random.randn())],
                         [self.v_w.update(np.random.randn())],
                         [self.w_w.update(np.random.randn())]])
        return np.concatenate(( self._steady_state, gust ))

