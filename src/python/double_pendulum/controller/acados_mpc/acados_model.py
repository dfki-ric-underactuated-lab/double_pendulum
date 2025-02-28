import os
from acados_template.acados_model import AcadosModel
from acados_template.acados_ocp import AcadosOcp
from acados_template.acados_ocp_solver import AcadosOcpSolver
from acados_template.acados_sim_solver import AcadosSimSolver
import casadi as cas
import numpy as np
import pandas as pd
from double_pendulum.controller.abstract_controller import AbstractController
import yaml
  
class PendulumModel:
    
    model = AcadosModel()
    
    th1   = cas.SX.sym('th1')
    th2   = cas.SX.sym('th2')
    thd1  = cas.SX.sym('thd1')
    thd2  = cas.SX.sym('thd2')

    th = cas.vertcat(th1, th2)
    thd = cas.vertcat(thd1, thd2)
    
    #shape(4,)
    x = cas.vertcat(th, thd)
    xdot = cas.SX.sym('xdot', x.shape)
    
    #Torque Input
    u1 = cas.SX.sym('u1')
    u2 = cas.SX.sym('u2')

    #shape(2,)
    u = cas.vertcat(u1, u2)
    
    x_labels = [r'$\theta$1 [rad]', r'$\theta$2 [rad]',  r'$\dot{\theta}1$ [rad/s]', r'$\dot{\theta}2$ [rad/s]']
    u_labels = ['u1', '$F$2']
    t_label = '$t$ [s]'
    
    def __init__(
        self,
        mass=[1.0, 1.0],
        length=[0.5, 0.5],
        inertia = [None,None],
        center_of_mass=[0.5, 0.5],
        damping=[0.5, 0.5],
        coulomb_fric=[0.0, 0.0],
        gravity=9.81,
        motor_intertia=0.0,
        gear_ratio=6,
        actuated_joint=0,
        ):
        
        model_name = 'pendulum_ode'

        self.m1 = mass[0] # mass of the ball [kg]
        self.m2 = mass[1] # mass of the ball [kg]
        self.l1 = length[0] # length of the rod [m]
        self.l2 = length[1] # length of the rod [m]
        self.I1 = inertia[0] if inertia[0] else self.m1*self.l1**2   #inertia
        self.I2 = inertia[1] if inertia[1] else self.m2*self.l2**2   #inertia
        self.r1 = center_of_mass[0] 
        self.r2 = center_of_mass[1]
        self.g = gravity # gravity constant [m/s^2]
        self.Ir = motor_intertia # motor inertia
        self.gr = gear_ratio # gear ratio
        self.b1 = damping[0]
        self.b2 = damping[1]
        self.cf1 = coulomb_fric[0]
        self.cf2 = coulomb_fric[1]
            
        if actuated_joint == 0:
            self.B = cas.SX([[1,0], [0,0]])  
        elif actuated_joint == 1:
            self.B = cas.SX([[0,0], [0,1]]) 
        else:
            self.B = cas.SX([[1,0], [0,1]])  

        #mass matrix
        #shape (2,2)
        self.M = cas.blockcat(
            self.I1 + self.I2 + self.m2*self.l1**2 + 2*self.m2*self.l1*self.r2*cas.cos(self.th2) + self.gr**2.0 * self.Ir + self.Ir, 
            self.I2 + self.m2*self.l1*self.r2*cas.cos(self.th2) - self.gr * self.Ir, 
            self.I2 + self.m2*self.l1*self.r2*cas.cos(self.th2) - self.gr * self.Ir,
            self.I2 + self.gr**2.0 * self.Ir  
        )

        self.inv_M = cas.solve(self.M, cas.SX.eye(self.M.size1()))

        #Coriolis Matrix
        #shape (2,2)
        self.C = cas.blockcat(
            -2*self.m2*self.l1*self.r2*cas.sin(self.th2)*self.thd2,
            -self.m2*self.l1*self.r2*cas.sin(self.th2)*self.thd2,
            self.m2*self.l1*self.r2*cas.sin(self.th2)*self.thd1,
            0
        )
        
        #self.gravity matrix
        #shape (2,1) 
        self.tau = cas.vertcat(
            -self.m1*self.g*self.r1*cas.sin(self.th1) -self.m2*self.g*(self.l1*cas.sin(self.th1)+self.r2*cas.sin(self.th1+self.th2)),
            -self.m2*self.g*self.r2*cas.sin(self.th1+self.th2)
        )
        
        #Coulomb Vector 
        #shape (2,1)
        self.F = cas.vertcat(
            self.b1*self.thd1 + self.cf1*cas.atan(100*self.thd1),
            self.b2*self.thd2 + self.cf2*cas.atan(100*self.thd2)
        )
        
        self.eom = self.inv_M@(self.B@self.u - self.C@self.thd + self.tau - self.F)
        
        # state space form xdot = [qdot, M**-1[tau + Bu - C qdot]]
        f_expl  = cas.vertcat(
            self.thd,
            self.eom
        )

        f_impl = self.xdot - f_expl

        self.model.f_impl_expr = f_impl
        self.model.f_expl_expr = f_expl
        self.model.x = self.x
        self.model.xdot = self.xdot
        self.model.u = self.u
        self.model.name = "double_pendulum"
        
    # for later use with own integrator
    def forward_dynamics(self,x, u):
        bu = self.B@self.u 
        rest = (self.C@self.thd) + self.tau -self.F
        fun = cas.Function('fun',[self.x, self.u],[self.eom]);
        force = cas.Function('Bu',[self.x, self.u],[bu]);
        minv = cas.Function('rest',[self.x, self.u],[rest]);
        acc = fun(x,u)
        return acc#
    
    def rhs(self,t,x,u):
        accn = self.forward_dynamics(x, u)

        # Next state
        res = np.zeros(2 * 2)
        res[0] = x[2]
        res[1] = x[3]
        res[2] = accn[0]
        res[3] = accn[1]
        
        return res
    
    # for better verbosity
    def mass_matrix(self, x):
        fun = cas.Function('mass_matrix',[self.x],[self.M]);
        m = fun(x)
        return m
    
    def coriolis_matrix(self, x):
        fun = cas.Function('coriolis_matrix',[self.x],[self.C]);
        c = fun(x)
        return c
    
    def gravity_matrix(self, x):
        fun = cas.Function('gravity_matrix',[self.x],[self.tau]);
        t = fun(x)
        return t
    
    def coulomb_vector(self, x):
        fun = cas.Function('coulomb_vector',[self.x],[self.F]);
        f = fun(x)
        return f
            
    def acados_model(self):
        return self.model

            
