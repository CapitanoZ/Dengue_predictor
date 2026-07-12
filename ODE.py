import torch
import torch.nn as nn

class SEIREnv_ODE_Engine(nn.Module):
    """
    Upgraded SEIRS + Environment dynamics engine.
    Introduces antibody decay (xi) and adds upper/lower bound clamping to absolutely prevent numerical crashes (NaN).
    """

    def __init__(self, sigma, gamma, omega, tau, xi, cloud_model):
        super(SEIREnv_ODE_Engine, self).__init__()
        self.sigma = sigma  # Transition rate from Exposed (E) to Infectious (I)
        self.gamma = gamma  # Recovery rate from Infectious (I) to Recovered (R)
        self.omega = omega  # Natural decay rate of environmental risk
        self.tau = tau      # Rate of pathogen shedding into the environment by infected individuals
        self.xi = xi        # Antibody decay rate for recovered individuals (R -> S)
        self.cloud_model = cloud_model


    def forward(self, t, state):
        # Extract states, adding C (Cumulative Cases) as state[5]
        S, E, I, R, Env, C = state[0], state[1], state[2], state[3], state[4], state[5]

        # [Safety Clamps] Prevent out-of-bounds values
        S = torch.clamp(S, min=0.0, max=1.0)
        E = torch.clamp(E, min=0.0, max=1.0)
        I = torch.clamp(I, min=0.0, max=1.0)
        R = torch.clamp(R, min=0.0, max=1.0)
        Env = torch.relu(Env)

        beta_H, beta_E, F_theta = self.cloud_model(t)

        dS_dt = -beta_H * S * I - beta_E * S * Env + self.xi * R
        dE_dt = beta_H * S * I + beta_E * S * Env - self.sigma * E
        dI_dt = self.sigma * E - self.gamma * I
        dR_dt = self.gamma * I - self.xi * R
        dEnv_dt = self.tau * I - self.omega * Env + F_theta
        
        # [New]: Calculate the instantaneous rate of daily new cases (effectively the inflow into compartment I)
        dC_dt = self.sigma * E 

        # Return the tensor including the new C state
        return torch.stack([dS_dt, dE_dt, dI_dt, dR_dt, dEnv_dt, dC_dt])