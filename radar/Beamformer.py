import time
# import cupy as np
import numpy as np
from scipy.optimize import newton_krylov
from copy import deepcopy

# import torch

class Beamformer(object):
    def __init__(self, num_steps, num_antennas):
        self.num_steps = num_steps
        self.num_antennas = num_antennas
        self.d_theta = np.pi / num_steps
        self.thetas = (np.arange(0, num_steps) + 0.5) * self.d_theta - np.pi / 2
        self.steering_vector = np.zeros((num_antennas, num_steps), dtype=np.complex128)
        for idx in range(0, num_antennas):
            # for FMCW radar, the steering vector is exp_param_0319_100cm_2(j 2 pi d sin(theta) / c)
            self.steering_vector[idx, :] = np.exp(1j * np.pi * idx * np.sin(self.thetas))
    
    def beamforming_func(self, rxx):
        raise NotImplementedError

    '''
        x:                  num_antennas * num_samples
        bm_weights:         num_antennas * num_steps
        power_spectrum:     num_steps * 1
    '''

    def steering(self, x):
        
        if x.ndim == 2: 
            # [num_antennas, num_samples]
            auto_correlation_matrix = np.matmul(x, np.conjugate(x.transpose(1, 0)))
            auto_correlation_matrix_norm = np.divide(auto_correlation_matrix, \
                np.diag(auto_correlation_matrix)[:, np.newaxis])
            bm_weights = self.beamforming_func(auto_correlation_matrix_norm)
            w = bm_weights
            w_H = np.conj(w).transpose()
            # power_spectrum: [num_angle_bins]
            power_spectrum = np.matmul(np.matmul(w_H, auto_correlation_matrix), w)
            power_spectrum = np.diag(power_spectrum)
            # No need to multiply the auto-correlation matrix
        
        if x.ndim == 3:
            # [num_range_bins, num_antennas, num_samples]
            self.num_range_bins = x.shape[0]
            
            auto_correlation_matrix = np.einsum('mnl,mlk->mnk', x, np.conjugate(x.transpose(0, 2, 1)))
            auto_correlation_matrix_norm = np.divide(auto_correlation_matrix, \
                np.einsum('mii->m', auto_correlation_matrix)[:, np.newaxis, np.newaxis])  
            bm_weights = self.beamforming_func(auto_correlation_matrix_norm) # [num_range_bins, num_antennas, num_angle_steps]
            w = bm_weights
            w_H = np.conj(w).transpose(0, 2, 1) 
            # power_spectrum: [num_range_bins, num_angle_bins]
            power_spectrum = np.einsum('mnl,mlk->mnk', np.einsum('mnl,mlk->mnk', w_H, auto_correlation_matrix), w)
            power_spectrum = np.einsum('mii->mi', power_spectrum)
            power_spectrum *= np.einsum('ijj->i', auto_correlation_matrix, optimize='optimal')[:, np.newaxis]
            
        elif x.ndim == 4: 
            # [num_slow_samples, num_range_bins, num_antennas, num_samples]
            self.num_slow_samples = x.shape[0]
            self.num_range_bins = x.shape[1]
            
            auto_correlation_matrix = np.einsum('xmnl,xmlk->xmnk', x, np.conjugate(x.transpose(0, 1, 3, 2)))
            auto_correlation_matrix_norm = np.divide(auto_correlation_matrix, \
                np.einsum('xmii->xm', auto_correlation_matrix)[:, :, np.newaxis, np.newaxis])
            bm_weights = self.beamforming_func(auto_correlation_matrix_norm)
            w = bm_weights
            w_H = np.conj(w).transpose(0, 1, 3, 2)
            # power_spectrum: [num_slow_samples, num_range_bins, num_angle_bins]
            power_spectrum = np.einsum('xmnl,xmlk->xmnk', np.einsum('xmnl,xmlk->xmnk', w_H, auto_correlation_matrix), w)
            power_spectrum = np.einsum('xmii->xmi', power_spectrum)
            power_spectrum *= np.einsum('xijj->xi', auto_correlation_matrix, optimize='optimal')[:, :, np.newaxis]
            
        return power_spectrum, bm_weights
         
    def update_gain_mat(self, gains):
        # self.gain_mat_guess = np.diag(gains / np.max(gains))    
        self.gain_mat_guess = np.einsum('ij,jk->ijk', gains, np.eye(self.num_antennas))
        self.gain_mat_guess /= np.max(self.gain_mat_guess, axis=(1, 2))[:, np.newaxis, np.newaxis]   # [num_range_bins, num_antennas, num_antennas]
 
class BartlettBeamformer(Beamformer):
    def beamforming_func(self, rxx):
        # return np.tile(self.steering_vector / self.num_antennas, (self.num_range_bins, 1, 1))
        if rxx.ndim == 2:
            return self.steering_vector
        
        elif rxx.ndim == 3: 
            return np.tile(self.steering_vector, (self.num_range_bins, 1, 1))
        
        elif rxx.ndim == 4: 
            return np.tile(self.steering_vector, (self.num_slow_samples, self.num_range_bins, 1, 1))

class CaponBeamformer(Beamformer):
    def beamforming_func(self, rxx):
        if rxx.ndim == 2:
            # Calculate the inverse of the auto-correlation matrix
            rxx_inv = np.linalg.inv(rxx)    # [num_antennas, num_antennas]
            weights = np.zeros((self.num_antennas, self.num_steps), dtype=np.complex128)
            # Create a matrix of steering vectors for all angles
            sv_aoa = self.steering_vector   # [num_antennas, num_steps]
            # Compute the numerator and denominator matrices 
            num_matrix = np.matmul(rxx_inv, sv_aoa) # [num_antennas, num_steps]
            den_matrix = np.matmul(np.matmul(sv_aoa.conj().transpose(1, 0), rxx_inv), sv_aoa)   # [num_steps, num_steps]
            # Use broadcasting to perform element-wise division
            weights = num_matrix / np.diag(den_matrix)[np.newaxis, :]
            return weights
            
        elif rxx.ndim == 3: # [num_range_bins, num_antennas, num_antennas]
            # Calculate the inverse of the auto-correlation matrix
            rxx_inv = np.linalg.inv(rxx)    # [num_range_bins, num_antennas, num_antennas]
            weights = np.zeros((self.num_range_bins, self.num_antennas, self.num_steps), dtype=np.complex128)
            # Create a matrix of steering vectors for all angles
            sv_aoa = np.tile(self.steering_vector, (self.num_range_bins, 1, 1)) # [num_range_bins, num_antennas, num_steps]
            # Compute the numerator and denominator matrices in one go
            num_matrix = np.einsum('ijk,ikl->ijl', rxx_inv, sv_aoa) # [num_range_bins, num_antennas, num_steps] 
            den_matrix = np.einsum('ijk,ikl->ijl', \
                np.einsum('ijk,ikl->ijl', sv_aoa.conj().transpose(0, 2, 1), rxx_inv), sv_aoa)   # [num_range_bins, num_steps, num_steps]
            # Use broadcasting to perform element-wise division
            weights = num_matrix / np.einsum("mii->mi", den_matrix)[: , np.newaxis, :]
            return weights

        elif rxx.ndim == 4:
            # Calculate the inverse of the auto-correlation matrix
            rxx_inv = np.linalg.inv(rxx)    # [num_slow_samples, num_range_bins, num_antennas, num_antennas]
            weights = np.zeros((self.num_slow_samples, self.num_range_bins, self.num_antennas, self.num_steps), dtype=np.complex128)
            # Create a matrix of steering vectors for all angles
            sv_aoa = np.tile(self.steering_vector, (self.num_slow_samples, self.num_range_bins, 1, 1)) # [num_slow_samples, num_range_bins, num_antennas, num_steps]
            # Compute the numerator and denominator matrices in one go
            num_matrix = np.einsum('xijk,xikl->xijl', rxx_inv, sv_aoa) # [num_slow_samples, num_range_bins, num_antennas, num_steps]
            den_matrix = np.einsum('xijk,xikl->xijl', \
                np.einsum('xijk,xikl->xijl', sv_aoa.conj().transpose(0, 1, 3, 2), rxx_inv), sv_aoa)   # [num_slow_samples, num_range_bins, num_steps, num_steps]
            # Use broadcasting to perform element-wise division
            weights = num_matrix / np.einsum("xmii->xmi", den_matrix)[: , :, np.newaxis, :]
            return weights

# class RobustCaponBeamformer(Beamformer):
#     def __init__(self, num_steps, num_antennas, antenna_gain_mat=None, threshold=0.2):
#         super(RobustCaponBeamformer, self).__init__(num_steps, num_antennas)
#         self.gain_mat_guess = antenna_gain_mat
#         self.threshold = np.array(num_antennas * threshold * threshold)
#         self.eye = np.eye(self.num_antennas)

#     def beamforming_func(self, rxx):

#         weights = np.zeros((self.num_range_bins, self.num_antennas, self.num_steps), dtype=np.complex128)
        
#         for range_idx in range(0, self.num_range_bins):
            
#             rxx_norm = rxx[range_idx, :, :] / np.max(np.abs(rxx[range_idx, :, :]))
#             rxx_inv = np.linalg.inv(rxx_norm)      
            
#             # Create a matrix of steering vectors for all angles
#             sv_aoa = self.steering_vector
#             sv_aoa = np.einsum('ii,ik->ik', self.gain_mat_guess[range_idx, :, :], sv_aoa)
#             # Estimate the robust steering vector for all angles
#             sv_robust = self.estimate_robust_steering_vector_torch(rxx_norm, sv_aoa)    # [num_antennas, num_steps]
#             sv_robust_H = sv_robust.conj().T    # [num_steps, num_antennas]
#             # Compute the numerator and denominator matrices in one go
#             num_matrix = np.matmul(rxx_inv, sv_robust)
#             den_matrix = np.matmul(np.matmul(sv_robust_H, rxx_inv), sv_robust)       
#             den_matrix = np.diag(den_matrix) 
#             # Use broadcasting to perform element-wise division
#             weights[range_idx, :, :] = num_matrix / den_matrix
            
#         return weights


#     def estimate_robust_steering_vector(self, rxx_norm, sv_guess):
        
#         import cupy as np
#         rxx_norm = np.asnumpy(rxx_norm)
#         sv_guess = np.asnumpy(sv_guess)
#         eye = np.asnumpy(self.eye)
#         thre = np.asnumpy(self.threshold)

#         import numpy as np
#         def _goal_func(lambda_):
#             lambda_ = np.array(lambda_)
#             tmp1 = np.linalg.inv(lambda_ * rxx_norm + eye)
#             tmp2 = np.matmul(tmp1, sv_guess)
#             tmp3 = np.power(np.linalg.norm(tmp2, ord=2), 2)
#             return tmp3 - thre
        
#         lambda_init = 1
#         lambda_est = newton_krylov(_goal_func, lambda_init, f_tol=1e-6) # numpy return
        
#         sv_cor = np.matmul(np.linalg.inv(lambda_est * rxx_norm + eye), sv_guess)
#         sv_est = sv_guess - sv_cor
        
#         import cupy as np
#         sv_est = np.array(sv_est)
        
#         return sv_est
    
    
#     def estimate_robust_steering_vector_torch(self, rxx_norm, sv_guess):
        
#         rxx_norm = torch.from_numpy(rxx_norm.get()) # [num_antennas, num_antennas]
#         sv_guess = torch.from_numpy(sv_guess.get()) # [num_antennas, num_steps]
#         eye = torch.from_numpy(self.eye.get())  # [num_antennas, num_antennas]
#         thre = torch.from_numpy(self.threshold.get()).repeat(self.num_steps)
        
#         def _goal_func(lambda_est):
#             # Add eye separately for each lambda_i
#             tmp0 = torch.einsum('i,jk->ijk', lambda_est, rxx_norm) + eye    # [num_steps, num_antennas, num_antennas]
#             tmp1 = torch.inverse(tmp0)  # [num_steps, num_antennas, num_antennas]
#             tmp2 = torch.einsum('ijj,ji->ij', tmp1, sv_guess) # [num_steps, num_antennas]
#             tmp3 = torch.norm(tmp2, p=2, dim=1)**2 # [num_steps]
#             return tmp3
        
#         # Optimize lambda with size [num_steps, 1]
#         # Randomly initialize lambda
#         lambda_est = 100 + torch.randn((self.num_steps), requires_grad=True)
#         lambda_est = torch.abs(lambda_est).detach().requires_grad_()
#         optimizer = torch.optim.LBFGS([lambda_est])
        
#         def closure():
#             optimizer.zero_grad()
#             loss = torch.nn.functional.mse_loss(_goal_func(lambda_est), thre)
#             loss.backward()
#             lambda_est.grad = torch.nn.functional.relu(lambda_est.grad)
#             return loss

#         optimizer.step(closure)

#         # print("Optimized loss: ", closure())
        
#         # Calculate the estimated steering vector using the optimized lambda
#         sv_cor = _goal_func(lambda_est)
#         sv_est = sv_guess - sv_cor

#         # Convert the result to NumPy if necessary
#         return np.array(sv_est.detach().numpy())
    
    
if __name__ == "__main__":
    
    import matplotlib.pyplot as plt
    
    slow_time_samples = np.load("./save/slow_time_samples_4.npy")
    
    range_spectrum = np.linalg.norm(slow_time_samples, ord=2, axis=(1, 2)) # [num_range_bins]
      
    
    print("Shape: ", slow_time_samples.shape)
    
    beamformer = CaponBeamformer(num_antennas=8, num_steps=181)
    # beamformer = BartlettBeamformer(num_antennas=8, num_steps=181)
    
    # For Robust Capon Beamforming, we need to estimate the gain matrix first
    # rs = np.abs(slow_time_samples)
    # strengths = np.percentile(rs, 95, axis=-1) 
    # beamformer.update_gain_mat(strengths)
    
    power_spectrums, _ = beamformer.steering(slow_time_samples)
    
    range_spectrum = np.linalg.norm(power_spectrums, ord=2, axis=-1) # [num_range_bins]
    
    plt.plot(range_spectrum.get())
    plt.show()  
    
    plt.imshow(np.abs(power_spectrums).get(), aspect='auto', cmap='jet')
    plt.show()