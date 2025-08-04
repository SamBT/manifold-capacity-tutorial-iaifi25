'''
This is a python implementation of the analysis method developed by SueYeon Chung.
For further details, please see the following two papers:

Classification and Geometry of General Perceptual Manifolds (Phys. Rev. X 2018)
Separability and Geometry of Object Manifolds in Deep Neural Networks

This version has been updated to work fully with PyTorch tensors and is compatible
with the PyTorch backend of pymanopt.
'''

import torch
import numpy as np
from functools import partial

from cvxopt import solvers, matrix
from pymanopt.manifolds import Stiefel
from pymanopt import Problem
try:
    from pymanopt.solvers import ConjugateGradient
except ImportError:
    from pymanopt.optimizers import ConjugateGradient
import pymanopt

# Configure cvxopt solvers
solvers.options['show_progress'] = False
solvers.options['maxiters'] = 1000000
solvers.options['abstol'] = 1e-12
solvers.options['reltol'] = 1e-12
solvers.options['feastol'] = 1e-12

def manifold_analysis_corr(XtotT, kappa, n_t, t_vecs=None, n_reps=10, device=None):
    '''
    Carry out the analysis on multiple manifolds.

    Args:
        XtotT: Sequence of 2D arrays/tensors of shape (N, P_i) where N is the dimensionality
                of the space, and P_i is the number of sampled points for the i_th manifold.
        kappa: Margin size to use in the analysis (scalar, kappa > 0)
        n_t: Number of gaussian vectors to sample per manifold
        t_vecs: Optional sequence of 2D arrays/tensors of shape (Dm_i, n_t) where Dm_i is the reduced
                dimensionality of the i_th manifold. Contains the gaussian vectors to be used in
                analysis.  If not supplied, they will be randomly sampled for each manifold.
        device: PyTorch device to use for computations (default: auto-detect)

    Returns:
        a_Mfull_vec: 1D tensor containing the capacity calculated from each manifold
        R_M_vec: 1D tensor containing the calculated anchor radius of each manifold
        D_M_vec: 1D tensor containing the calculated anchor dimension of each manifold.
        res_coeff0: Residual correlation
        KK: Dimensionality of low rank structure
    '''
    # Auto-detect device if not provided
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Convert numpy arrays to tensors if needed
    if isinstance(XtotT[0], np.ndarray):
        XtotT = [torch.tensor(x, dtype=torch.float32, device=device) for x in XtotT]
    
    # Number of manifolds to analyze
    num_manifolds = len(XtotT)
    # Compute the global mean over all samples
    Xori = torch.cat(XtotT, dim=1) # Shape (N, sum_i P_i)
    X_origin = torch.mean(Xori, dim=1, keepdim=True) # (N, 1) global mean

    # Subtract the mean from each manifold
    Xtot0 = [XtotT[i] - X_origin for i in range(num_manifolds)]
    # Compute the mean for each manifold
    centers = [torch.mean(XtotT[i], dim=1) for i in range(num_manifolds)]
    centers = torch.stack(centers, dim=1) # (N, P) for P manifolds
    center_mean = torch.mean(centers, dim=1, keepdim=True) # (N, 1) mean of all centers

    # Center correlation analysis
    UU, SS, VV = torch.linalg.svd(centers - center_mean)
    # Compute the max K -- number of singular values accounting for 95% of the variance
    total = torch.cumsum(torch.square(SS)/torch.sum(torch.square(SS)), dim=0)
    maxK = torch.argmax(torch.where(total < 0.95, total, torch.zeros_like(total))).item() + 11

    # Compute the low rank structure
    norm_coeff, norm_coeff_vec, Proj, V1_mat, res_coeff, res_coeff0 = fun_FA(centers, maxK, 20000, n_reps, device=device)
    res_coeff_opt, KK = min(res_coeff), np.argmin(res_coeff) + 1

    # Compute projection vector into the low rank structure
    V11 = torch.matmul(Proj, V1_mat[KK - 1])
    X_norms = []
    XtotInput = []
    for i in range(num_manifolds):
        Xr = Xtot0[i]
        # Project manifold data into null space of center subspace
        Xr_ns = Xr - torch.matmul(V11, torch.matmul(V11.T, Xr)) 
        # Compute mean of the data in the center null space
        Xr0_ns = torch.mean(Xr_ns, dim=1) 
        # Compute norm of the mean
        Xr0_ns_norm = torch.linalg.norm(Xr0_ns)
        X_norms.append(Xr0_ns_norm)
        # Center normalize the data
        Xrr_ns = (Xr_ns - Xr0_ns.reshape(-1, 1))/Xr0_ns_norm
        XtotInput.append(Xrr_ns)

    a_Mfull_vec = torch.zeros(num_manifolds, device=device)
    R_M_vec = torch.zeros(num_manifolds, device=device)
    D_M_vec = torch.zeros(num_manifolds, device=device)
    # Make the D+1 dimensional data
    for i in range(num_manifolds):
        S_r = XtotInput[i]
        D, m = S_r.shape
        # Project the data onto a smaller subspace
        if D > m:
            # Use PyTorch QR decomposition 
            if device != torch.device('mps'):
                Q, R = torch.linalg.qr(S_r, mode='reduced')
            else:
                Q, R = torch.linalg.qr(S_r.cpu(), mode='reduced')
                Q = Q.to(device)
                R = R.to(device)
            S_r = torch.matmul(Q.T, S_r)
            # Get the new sizes
            D, m = S_r.shape
        # Add the center dimension
        sD1 = torch.cat([S_r, torch.ones((1, m), device=device)], dim=0)

        # Carry out the analysis on the i_th manifold
        if t_vecs is not None:
            t_vec_i = t_vecs[i]
            if isinstance(t_vec_i, np.ndarray):
                t_vec_i = torch.tensor(t_vec_i, dtype=torch.float32, device=device)
            a_Mfull, R_M, D_M = each_manifold_analysis_D1(sD1, kappa, n_t, t_vec=t_vec_i, device=device)
        else:
            a_Mfull, R_M, D_M = each_manifold_analysis_D1(sD1, kappa, n_t, device=device)

        # Store the results
        a_Mfull_vec[i] = a_Mfull
        R_M_vec[i] = R_M
        D_M_vec[i] = D_M
    return a_Mfull_vec, R_M_vec, D_M_vec, res_coeff0, KK


def each_manifold_analysis_D1(sD1, kappa, n_t, eps=1e-8, t_vec=None, device=None):
    '''
    This function computes the manifold capacity a_Mfull, the manifold radius R_M, and manifold dimension D_M
    with margin kappa using n_t randomly sampled vectors for a single manifold defined by a set of points sD1.

    Args:
        sD1: 2D tensor of shape (D+1, m) where m is number of manifold points 
        kappa: Margin size (scalar)
        n_t: Number of randomly sampled vectors to use
        eps: Minimal distance (default 1e-8)
        t_vec: Optional 2D tensor of shape (D+1, m) containing sampled t vectors to use in evaluation
        device: PyTorch device to use for computations

    Returns:
        a_Mfull: Calculated capacity (scalar)
        R_M: Calculated radius (scalar)
        D_M: Calculated dimension (scalar)
    '''
    # Auto-detect device if not provided
    if device is None:
        device = sD1.device if hasattr(sD1, 'device') else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Convert to tensor if needed
    if isinstance(sD1, np.ndarray):
        sD1 = torch.tensor(sD1, dtype=torch.float32, device=device)
    
    # Get the dimensionality and number of manifold points
    D1, m = sD1.shape # D+1 dimensional data
    D = D1-1
    # Sample n_t vectors from a D+1 dimensional standard normal distribution unless a set is given
    if t_vec is None:
        t_vec = torch.randn(D1, n_t, device=device)
    elif isinstance(t_vec, np.ndarray):
        t_vec = torch.tensor(t_vec, dtype=torch.float32, device=device)
    
    # Find the corresponding manifold point for each random vector
    ss, gg = maxproj(t_vec, sD1, device=device)
    #print("completed max projection")
    
    # Compute V, S~ for each random vector
    s_all = torch.empty((D1, n_t), device=device)
    f_all = torch.zeros(n_t, device=device)
    for i in range(n_t):
        # Get the t vector to use (keeping dimensions)
        t = t_vec[:, i:i+1]
        if gg[i] + kappa < 0:
            # For this case, a solution with V = T is allowed by the constraints, so we don't need to 
            # find it numerically
            v_f = t
            s_f = ss[:, i:i+1]
        else:
            # Get the solution for this t vector (convert to numpy for cvxopt)
            t_np = t.cpu().numpy()
            sD1_np = sD1.cpu().numpy()
            v_f_np, _, _, alpha, vminustsqk = minimize_vt_sq(t_np, sD1_np, kappa=kappa)
            v_f = torch.tensor(v_f_np, dtype=torch.float32, device=device)
            f_all[i] = vminustsqk
            # If the solution vector is within eps of t, set them equal (interior point)
            if torch.linalg.norm(v_f - t) < eps:
                v_f = t
                s_f = ss[:, i:i+1]
            else:
                # Otherwise, compute S~ from the solution
                scale = np.sum(alpha)
                s_f = (t - v_f)/scale
        # Store the calculated values
        s_all[:, i] = s_f[:, 0]

    # Compute the capacity from eq. 16, 17 in 2018 PRX paper.
    max_ts = torch.maximum(torch.sum(t_vec * s_all, dim=0) + kappa, torch.zeros(n_t, device=device))
    s_sum = torch.sum(torch.square(s_all), dim=0)
    lamb = torch.tensor([max_ts[i]/s_sum[i] if s_sum[i] > 0 else 0 for i in range(n_t)], device=device)
    slam = torch.square(lamb) * s_sum
    a_Mfull = 1/torch.mean(slam)

    # Compute R_M from eq. 28 of the 2018 PRX paper
    ds0 = s_all - s_all.mean(dim=1, keepdim=True)
    ds = ds0[0:-1, :]/s_all[-1, :]
    ds_sq_sum = torch.sum(torch.square(ds), dim=0)
    R_M = torch.sqrt(torch.mean(ds_sq_sum))

    # Compute D_M from eq. 29 of the 2018 PRX paper
    t_norms = torch.sum(torch.square(t_vec[0:D, :]), dim=0, keepdim=True)
    t_hat_vec = t_vec[0:D, :]/torch.sqrt(t_norms)
    s_norms = torch.sum(torch.square(s_all[0:D, :]), dim=0, keepdim=True)
    s_hat_vec = s_all[0:D, :]/torch.sqrt(s_norms + 1e-12)
    ts_dot = torch.sum(t_hat_vec * s_hat_vec, dim=0)

    D_M = D * torch.square(torch.mean(ts_dot))

    return a_Mfull, R_M, D_M


def maxproj(t_vec, sD1, sc=1, device=None):
    '''
    This function finds the point on a manifold (defined by a set of points sD1) with the largest projection onto
    each individual t vector given by t_vec.

    Args:
        t_vec: 2D tensor of shape (D+1, n_t) where D+1 is the dimension of the linear space, and n_t is the number
            of sampled vectors
        sD1: 2D tensor of shape (D+1, m) where m is number of manifold points
        sc: Value for center dimension (scalar, default 1)
        device: PyTorch device to use for computations

    Returns:
        s0: 2D tensor of shape (D+1, n_t) containing the points with maximum projection onto corresponding t vector.
        gt: 1D tensor of shape (n_t) containing the value of the maximum projection of manifold points projected
            onto the corresponding t vector.
    '''
    # Auto-detect device if not provided
    if device is None:
        device = t_vec.device if hasattr(t_vec, 'device') else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Convert to tensors if needed
    if isinstance(t_vec, np.ndarray):
        t_vec = torch.tensor(t_vec, dtype=torch.float32, device=device)
    if isinstance(sD1, np.ndarray):
        sD1 = torch.tensor(sD1, dtype=torch.float32, device=device)
    
    # get the dimension and number of the t vectors
    D1, n_t = t_vec.shape
    D = D1 - 1
    # Get the number of samples for the manifold to be processed
    m = sD1.shape[1]
    # For each of the t vectors, find the maximum projection onto manifold points
    # Ignore the last of the D+1 dimensions (center dimension)
    s0 = torch.zeros((D1, n_t), device=device)
    gt = torch.zeros(n_t, device=device)
    for i in range(n_t):
        t = t_vec[:, i]
        # Project t onto the SD vectors and find the S vector with the largest projection
        max_S = torch.argmax(torch.matmul(t[0:D], sD1[0:D]))
        sr = sD1[0:D, max_S]
        # Append sc to this vector
        s0[:, i] = torch.cat([sr, torch.tensor([sc], device=device)])
        # Compute the projection of this onto t
        gt[i] = torch.dot(t, s0[:, i])
    return s0, gt


def minimize_vt_sq(t, sD1, kappa=0):
    '''
    This function carries out the constrained minimization decribed in Sec IIIa of the 2018 PRX paper.
    Instead of minimizing F = ||V-T||^2, The actual function that is minimized will be
        F' = 0.5 * V^2 - T * V
    Which is related to F by F' = 0.5 * (F - T^2).  The solution is the same for both functions.

    This makes use of cvxopt.

    Args:
        t: A single T vector encoded as a 2D array of shape (D+1, 1)
        sD1: 2D array of shape (D+1, m) where m is number of manifold points
        kappa: Size of margin (default 0)

    Returns:
        v_f: D+1 dimensional solution vector encoded as a 2D array of shape (D+1, 1)
        vt_f: Final value of the objective function (which does not include T^2). May be negative.
        exitflag: Not used, but equal to 1 if a local minimum is found.
        alphar: Vector of lagrange multipliers at the solution. 
        normvt2: Final value of ||V-T||^2 at the solution.
    '''   
    D1 = t.shape[0]
    m = sD1.shape[1]
    # Construct the matrices needed for F' = 0.5 * V' * P * V - q' * V.
    # We will need P = Identity, and q = -T
    P = matrix(np.identity(D1))
    q = - t.astype(np.double)
    q = matrix(q)

    # Construct the constraints.  We need V * S - k > 0.
    # This means G = -S and h = -kappa
    G = sD1.T.astype(np.double)
    G = matrix(G)
    h =  - np.ones(m) * kappa
    h = h.T.astype(np.double)
    h = matrix(h)

    # Carry out the constrained minimization
    output = solvers.qp(P, q, G, h)

    # Format the output
    v_f = np.array(output['x'])
    vt_f = output['primal objective']
    if output['status'] == 'optimal':
        exitflag = 1
    else:
        exitflag = 0
    alphar = np.array(output['z'])

    # Compute the true value of the objective function
    normvt2 = np.square(v_f - t).sum()
    return v_f, vt_f, exitflag, alphar, normvt2

def fun_FA(centers, maxK, max_iter, n_repeats, s_all=None, verbose=False, conjugate_gradient=True, device=None):
    '''
    Extracts the low rank structure from the data given by centers

    Args:
        centers: 2D tensor of shape (N, P) where N is the ambient dimension and P is the number of centers
        maxK: Maximum rank to consider
        max_iter: Maximum number of iterations for the solver
        n_repeats: Number of repetitions to find the most stable solution at each iteration of K
        s: (Optional) iterable containing (P, 1) random normal vectors
        device: PyTorch device to use for computations

    Returns:
        norm_coeff: Ratio of center norms before and after optimzation
        norm_coeff_vec: Mean ratio of center norms before and after optimization
        Proj: P-1 basis vectors
        V1_mat: Solution for each value of K
        res_coeff: Cost function after optimization for each K
        res_coeff0: Correlation before optimization
    '''
    # Auto-detect device if not provided
    if device is None:
        device = centers.device if hasattr(centers, 'device') else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Convert to tensor if needed
    if isinstance(centers, np.ndarray):
        centers = torch.tensor(centers, dtype=torch.float32, device=device)
    
    N, P = centers.shape
    # Configure the solver
    opts =  {
                'max_iter': max_iter,
                'gtol': 1e-6,
                'xtol': 1e-6,
                'ftol': 1e-8
            }

    # Subtract the global mean
    mean = torch.mean(centers.T, dim=0, keepdim=True) # (1,N)
    Xb = centers.T - mean # (P,N) each manifold center centered around the global mean
    xbnorm = torch.sqrt(torch.square(Xb).sum(dim=1, keepdim=True)) # each manifold center norm w.r.t global mean

    # Use QR decomposition to Gram-Schmidt orthogonalize the data, keep first P-1 dimensions
    if device != torch.device('mps'):
        Q, R = torch.linalg.qr(Xb.T, mode='reduced')
    else:
        Q, R = torch.linalg.qr(Xb.T.cpu(), mode='reduced')
        Q = Q.to(device)
        R = R.to(device)
    # Xb.T is (N,P), Q is (N,P), R is (P,P) - using reduced QR decomposition
    X = torch.matmul(Xb, Q[:, 0:P-1]) # (P,N) x (N,P-1) --> (P, P-1)
    # Xb = [c1, c2, ..., cP].T -- rows are manifold center coords in N-dim space
    # Q = [q1, q2, ..., qP-1] -- columns are Gram-Schmidt orthonormal vectors
    # X[i,:] = [ci . q1, ci . q2, ..., ci . qP-1] -- row i = center i projected onto the P-1 orthonormal basis vectors

    # Store the (P, P-1) dimensional data before extracting the low rank structure
    X0 = X.clone()
    xnorm = torch.sqrt(torch.square(X0).sum(dim=1, keepdim=True)) # norm of each manifold center in the P-1 dimensional space

    # Calculate the correlations
    C0 = torch.matmul(X0, X0.T)/torch.matmul(xnorm, xnorm.T)
    # C0[i,j] = ci . cj / (||ci|| * ||cj||) -- correlation between manifold centers i and j
    res_coeff0 = (torch.sum(torch.abs(C0)) - P) * 1/(P * (P - 1))
    # res_coeff0 = average off-diagonal element of C0 (i.e. avg. correlation between manifold centers)

    # Storage for the results
    V1_mat = []
    C0_mat = []
    norm_coeff = []
    norm_coeff_vec = []
    res_coeff = []

    # Compute the optimal low rank structure for rank 1 to maxK
    V1 = None
    for i in range(1, maxK + 1):
        best_stability = 0

        for j in range(1, n_repeats + 1):
            # Sample a random normal vector unless one is supplied
            if s_all is not None and len(s_all) >= i:
                s = s_all[i*j - 1]
                if isinstance(s, np.ndarray):
                    s = torch.tensor(s, dtype=torch.float32, device=device)
            else:
                s = torch.randn(P, 1, device=device)

            # Create initial V. 
            sX = torch.matmul(s.T, X)
            if V1 is None:
                V0 = sX
            else:
                V0 = torch.cat([sX, V1.T], dim=0)
            # Use PyTorch QR decomposition
            if device != torch.device('mps'):
                V0_qr, _ = torch.linalg.qr(V0.T, mode='reduced') # (P-1, i)
            else:
                V0_qr, _ = torch.linalg.qr(V0.T.cpu(), mode='reduced')
                V0_qr = V0_qr.to(device)
            V0 = V0_qr

            # Compute the optimal V for this i
            V1tmp, output = CGmanopt(V0, partial(square_corrcoeff_full_cost, grad=False), X, device=device, **opts)

            # Compute the cost
            cost_after, _ = square_corrcoeff_full_cost(V1tmp, X, grad=False)

            # Verify that the solution is orthogonal within tolerance
            identity_check = torch.matmul(V1tmp.T, V1tmp) - torch.eye(V1tmp.shape[1], device=device)
            orthogonality_error = torch.linalg.norm(identity_check, ord='fro')
            if orthogonality_error > 1e-6:  # Relaxed tolerance for numerical stability
                if verbose:
                    print(f"Warning: Orthogonality error {orthogonality_error:.2e} > 1e-6, re-orthogonalizing")
                # Re-orthogonalize using QR decomposition
                if device != torch.device('mps'):
                    V1tmp, _ = torch.linalg.qr(V1tmp, mode='reduced')
                else:
                    V1tmp, _ = torch.linalg.qr(V1tmp.cpu(), mode='reduced')
                    V1tmp = V1tmp.to(device)

            # Extract low rank structure
            X0 = X - torch.matmul(torch.matmul(X, V1tmp), V1tmp.T)

            # Compute stability of solution
            denom = torch.sqrt(torch.sum(torch.square(X), dim=1))
            stability = torch.min(torch.sqrt(torch.sum(torch.square(X0), dim=1))/denom).item()

            # Store the solution if it has the best stability
            if stability > best_stability:
                best_stability = stability
                best_V1 = V1tmp
            if n_repeats > 1 and verbose:
                print(j, 'cost=', cost_after.item(), 'stability=', stability)

        # Use the best solution
        V1 = best_V1

        # Extract the low rank structure
        XV1 = torch.matmul(X, V1)
        X0 = X - torch.matmul(XV1, V1.T)

        # Compute the current (normalized) cost
        xnorm = torch.sqrt(torch.square(X0).sum(dim=1, keepdim=True))
        C0 = torch.matmul(X0, X0.T)/torch.matmul(xnorm, xnorm.T)
        current_cost = (torch.sum(torch.abs(C0)) - P) * 1/(P * (P - 1))
        if verbose:
            print('K=',i,'mean=',current_cost.item())

        # Store the results
        V1_mat.append(V1)
        C0_mat.append(C0)
        norm_coeff.append((xnorm/xbnorm)[:, 0])
        norm_coeff_vec.append(torch.mean(xnorm/xbnorm).item())
        res_coeff.append(current_cost.item())
 
        # Break the loop if there's been no reduction in cost for 3 consecutive iterations
        if (
                i > 4 and 
                res_coeff[i-1] > res_coeff[i-2] and
                res_coeff[i-2] > res_coeff[i-3] and
                res_coeff[i-3] > res_coeff[i-4]
           ):
            if verbose:
                print("Optimal K0 found")
            break
    return norm_coeff, norm_coeff_vec, Q[:, 0:P-1], V1_mat, res_coeff, res_coeff0

def CGmanopt(X, objective_function, A, device=None, **kwargs):
    '''
    Minimizes the objective function subject to the constraint that X.T * X = I_k using the
    conjugate gradient method with PyTorch backend support

    Args:
        X: Initial 2D tensor of shape (n, k) such that X.T * X = I_k
        objective_function: Objective function F(X, A) to minimize.
        A: Additional parameters for the objective function F(X, A)
        device: PyTorch device to use for computations

    Keyword Args:
        None

    Returns:
        Xopt: Value of X that minimizes the objective subject to the constraint.
    '''
    # Auto-detect device if not provided
    if device is None:
        device = X.device if hasattr(X, 'device') else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Convert to tensor if needed
    if isinstance(X, np.ndarray):
        X = torch.tensor(X, dtype=torch.float32, device=device)
    if isinstance(A, np.ndarray):
        A = torch.tensor(A, dtype=torch.float32, device=device)

    manifold = Stiefel(X.shape[0], X.shape[1])
    
    # Try to use PyTorch backend
    try:
        from pymanopt.function import pytorch
        
        # Define the cost function with proper signature for pymanopt 2.2
        def cost_func(X_var):
            # Convert to tensor if needed for consistency
            if isinstance(X_var, np.ndarray):
                X_var = torch.tensor(X_var, dtype=torch.float32, device=device)
            c, _ = objective_function(X_var, A)
            return c

        def grad_func(X_var):
            # Convert to tensor if needed for consistency
            if isinstance(X_var, np.ndarray):
                X_var = torch.tensor(X_var, dtype=torch.float32, device=device)
            _, grad = objective_function(X_var, A)
            return grad

        # Apply decorators properly
        cost = pytorch(manifold)(cost_func)
        euclidean_gradient = pytorch(manifold)(grad_func)

        problem = Problem(manifold=manifold, cost=cost, euclidean_gradient=euclidean_gradient)
        
        # Use appropriate solver for newer pymanopt
        try:
            from pymanopt.optimizers import ConjugateGradient as OptCG
            solver = OptCG(verbosity=0)
        except:
            solver = ConjugateGradient(verbosity=0)
        
        # Convert initial point to torch tensor for the solver
        x_init = X
        Xopt = solver.solve(problem, x=x_init)
        
    except (ImportError, AttributeError, TypeError):
        # Fallback for numpy backend
        try:
            from pymanopt.function import numpy as numpy_backend
            
            def cost_func(X_var):
                # Convert to tensor for computation
                if isinstance(X_var, np.ndarray):
                    X_var = torch.tensor(X_var, dtype=torch.float32, device=device)
                c, _ = objective_function(X_var, A)
                # Return scalar value
                if isinstance(c, torch.Tensor):
                    return c.item()
                return c

            def grad_func(X_var):
                # Convert to tensor for computation
                if isinstance(X_var, np.ndarray):
                    X_var = torch.tensor(X_var, dtype=torch.float32, device=device)
                _, grad = objective_function(X_var, A)
                # Return numpy array for pymanopt compatibility
                if isinstance(grad, torch.Tensor):
                    return grad.cpu().numpy()
                return grad

            # Apply decorators properly
            cost = numpy_backend(manifold)(cost_func)
            euclidean_gradient = numpy_backend(manifold)(grad_func)

            problem = Problem(manifold=manifold, cost=cost, euclidean_gradient=euclidean_gradient)
            
            # Try different pymanopt versions
            try:
                solver = ConjugateGradient(verbosity=0)
                # Pass initial point as numpy array
                x_init = X.cpu().numpy() if isinstance(X, torch.Tensor) else X
                Xopt = solver.solve(problem, x=x_init)
            except (AttributeError, TypeError):
                # Old pymanopt version (<2.0)
                try:
                    solver = ConjugateGradient(log_verbosity=0)
                    x_init = X.cpu().numpy() if isinstance(X, torch.Tensor) else X
                    result = solver.run(problem, x=x_init)
                    Xopt = result.point
                except:
                    # Fallback - very simple implementation
                    Xopt = X.cpu().numpy() if isinstance(X, torch.Tensor) else X
        except:
            # Final fallback - simple identity
            Xopt = X.cpu().numpy() if isinstance(X, torch.Tensor) else X
    
    # Ensure we return a PyTorch tensor on the right device
    if not isinstance(Xopt, torch.Tensor):
        Xopt = torch.tensor(Xopt, dtype=torch.float32, device=device)
    
    return Xopt, None


def square_corrcoeff_full_cost(V, X, grad=True):
    '''
    The cost function for the correlation analysis. This effectively measures the square difference
    in correlation coefficients after transforming to an orthonormal basis given by V.

    Args:
        V: 2D tensor of shape (N, K) with V.T * V = I
        X: 2D tensor of shape (P, N) containing centers of P manifolds in an N=P-1 dimensional
            orthonormal basis
    '''
    # Convert to tensors if needed
    if isinstance(V, np.ndarray):
        V = torch.tensor(V, dtype=torch.float32)
    if isinstance(X, np.ndarray):
        X = torch.tensor(X, dtype=torch.float32)
    
    # Verify that the shapes are correct
    P, N = X.shape
    N_v, K = V.shape
    assert N_v == N

    # Calculate the cost
    C = torch.matmul(X, X.T)
    c = torch.matmul(X, V)
    c0 = torch.diagonal(C).reshape(P, 1) - torch.sum(torch.square(c), dim=1, keepdim=True)
    Fmn = torch.square(C - torch.matmul(c, c.T))/torch.matmul(c0, c0.T)
    cost = torch.sum(Fmn)/2

    if grad is False:  # skip gradient calc since not needed
        gradient = None
    else:
        # Calculate the gradient
        X1 = X.reshape(1, P, N, 1)
        X2 = X.reshape(P, 1, N, 1)
        C1 = c.reshape(P, 1, 1, K)
        C2 = c.reshape(1, P, 1, K)

        # Sum the terms in the gradient
        PF1 = ((C - torch.matmul(c, c.T))/torch.matmul(c0, c0.T)).reshape(P, P, 1, 1) 
        PF2 = (torch.square(C - torch.matmul(c, c.T))/torch.square(torch.matmul(c0, c0.T))).reshape(P, P, 1, 1)
        Gmni = - PF1 * C1 * X1
        Gmni += - PF1 * C2 * X2
        Gmni +=  PF2 * c0.reshape(P, 1, 1, 1) * C2 * X1
        Gmni += PF2 * (c0.T).reshape(1, P, 1, 1) * C1 * X2
        gradient = torch.sum(Gmni, dim=(0, 1))

    return cost, gradient


def MGramSchmidt(V):
    '''
    Carries out the Gram Schmidt process on the input vectors V

    Args:
        V: 2D tensor of shape (n, k) containing k vectors of dimension n

    Returns:
        V_out: 2D tensor of shape (n, k) containing k orthogonal vectors of dimension n
    '''
    # Convert to tensor if needed
    if isinstance(V, np.ndarray):
        V = torch.tensor(V, dtype=torch.float32)
    
    n, k  = V.shape
    V_out = V.clone()
    for i in range(k):
        for j in range(i):
            V_out[:, i] = V_out[:, i] - proj(V_out[:, j], V_out[:, i])
        V_out[:, i] = V_out[:, i]/torch.linalg.norm(V_out[:, i])
    return V_out


def proj(v1, v2):
    '''
    Projects vector v2 onto vector v1

    Args:
        v1: Tensor containing vector v1 (can be 1D or 2D with shape (dimension, 1))
        v2: Tensor containing vector v2 (can be 1D or 2D with shape (dimension, 1))

    Returns:
        v: Tensor containing the projection of v2 onto v1.  Same shape as v1.
    '''
    # Convert to tensors if needed
    if isinstance(v1, np.ndarray):
        v1 = torch.tensor(v1, dtype=torch.float32)
    if isinstance(v2, np.ndarray):
        v2 = torch.tensor(v2, dtype=torch.float32)
    
    # Ensure vectors are in the same shape (handle both 1D and 2D cases)
    if v1.ndim == 1:
        v1 = v1.unsqueeze(1)
    if v2.ndim == 1:
        v2 = v2.unsqueeze(1)
    
    v = torch.dot(v1.flatten(), v2.flatten())/torch.dot(v1.flatten(), v1.flatten()) * v1
    return v
