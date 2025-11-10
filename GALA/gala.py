import pygad
import torch
import numpy as np

from . import utils

# set default dtype
torch.set_default_dtype(torch.float64)
# set defauly device
device = "cuda:0" if torch.cuda.is_available() else "cpu"

last_fitness = 0.0

def gala(xI,I,xJ,J,
          a = 1.0, p = 2.0, expand = 1.0, nt = 3,
          epV = 1.0,
          sigmaM = 1.0, sigmaB = 1.0, sigmaR = 1e3,
          gene_space = None, muB = None, random_seed = None,
          num_parents_mating = 50, sol_per_pop = 100,
          mutation_probability =  0.2,crossover_probability = 0.8,
          num_generations = 500, num_iterations = 2000, num_repeats = 10):
    '''
    Convert data to Tensor
    '''
    I = torch.as_tensor(I,device=device)                         
    J = torch.as_tensor(J,device=device)
    xI = [torch.tensor(x,device=device) for x in xI]
    xJ = [torch.tensor(x,device=device) for x in xJ]
    XJ = torch.stack(torch.meshgrid(*xJ,indexing='ij'),-1)
    
    minv = torch.as_tensor([x[0] for x in xI], device=device)
    maxv = torch.as_tensor([x[-1] for x in xI],device=device)
    minv,maxv = (minv+maxv)*0.5 + 0.5*torch.tensor([-1.0,1.0],device=device)[...,None]*(maxv-minv)*expand
    xv = [torch.arange(m,M,a*0.5,device=device) for m,M in zip(minv,maxv)]
    XV = torch.stack(torch.meshgrid(*xv,indexing = 'ij'),-1)
    v = torch.zeros((nt,XV.shape[0],XV.shape[1],XV.shape[2]),device=device,requires_grad=True)
    
    dv = torch.as_tensor([x[1]-x[0] for x in xv],device=device)
    fv = [torch.arange(n,device=device)/n/d for n,d in zip(XV.shape,dv)]
    FV = torch.stack(torch.meshgrid(*fv,indexing='ij'),-1)
    LL = (1.0 + 2.0*a**2* torch.sum( (1.0 - torch.cos(2.0*np.pi*FV*dv))/dv**2 ,-1))**(p*2.0)
    K = 1.0/LL
    DV = torch.prod(dv)
    
    
    WM = torch.ones(J[0].shape,device= device)*0.5
    WB = torch.ones(J[0].shape,device= device)*0.5
    estimate_muB = muB is None
    ###
    ###  
    ### 
    # Compute default search ranges from data
    rIy = (xI[0].max() - xI[0].min()).item()
    rIx = (xI[1].max() - xI[1].min()).item()
    rJy = (xJ[0].max() - xJ[0].min()).item()
    rJx = (xJ[1].max() - xJ[1].min()).item()
    sx_range = rIx / rJx if rIx > rJx else rJx / rIx  
    sy_range = rIy / rJy if rIy > rJy else rJy / rIy

    default_gene_space = [
        {'low': -180.0, 'high': 180.0},     # angle
        {'low': -(sy_range+ 0.1), 'high': (sy_range+ 0.1)},     # sy
        {'low': -(sx_range+ 0.1), 'high': (sx_range+ 0.1)},     # sx
        {'low': -rJy/2, 'high': rJy/2},     # ty
        {'low': -rJx/2, 'high': rJx/2},     # tx
    ]
    # If user provided partial gene_space, update only specified entries
    if gene_space is not None:
        for i, g in enumerate(gene_space):
            if g:  # only update if user provides a non-empty dict
                g_fixed = g.copy()
                if 'low' not in g_fixed or g_fixed['low'] is None:
                    g_fixed['low'] = default_gene_space[i]['low']
                if 'high' not in g_fixed or g_fixed['high'] is None:
                    g_fixed['high'] = default_gene_space[i]['high']
                default_gene_space[i].update(g_fixed)
        
    # Identify fixed vs variable parameters
    final_gene_space = []
    fixed_params = {}
    variable_indices = []

    for i, g in enumerate(default_gene_space):
        if g['low'] == g['high']:
            fixed_params[i] = g['low']
        else:
            final_gene_space.append(g)
            variable_indices.append(i)

    # Define the fitness function
    def fitness_func(ga, solution, solution_idx):
        full_params = [None] * 5
        var_cursor = 0
        for i in range(5):
            if i in fixed_params:
                full_params[i] = fixed_params[i]
            else:
                full_params[i] = solution[var_cursor]
                var_cursor += 1
        angle, sy, sx, ty, tx = full_params
        
        A = utils.computeA(angle, sy, sx, ty, tx, xI, xJ)
        fAI = utils.transform_image_with_A(A, xJ, xI, I)
        mse = 1 / (1 + torch.sum((fAI - J) ** 2))
        return mse.item()
    
    def on_generation(ga):
        global last_fitness
        if ga.generations_completed % 200 == 0:
            print(f"Generation = {ga.generations_completed}")
            print(f"Fitness    = {ga.best_solution(pop_fitness=ga.last_generation_fitness)[1]}")
            print(f"Change     = {ga.best_solution(pop_fitness=ga.last_generation_fitness)[1] - last_fitness}")
            last_fitness = ga.best_solution(pop_fitness=ga.last_generation_fitness)[1]

    Esave = []
    Aparams = []
    for ite in range(num_repeats):
        print(f'Iteration {ite + 1} of {num_repeats}')
        # ---- Run GA ----
        ga = pygad.GA(
            num_generations = num_generations, 
            num_parents_mating = num_parents_mating,
            mutation_probability = mutation_probability,
            crossover_probability = crossover_probability,
            sol_per_pop = sol_per_pop,
            num_genes = len(final_gene_space),  
            fitness_func = fitness_func,
            gene_space = final_gene_space,
            on_generation= on_generation,
            random_seed= random_seed)
        
        ga.run()
        # Decode best solution
        best_solution, _, _ = ga.best_solution(ga.last_generation_fitness)
        full_solution = [None] * 5
        var_cursor = 0
        for i in range(5):
            if i in fixed_params:
                full_solution[i] = fixed_params[i]
            else:
                full_solution[i] = best_solution[var_cursor]
                var_cursor += 1

        best_angle, best_sy,best_sx,best_ty, best_tx = full_solution
        A = utils.computeA(best_angle, best_sy,best_sx, best_ty, best_tx, xI, xJ)
        print(f"Best transformation: angle={best_angle:.2f}, sx={best_sx:.2f}, sy={best_sy:.2f}, tx={best_tx:.2f}, ty={best_ty:.2f}")
        Aparams.append(full_solution)
        
        del ga
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # ---- Run LDDMM ----
        for it in range(num_iterations):
            ### transformed image
            fAI = utils.transform_image_source_to_target(xv,v,A,xI,I,XJ)
            # Energy
            EM = torch.sum((fAI - J)**2*WM)/2.0/sigmaM**2
            ER = torch.sum(torch.sum(torch.abs(torch.fft.fftn(v,dim=(1,2)))**2,dim=(0,-1))*LL)*DV/2.0/v.shape[1]/v.shape[2]/sigmaR**2
            E = EM + ER
            
            Esave.append( [E.item(), EM.item(), ER.item()] )
            # gradient update
            E.backward()
            with torch.no_grad():            
                # v grad
                vgrad = v.grad
                # smooth it
                vgrad = torch.fft.ifftn(torch.fft.fftn(vgrad,dim=(1,2))*K[...,None],dim=(1,2)).real
                v -= vgrad*epV
                v.grad.zero_()
                
            # update weights
            if not it%10:
                with torch.no_grad():
                    # M step for these params
                    if estimate_muB:
                        muB = torch.sum(WB*J,dim=(-1,-2))/torch.sum(WB)

                    if it >= 50:
                        W = torch.stack((WM,WB))
                        pi = torch.sum(W,dim=(1,2))
                        pi += torch.max(pi)*1e-6
                        pi /= torch.sum(pi)

                        # now the E step, update the weights
                        WM = pi[0]* torch.exp( -torch.sum((fAI - J)**2,0)/2.0/sigmaM**2 )/np.sqrt(2.0*np.pi*sigmaM**2)**J.shape[0]
                        WB = pi[1]* torch.exp( -torch.sum((muB[...,None,None] - J)**2,0)/2.0/sigmaB**2 )/np.sqrt(2.0*np.pi*sigmaB**2)**J.shape[0]
                        WS = WM+WB
                        WS += torch.max(WS)*1e-6
                        WM /= WS
                        WB /= WS
                        
            if not it % 500 or it == (num_iterations - 1):
                print(f'{it} of {num_iterations}')
            
        del vgrad, fAI, EM, ER, E, W, pi, WS
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
                     
    return {
            'v': v.clone().detach(), 
            'xv': xv, 
            'A':A.to(device),
            'E': Esave,
            'WM': WM.clone().detach(),
            'WB': WB.clone().detach(),
            'Aparams': Aparams
        }



def stage1_GA(xI, I, xJ, J, gene_space=None, 
              num_generations=800, num_parents_mating=50, sol_per_pop=100,
              mutation_probability=0.2, crossover_probability=0.8, 
              random_seed=None):

    # Compute default search ranges from data
    rIy = (xI[0].max() - xI[0].min()).item()
    rIx = (xI[1].max() - xI[1].min()).item()
    rJy = (xJ[0].max() - xJ[0].min()).item()
    rJx = (xJ[1].max() - xJ[1].min()).item()
    sx_range = rIx / rJx if rIx > rJx else rJx / rIx  
    sy_range = rIy / rJy if rIy > rJy else rJy / rIy

    default_gene_space = [
        {'low': -180.0, 'high': 180.0},                 # angle
        {'low': -(sy_range + 0.1), 'high': (sy_range + 0.1)},  # sy
        {'low': -(sx_range + 0.1), 'high': (sx_range + 0.1)},  # sx
        {'low': -rJy / 2, 'high': rJy / 2},             # ty
        {'low': -rJx / 2, 'high': rJx / 2},             # tx
    ]

    # Update with user-defined gene_space
    if gene_space is not None:
        for i, g in enumerate(gene_space):
            if g:
                g_fixed = g.copy()
                if 'low' not in g_fixed or g_fixed['low'] is None:
                    g_fixed['low'] = default_gene_space[i]['low']
                if 'high' not in g_fixed or g_fixed['high'] is None:
                    g_fixed['high'] = default_gene_space[i]['high']
                default_gene_space[i].update(g_fixed)

    # Identify fixed vs variable parameters
    final_gene_space = []
    fixed_params = {}
    variable_indices = []

    for i, g in enumerate(default_gene_space):
        if g['low'] == g['high']:
            fixed_params[i] = g['low']
        else:
            final_gene_space.append(g)
            variable_indices.append(i)

    # Define the fitness function
    def fitness_func(ga, solution, solution_idx):
        full_params = [None] * 5
        var_cursor = 0
        for i in range(5):
            if i in fixed_params:
                full_params[i] = fixed_params[i]
            else:
                full_params[i] = solution[var_cursor]
                var_cursor += 1
        angle, sy, sx, ty, tx = full_params
        A = utils.computeA(angle, sy, sx, ty, tx, xI, xJ)
        fAI = utils.transform_image_with_A(A, xJ, xI, I)
        mse = 1 / (1 + torch.sum((fAI - J) ** 2))
        return mse.numpy()

    # Optional: monitor progress
    def on_generation(ga):
        global last_fitness
        if ga.generations_completed % 100 == 0:
            print(f"Generation = {ga.generations_completed}")
            print(f"Fitness    = {ga.best_solution(pop_fitness=ga.last_generation_fitness)[1]}")
            print(f"Change     = {ga.best_solution(pop_fitness=ga.last_generation_fitness)[1] - last_fitness}")
            last_fitness = ga.best_solution(pop_fitness=ga.last_generation_fitness)[1]

    # Create GA instance
    ga = pygad.GA(
        num_generations=num_generations,
        num_parents_mating=num_parents_mating,
        mutation_probability=mutation_probability,
        crossover_probability=crossover_probability,
        sol_per_pop=sol_per_pop,
        num_genes=len(final_gene_space),
        gene_space=final_gene_space,
        fitness_func=fitness_func,
        on_generation=on_generation,
        random_seed=random_seed
    )

    ga.run()

    # Decode best solution
    best_solution, _, _ = ga.best_solution(ga.last_generation_fitness)
    full_solution = [None] * 5
    var_cursor = 0
    for i in range(5):
        if i in fixed_params:
            full_solution[i] = fixed_params[i]
        else:
            full_solution[i] = best_solution[var_cursor]
            var_cursor += 1

    return full_solution, ga.best_solutions_fitness


def stage2_LDDMM(xI,I,xJ,J,A = None, 
                 num_iterations = 5000,
                 a = 1.0, p = 2.0, expand = 1.0, nt = 3,
                 epV = 1.0, muB = None,
                 sigmaM = 1.0, sigmaB = 1.0, sigmaR = 1e3):
    # change to torch
    if A is not None:
        # change to torch
        if isinstance(A, torch.Tensor):
            A = torch.clone(A).to(device)
        else :
            A = torch.tensor(A)
    else:
        A = torch.eye(3,device = device)
    
    estimate_muB = muB is None
    
    I = torch.as_tensor(I,device=device)                         
    J = torch.as_tensor(J,device=device)
    xI = [torch.tensor(x,device=device) for x in xI]
    xJ = [torch.tensor(x,device=device) for x in xJ]
    XJ = torch.stack(torch.meshgrid(*xJ,indexing='ij'),-1)
    

    minv = torch.as_tensor([x[0] for x in xI],device=device)
    maxv = torch.as_tensor([x[-1] for x in xI],device=device)
    minv,maxv = (minv+maxv)*0.5 + 0.5*torch.tensor([-1.0,1.0],device=device)[...,None]*(maxv-minv)*expand
    xv = [torch.arange(m,M,a*0.5,device=device) for m,M in zip(minv,maxv)]
    XV = torch.stack(torch.meshgrid(*xv,indexing='ij'),-1)
    v = torch.zeros((nt,XV.shape[0],XV.shape[1],XV.shape[2]),device=device,requires_grad=True)

    dv = torch.as_tensor([x[1]-x[0] for x in xv],device=device)
    fv = [torch.arange(n,device=device)/n/d for n,d in zip(XV.shape,dv)]
    FV = torch.stack(torch.meshgrid(*fv,indexing='ij'),-1)
    LL = (1.0 + 2.0*a**2* torch.sum( (1.0 - torch.cos(2.0*np.pi*FV*dv))/dv**2 ,-1))**(p*2.0)
    K = 1.0/LL
    DV = torch.prod(dv)

    WM = torch.ones(J[0].shape,dtype=J.dtype,device=J.device)*0.5
    WB = torch.ones(J[0].shape,dtype=J.dtype,device=J.device)*0.5
    
    Esave = []
    for it in range(num_iterations):
        ### transformed image
        fAI = utils.transform_image_source_to_target(xv,v,A,xI,I,XJ)

        # objective function
        EM = torch.sum((fAI - J)**2*WM)/2.0/sigmaM**2
        ER = torch.sum(torch.sum(torch.abs(torch.fft.fftn(v,dim=(1,2)))**2,dim=(0,-1))*LL)*DV/2.0/v.shape[1]/v.shape[2]/sigmaR**2
        E = EM + ER
        Esave.append( [E.item(), EM.item(), ER.item()] )
        
        # gradient update
        E.backward()
        with torch.no_grad():            
            # v grad
            vgrad = v.grad
            # smooth it
            vgrad = torch.fft.ifftn(torch.fft.fftn(vgrad,dim=(1,2))*K[...,None],dim=(1,2)).real
            v -= vgrad*epV
            v.grad.zero_()


        # update weights
        if not it%10:
            with torch.no_grad():
                # M step for these params
                if estimate_muB:
                    muB = torch.sum(WB*J,dim=(-1,-2))/torch.sum(WB)

                if it >= 50:
                    W = torch.stack((WM,WB))
                    pi = torch.sum(W,dim=(1,2))
                    pi += torch.max(pi)*1e-6
                    pi /= torch.sum(pi)

                    # now the E step, update the weights
                    WM = pi[0]* torch.exp( -torch.sum((fAI - J)**2,0)/2.0/sigmaM**2 )/np.sqrt(2.0*np.pi*sigmaM**2)**J.shape[0]
                    WB = pi[1]* torch.exp( -torch.sum((muB[...,None,None] - J)**2,0)/2.0/sigmaB**2 )/np.sqrt(2.0*np.pi*sigmaB**2)**J.shape[0]
                    WS = WM+WB
                    WS += torch.max(WS)*1e-6
                    WM /= WS
                    WB /= WS
        if not it % 500 or it == (num_iterations - 1):
            print(f'{it} of {num_iterations}')
            
    return {
        'v': v.clone().detach(), 
        'xv': xv, 
        'A':A.to(device),
        'E': Esave,
        'WM': WM.clone().detach(),
        'WB': WB.clone().detach()
    }
       
        


        

    

    
    
    
