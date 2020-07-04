# Annealed Importance Sampling

An exercise to implement and visualize annealed importance sampling in NumPy.

Visualization: https://youtu.be/xS3sHLodVpM 

Usage:

    from ais import AIS
    
    ais = AIS(D=2, num_iter=30, parallel_rounds=1000)
    ais.compute_w(target)
    
    pred_Z = ais.partition()  # estimated partition function
    pred_mean = ais.mean()    # estimated mean
    trajectories = ais.traj   # sampling trajectories
    
    
