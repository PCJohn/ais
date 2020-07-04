import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

from ais import AIS

RANGE = 20
DISP_SIZE = 400
FPS = 30

IM = np.zeros((DISP_SIZE,DISP_SIZE,3),dtype=np.uint8)

def clear(frame):
    frame[:,:,:] = 0

def disp(frame):
    cv2.imshow('frame',frame)
    cv2.waitKey(0)

def to_pixel(t):
    return np.uint8((t+RANGE)/(2*RANGE) * DISP_SIZE)

def col(c,p):
    if c == 'R':
        return (0,0,int(p*255))
    elif c == 'B':
       return (int(p*255),0,0)
    elif c == 'G':
        return (0,int(p*255),0)
    elif c == 'Y':
        return (0,int(p*255),int(p*255))

def plot_dist(density,c='R'):
    xy = np.mgrid[-RANGE:RANGE:0.05, -RANGE:RANGE:0.05].reshape(2,-1).T
    probs = np.array(list(map(density,xy)))
    probs /= probs.max()
    pix = to_pixel(xy)
    for p,x,y in zip(probs, pix[:,0], pix[:,1]):
        if p > 0.05:
            cv2.circle(IM, (x,y), 2, color=col(c,p), thickness=1)

def show_text(s,pos,col=(255,255,255)):
    cv2.putText(IM,s,pos,cv2.FONT_HERSHEY_SIMPLEX,1,col,2)

def save_vid(frames,out_file):
    out_size = (400,400)
    fourcc = cv2.VideoWriter_fourcc(*"DIVX")
    out = cv2.VideoWriter(out_file,fourcc,FPS,out_size)
    for f in frames:
        f = cv2.resize(f,out_size)
        out.write(f)
    out.release()



if __name__ == '__main__':
    
    # Setup target distribution -- unnormalized gaussian
    target_mean = [-12,-12]
    target_cov = [[3,1],[2,3]]
    K = 10 # some scalar to unnormalize the pdf
    dist = multivariate_normal(target_mean,target_cov)
    target = lambda x: K * dist.pdf(x)
    ####

    # Run AIS to compute trajectories and weights
    ais = AIS(D=2, num_iter=30, parallel_rounds=5000)
    ais.compute_w(target)
    pred_Z = ais.partition()
    pred_mean = ais.mean()[0]
    ####
    
    # Start generating video
    vid = []

    # Display proposal
    plot_dist(ais.p_0, c='R')
    show_text('Proposal Distribution',(50,350))
    vid.extend([IM.copy()]*2*FPS)
    clear(IM)
    
    # Display target
    plot_dist(target, c='G')
    show_text('Target Distribution',(50,350))
    vid.extend([IM.copy()]*2*FPS)
    clear(IM)

    # Display intermediate disturbutions
    betas = np.linspace(0,1,10)
    plot_dist(ais.p_0, c='R')
    plot_dist(target, c='G')
    f = IM.copy()
    clear(IM)
    for beta in betas:
        intermediate = lambda x: ais.p_j(x,target,beta)
        plot_dist(intermediate,c='Y')
        IM = np.uint8(IM*0.5 + 0.5*f)
        show_text('Intermediate beta='+str(beta)[:4],(20,350),col=(0,255,255))
        vid.extend([IM.copy()]*FPS)
        clear(IM)

    # Display random trajectories
    plot_dist(ais.p_0, c='R')
    plot_dist(target, c='G')
    f = IM.copy()
    clear(IM)
    for i in range(2):
        t = np.random.randint(ais.traj[0].shape[0])
        traj = [step[t] for step in ais.traj]
        for pos in traj:
            IM = f.copy()
            p = to_pixel(pos)
            cv2.circle(IM, (p[0],p[1]), 3, color=(255,255,255), thickness=-1)
            show_text('Sample Trajectory '+str(i+1),(25,350))
            vid.append(IM.copy())
            clear(IM)

    # Display true and estimated mean of target distribution
    plot_dist(ais.p_0, c='R')
    plot_dist(target, c='G')
    p = to_pixel(np.float32(target_mean))
    cv2.circle(IM, (p[0],p[1]), 5, color=(255,0,0), thickness=-1)
    show_text('True Mean',(25,320),col=(255,0,0))
    p = to_pixel(ais.mean()[0])
    cv2.circle(IM, (p[0],p[1]), 5, color=(255,255,255), thickness=-1)
    show_text('Estimated Mean',(25,380),col=(255,255,255))
    vid.extend([IM.copy()]*3*FPS)
    clear(IM)

    # Save video
    save_vid(vid,'ais_viz.mp4')



