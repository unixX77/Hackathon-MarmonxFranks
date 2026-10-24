import sys, yaml, time, numpy as np, cv2
sys.path.insert(0,'/home/ameed/franka_ws/src/fr3_pick_place/scripts')
import handeye_calibrate as h
c=yaml.safe_load(open(sys.argv[1] if len(sys.argv)>1 else '/home/ameed/franka_ws/src/fr3_pick_place/config/handeye_charuco.yaml'))
import rclpy; rclpy.init(); col=h.Collector(c)
dic,board=h.make_board(c); col.spin(1.5)
k=np.array(col.info.k).reshape(3,3); d=np.array(col.info.d)
frames=[]; last=None; t0=time.time()
while len(frames)<40 and time.time()-t0<12:
    col.spin(0.05)
    if col.image is not None and col.image.header.stamp!=last:
        last=col.image.header.stamp; frames.append(col.bridge.imgmsg_to_cv2(col.image,desired_encoding='mono8'))
print('frames',len(frames),' size',frames[0].shape)
P={}; sharp=[]; sat=[]; bl=[]; wh=[]
for g in frames:
    co,ids,_=cv2.aruco.detectMarkers(g,dic)
    if ids is None: continue
    n,cc,ci=cv2.aruco.interpolateCornersCharuco(co,ids,g,board,cameraMatrix=k,distCoeffs=d)
    if n<20: continue
    for i,p in zip(ci.flatten(),cc.reshape(-1,2)): P.setdefault(int(i),[]).append(p)
    pts=cc.reshape(-1,2); x0,y0=np.maximum(pts.min(0)-20,0).astype(int); x1,y1=(pts.max(0)+20).astype(int)
    roi=g[y0:y1,x0:x1]; sharp.append(cv2.Laplacian(roi,cv2.CV_64F).var()); sat.append(float(np.mean(roi>=250)))
    bl.append(float(np.percentile(roi,5))); wh.append(float(np.percentile(roi,95)))
jit=np.array([np.std(np.array(v),axis=0) for v in P.values() if len(v)>=15])
print(f'board found in {len(sharp)}/{len(frames)} frames')
print(f'sharpness (Laplacian var): {np.mean(sharp):.0f}   [>~300 good, <150 blurry]')
print(f'contrast  black {np.mean(bl):.0f}  white {np.mean(wh):.0f}  (gap {np.mean(wh)-np.mean(bl):.0f}, want >120)')
print(f'saturated (glare) pixels in board area: {100*np.mean(sat):.2f}%   [want <1%]')
print(f'corner jitter between frames: mean {jit.mean():.3f}px  max {jit.max():.3f}px   [want <0.15px]')
