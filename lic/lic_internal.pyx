
import numpy as np
cimport numpy as np

cdef void _advance(float vx, float vy,
        int* x, int* y, float*fx, float*fy, int w, int h):
    cdef float tx, ty
    if vx>=0:
        tx = (1-fx[0])/vx
    else:
        tx = -fx[0]/vx
    if vy>=0:
        ty = (1-fy[0])/vy
    else:
        ty = -fy[0]/vy
    if tx<ty:
        if vx>=0:
            x[0]+=1
            fx[0]=0
        else:
            x[0]-=1
            fx[0]=1
        fy[0]+=tx*vy
    else:
        if vy>=0:
            y[0]+=1
            fy[0]=0
        else:
            y[0]-=1
            fy[0]=1
        fx[0]+=ty*vx
    if x[0]>=w:
        x[0]=w-1 # FIXME: other boundary conditions?
    if x[0]<0:
        x[0]=0 # FIXME: other boundary conditions?
    if y[0]<0:
        y[0]=0 # FIXME: other boundary conditions?
    if y[0]>=h:
        y[0]=h-1 # FIXME: other boundary conditions?


#np.ndarray[float, ndim=2]
def line_integral_convolution(
        np.ndarray[float, ndim=3] vectors,
        np.ndarray[float, ndim=2] texture,
        np.ndarray[float, ndim=1] kernel):
    cdef int i,j,k,x,y
    cdef int h,w,kernellen
    cdef int t
    cdef float fx, fy, tx, ty
    cdef np.ndarray[float, ndim=2] result

    w = vectors.shape[0]
    h = vectors.shape[1]
    t = vectors.shape[2]
    kernellen = kernel.shape[0]
    print(w, h, t)
    if t!=2:
        raise ValueError("Vectors must have two components (not %d)" % t)
    result = np.zeros((w,h),dtype=np.float32)

    for i in range(w):
        for j in range(h):
            fx = 0.5
            fy = 0.5

            k = kernellen//2
            result[i,j] += kernel[k]*texture[i,j]
            while k<kernellen-1:
                _advance(vectors[i,j,0],vectors[i,j,1],
                        &i, &j, &fx, &fy, w, h)
                k+=1
                result[i,j] += kernel[k]*texture[i,j]


            while k>0:
                _advance(-vectors[i,j,0],-vectors[i,j,1],
                        &i, &j, &fx, &fy, w, h)
                k-=1
                result[i,j] += kernel[k]*texture[i,j]

    return result
