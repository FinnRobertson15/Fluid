import torch
import torch.nn.functional as F
import numpy as np
from HelperFunctions import *

class Fluid:
    def __init__(self, size, density=1000, gravity=-9.81, numIters=100, overRelaxation=1, boundaries=False, device=torch.device('cuda')):
        self.n = size+2 if boundaries else size
        self.numCells = size**2
        self.density = density
        self.gravity = gravity
        self.numIters = numIters
        self.overRelaxation = overRelaxation
        self.boundaries = boundaries
        self.device = device

        self.v = torch.zeros(self.n, self.n).to(device)
        self.u = torch.zeros(self.n, self.n).to(device)
        self.m = torch.ones(self.n, self.n).to(device)

        self.p = torch.zeros(self.n, self.n).to(device)
        w = 2
        self.s = F.pad(torch.ones(self.n-2*w, self.n-2*w).bool(), (w,w,w,w), value=False).to(device)

        centre = self.n / 2
        x = (torch.arange(self.n) - centre).pow(2)
        circle = normalize(torch.sqrt(x + x[:,None])).to(device)
        self.s &= circle > 0.25
        self.s[[0,1,-2,-1], round(centre)-10:round(centre)+10] = True
        self.s[round(centre)-10:round(centre)+10,[0,1,-2,-1]] = True


        self.x = torch.arange(self.n)[:,None].expand(-1,self.n).to(device)
        self.y = torch.arange(self.n)[None,:].expand(self.n,-1).to(device)
        self.iteration = 0


    def render(self):
        def hsv_to_rgb_torch(h, s, v):
            # h, s, v: shape (n,)
            i = torch.floor(h * 6)
            f = h * 6 - i
            i = i.long() % 6

            p = v * (1 - s)
            q = v * (1 - f * s)
            t = v * (1 - (1 - f) * s)

            r = torch.zeros_like(h)
            g = torch.zeros_like(h)
            b = torch.zeros_like(h)

            idx = i == 0
            r[idx], g[idx], b[idx] = v[idx], t[idx], p[idx]

            idx = i == 1
            r[idx], g[idx], b[idx] = q[idx], v[idx], p[idx]

            idx = i == 2
            r[idx], g[idx], b[idx] = p[idx], v[idx], t[idx]

            idx = i == 3
            r[idx], g[idx], b[idx] = p[idx], q[idx], v[idx]

            idx = i == 4
            r[idx], g[idx], b[idx] = t[idx], p[idx], v[idx]

            idx = i == 5
            r[idx], g[idx], b[idx] = v[idx], p[idx], q[idx]

            return torch.stack([r, g, b], dim=-1)
        
        angles = torch.atan2(self.u, self.v)
        hues = (angles + torch.pi) / (2 * torch.pi)

        kS = 0.5
        s = (1-kS) + kS * self.m.clamp(0, 1)
        kV = 0.1
        v = (1-kV) + kV * self.p.clamp(0, 1)

        rgb = 255 * hsv_to_rgb_torch(hues, s, v)
        rgb = torch.where(self.s[:,:,None], rgb, 0)
        rgb = rgb.view(self.n, self.n, 3)
        if self.boundaries:
            rgb = rgb[1:-1, 1:-1]
        rgb = torch.rot90(rgb, k=1, dims=(0, 1)).flip(dims=[0])
        return rgb

    def integrate(self, dt):
        # self.s = gen_box(self.n, self.iteration).to(self.device)
        mask = self.s & self.s.roll(1, 1)
        if self.boundaries:
            mask = F.pad(mask[1:, 1:-1], (1,1,1,0), value=False)
        self.v[mask] -= self.gravity * dt
        self.iteration += 1

    def solveIncompressibility(self, dt):

        ss = torch.stack([self.s.roll((-i,-j),(0,1)).float() for i,j in [[-1,0],[1,0],[0,-1],[0,1]]], 0)
        s = ss.sum(0)
        mask = self.s & (s > 0)
        if self.boundaries:
            mask = F.pad(mask[1:-1,1:-1], (1,1,1,1), value=False)
        maskX = mask.roll(1,0)
        maskY = mask.roll(1,1)


        for iter in range(self.numIters):
            div = self.u.roll(-1,0) - self.u + self.v.roll(-1,1) - self.v
            p = (self.overRelaxation * -div / s.clamp(1))

            sx0, sx1, sy0, sy1 = ss * p
            self.u[mask] -= sx0[mask]
            self.v[mask] -= sy0[mask]
            self.u[maskX] += sx1.roll(1,0)[maskX]
            self.v[maskY] += sy1.roll(1,1)[maskY]
        self.p = p * self.density / dt

        # Extrapolate
        validX = mask & maskX
        validY = mask & maskY
        for iter in range(self.numIters):
            neighboursX = torch.stack([validX.roll((-i,-j),(0,1))for i,j in [[-1,0],[1,0],[0,-1],[0,1]]], 0).any(0)
            extrapolateX = ~validX & neighboursX
            updateX = extrapolateX.any()
            if updateX:
                uu = torch.stack([self.u.roll((-i,-j),(0,1)).float() for i,j in [[-1,0],[1,0],[0,-1],[0,1]]], -1)[extrapolateX]
                ss = torch.stack([validX.roll((-i,-j),(0,1)).float() for i,j in [[-1,0],[1,0],[0,-1],[0,1]]], -1)[extrapolateX]
                self.u[extrapolateX] = (uu * ss).sum(-1) / ss.sum(-1)
                validX |= extrapolateX

            neighboursY = torch.stack([validY.roll((-i,-j),(0,1)) for i,j in [[-1,0],[1,0],[0,-1],[0,1]]], 0).any(0)
            extrapolateY = ~validY & neighboursY
            updateY = extrapolateY.any()
            if updateY:
                vv = torch.stack([self.v.roll((-i,-j),(0,1)).float() for i,j in [[-1,0],[1,0],[0,-1],[0,1]]], -1)[extrapolateY]
                ss = torch.stack([validY.roll((-i,-j),(0,1)).float() for i,j in [[-1,0],[1,0],[0,-1],[0,1]]], -1)[extrapolateY]
                self.v[extrapolateY] = (vv * ss).sum(-1) / ss.sum(-1)
                validY |= extrapolateY
            
            if not (updateX | updateY):
                break

    def sampleField(self, dt, field):
        n = self.n
        u, v = self.u, self.v
        dx, dy = 0, 0
        match field:
            case 'u':
                f = self.u
                v = torch.stack([self.v.roll((-i,-j), (0,1)) for i, j in [[-1,0],[0,0],[-1,1],[0,1]]],0).mean(0)
                dy = 0.5
            case 'v':
                f = self.v
                u = torch.stack([self.u.roll((-i,-j), (0,1)) for i, j in [[0,-1],[0,0],[1,-1],[1,0]]],0).mean(0)
                dx = 0.5
            case 'm':
                f = self.m
                u = torch.stack([self.u.roll(-i, 0) for i in range(2)],0).mean(0)
                v = torch.stack([self.v.roll(-j, 1) for j in range(2)],0).mean(0)
                dx, dy = 0.5, 0.5

        x = (self.x+dx) - dt * u
        y = (self.y+dy) - dt * v


        if self.boundaries:
            x = x.clamp(1,n)
            y = y.clamp(1,n)

        x0 = (x-dx).floor().int()
        tx = (x-dx)-x0
        x1 = x0+1

        y0 = (y-dy).floor().int()
        ty = (y-dy)-y0
        y1 = y0+1

        sx, sy = 1-tx, 1-ty
        if self.boundaries:
            x0, x1 = x0.clamp(0, self.n-1), x1.clamp(0, self.n-1)
            y0, y1 = y0.clamp(0, self.n-1), y1.clamp(0, self.n-1)
        else:
            x0, x1 = x0 % self.n, x1 % self.n
            y0, y1 = y0 % self.n, y1 % self.n


        val = sx*sy * f[x0, y0] + \
        tx*sy * f[x1, y0] + \
        tx*ty * f[x1, y1] + \
        sx*ty * f[x0, y1]

        return val

    def advectVel(self, dt):
        maskU = self.s & self.s.roll(1, 0)
        maskV = self.s & self.s.roll(1, 1)
        if self.boundaries:
            maskU = F.pad(maskU[1:, 1:-1], (1,1,1,0), value=False)
            maskV = F.pad(maskV[1:-1, 1:], (1,0,1,1), value=False)
        self.u[maskU], self.v[maskV] = self.sampleField(dt, 'u')[maskU], self.sampleField(dt, 'v')[maskV]

    def advectSmoke(self, dt):
        mask = self.s
        if self.boundaries:
            mask = F.pad(mask[1:-1,1:-1], (1,1,1,1), value=False)
        self.m[mask] = self.sampleField(dt, 'm')[mask]
