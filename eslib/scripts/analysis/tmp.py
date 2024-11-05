#!/usr/bin/python

import numpy as np
import sys, glob, math

__all__ = ['main']

def abc2h(a, b, c, alpha, beta, gamma):
   h = np.zeros((3,3), float)
   h[0,0] = a
   h[0,1] = b * math.cos(gamma)
   h[0,2] = c * math.cos(beta)
   h[1,1] = b * math.sin(gamma)
   h[1,2] = (b * c * math.cos(alpha) - h[0,1] * h[0,2]) / h[1,1]
   h[2,2] = math.sqrt(c**2 - h[0,2]**2 - h[1,2]**2)
   return h

def invert_ut3x3(h):
   ih = np.zeros((3,3), float)
   for i in range(3):
      ih[i,i] = 1.0 / h[i,i]
   ih[0,1] = -ih[0,0] * h[0,1] * ih[1,1]
   ih[1,2] = -ih[1,1] * h[1,2] * ih[2,2]
   ih[0,2] = -ih[1,2] * h[0,1] * ih[0,0] - ih[0,0] * h[0,2] * ih[2,2]
   return ih

def pbcdist(q1, q2, h, ih):
   s = np.dot(ih, q1 - q2)
   for i in range(3):
      s[i] -= round(s[i])
   return np.dot(h, s)

def read_frame(filedesc):
   natoms = int(filedesc.readline())
   comment = filedesc.readline()

   cell = np.zeros(6, float)
   names = []
   q = np.zeros((natoms, 3), float)
   cell[:] = comment.split()[2:8]
   cell *= [1, 1, 1, 0.017453293, 0.017453293, 0.017453293]

   for i in range(natoms):
      line = filedesc.readline().split()
      names.append(line[0])  # No need to decode
      q[i] = line[1:4]

   return [cell, names, q]

def main(prefix, reference):
   # the reference structure should have zero centroid (not center of mass, don't want to use masses here)
   iref = open(reference, "r")
   [cell, names, ref] = read_frame(iref)
   nr = len(ref)

   # Here we read directly the position of the centroid   
   ipos = open(prefix + ".xc.xyz", "r")
   ivc = open(prefix + ".vc.xyz", "r")

   ofile = open(prefix + ".vcalign.xyz", "w")
   oposfile = open(prefix + ".xcalign.xyz", "w")

   ifr = 0
   dt = 20.670687 * 2  # 1 fs in atomic time. CHANGE THIS ACCORDINGLY!
   angtobohr = 1.8897261
   kt = np.zeros((nr, 3, 3), float)
   rpos = np.zeros((nr, 3), float)
   rposold = np.zeros((nr, 3), float)
   rvel = np.zeros((nr, 3), float)

   while True:
      try:
         [cell, names, posc] = read_frame(ipos)
         [cell, names, velc] = read_frame(ivc)
      except:
         sys.exit(0)

      nat = len(posc)
      if ifr > 0:
         ofile.write("%d\n# Velocity filtered by rotating frame. Frame: %d\n" % (nat, ifr))
      oposfile.write("%d\n# Position rotated to molecular ref. Frame: %d\n" % (nat, ifr))
      for i in range(0, nat, nr):
         jvel = velc[i:i+nr]
         jpos = posc[i:i+nr]
         jnames = names[i:i+nr]
         sc, rot = kabsch(jpos, ref)
         rpos[:] = jpos
         for v in rpos:
            v[:] = np.dot(rot, v - sc)
         for j in range(nr):
            oposfile.write("%6s  %15.7e  %15.7e  %15.7e\n" % (jnames[j], rpos[j, 0], rpos[j, 1], rpos[j, 2]))
         if ifr == 0:
            rposold[:] = rpos[:]
         else:
            rvel = (rpos - rposold) * angtobohr / dt
            rposold[:] = rpos[:]
         if ifr > 0:
            for j in range(nr):
               ofile.write("%6s  %15.7e  %15.7e  %15.7e\n" % (jnames[j], rvel[j, 0], rvel[j, 1], rvel[j, 2]))
      ifr += 1

def kabsch(pos, ref):
   cp = np.zeros(len(pos[0]))
   for i in range(len(pos)):
      cp += pos[i, :]
   cp = cp / len(pos)
   spos = pos.copy()
   for v in spos:
      v -= cp

   A = np.dot(ref.T, spos)
   U, s, V = np.linalg.svd(A)
   d = np.sign(np.linalg.det(np.dot(V.T, U.T)))
   V[2, :] *= d
   return (cp, np.dot(U, V))

if __name__ == '__main__':
   main(sys.argv[1], sys.argv[2])










