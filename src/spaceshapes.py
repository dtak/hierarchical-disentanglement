import sys
import os
import pickle
import pdb
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imresize
from scipy.ndimage import rotate, shift
from scipy.special import expit

def circle_through(p1, p2, p3):
  """
  Returns the center and radius of the circle passing the given 3 points.
  In case the 3 points form a line, returns (None, infinity).
  """
  temp = p2[0] * p2[0] + p2[1] * p2[1]
  bc = (p1[0] * p1[0] + p1[1] * p1[1] - temp) / 2
  cd = (temp - p3[0] * p3[0] - p3[1] * p3[1]) / 2
  det = (p1[0] - p2[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p2[1])

  if abs(det) < 1.0e-6:
    return (None, None, np.inf)

  # Center of circle
  cx = (bc*(p2[1] - p3[1]) - cd*(p1[1] - p2[1])) / det
  cy = ((p1[0] - p2[0]) * cd - (p2[0] - p3[0]) * bc) / det

  rad = np.sqrt((cx - p1[0])**2 + (cy - p1[1])**2)
  return (cx, cy, rad)

def within_circle(x,y,circle):
  cx,cy,rad = circle
  return np.sqrt((x-cx)**2 + (y-cy)**2) <= rad

def within_rect(x,y,rect):
  xmin, ymin, dx, dy = rect
  return x>xmin and x<xmin+dx and y>ymin and y<ymin+dy

def within_triangle(x,y,tri):
  sign = lambda x1,y1,x2,y2,x3,y3: (x1 - x3)*(y2 - y3) - (x2 - x3)*(y1 - y3)
  x1,y1=tri[0]
  x2,y2=tri[1]
  x3,y3=tri[2]
  d1 = sign(x,y,x1,y1,x2,y2)
  d2 = sign(x,y,x2,y2,x3,y3)
  d3 = sign(x,y,x3,y3,x1,y1)
  has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
  has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
  return not (has_neg and has_pos)

class Jet():
  def __init__(self, x1, y1, width, height):
    y2 = y1 - height
    x2 = x1 + width
    yC = y2 + width/2.0
    xC = (x1+x2)/2.0
    x1C = (x1+xC)/2.0
    x2C = (xC+x2)/2.0
    self.tri1 = ((x1,y2), (xC,y2), (x1C, yC))
    self.tri2 = ((xC,y2), (x2,y2), (x2C, yC))
    self.rect = (x1, y2, width, height)

  def contains(self,x, y):
    return within_rect(x,y,self.rect)

def square_star(y,x,a,radius=0.8):
  x,y = abs(x),abs(y)
  xmax = radius
  ymax = radius - x
  if a > 0:
    p1 = [0,radius]
    p2 = [radius,0]
    p3 = [radius/2.0-a, radius/2.0-a]
    if within_circle(x,y,circle_through(p1,p2,p3)):
      return 0
  return int(x <= xmax) * int(y <= ymax)

def circle_moon(y,x,a,radius=0.75):
  if not within_circle(x,y,(0,0,radius)):
    return 0
  ypos = y
  Xpos = np.sqrt(radius**2 - y**2)
  Rpos = 2 * Xpos
  Phase = a
  if Phase < 0.5:
    Xpos1 = -Xpos
    Xpos2 = Rpos - 2*Phase*Rpos - Xpos
  else:
    Xpos1 = Xpos
    Xpos2 = Xpos - 2*Phase*Rpos + Rpos
  if x > min(Xpos1,Xpos2) and x < max(Xpos1,Xpos2):
    return 1
  return 0

def triangle_ship(y,x,b,c,radius=1.1,nudge=0.1):
  rt3 = np.sqrt(3)
  rt32 = rt3/2.0
  in_thrusters = False
  in_exhaust = False
  if b == 1:
    width = radius/3.0
    in_thrusters = (y <= -nudge)*(y > -0.1*radius) * (np.abs(x)<width)
    in_exhaust = Jet(-width, -0.1*radius, 2*width, c).contains(x,y)
  elif b == 2:
    width = radius/6.0
    in_thrusters = (y <= -nudge)*(y > -0.1*radius) * (np.abs(x-radius/4)<width or np.abs(x+radius/4)<width)
    in_exhaust1 = Jet(-2.5*radius/6,          -0.1*radius, 2*width, c).contains(x,y)
    in_exhaust2 = Jet(-2.5*radius/6+radius/2, -0.1*radius, 2*width, c).contains(x,y)
    in_exhaust = in_exhaust1 or in_exhaust2
  in_triangle = (y < rt32*radius-rt3*x) * (y < rt32*radius+rt3*x) * (y > 0)

  return in_triangle or in_thrusters or in_exhaust

RES = 256
pixels = range(RES)

def map_pixel(i,j):
  x = (i/RES)*4 - 2
  y = (j/RES)*4 - 2
  return y,x

def make_square(cy,cx,a,**kw):
  result = np.zeros((RES,RES))
  for i in pixels:
    for j in pixels:
      y,x = map_pixel(i,j)
      color = square_star(x-cx, y-cy, a,**kw)
      result[i,j] = color
  return result

def make_circle(cx,cy,a,**kw):
  result = np.zeros((RES,RES))
  for i in pixels:
    for j in pixels:
      y,x = map_pixel(i,j)
      color = circle_moon(x-cx, y-cy, a,**kw)
      result[i,j] = color
  return result

def make_triangle(cx,cy,a,b,c,**kw):
  rot = np.array([
    [np.cos(a),-np.sin(a)],
    [np.sin(a),np.cos(a)]])
  result = np.zeros((RES,RES))
  for i in pixels:
    for j in pixels:
      y,x = np.dot(rot, np.array(map_pixel(i,j)))
      color = triangle_ship(x-cx, y-cy, b, c, **kw)
      result[i,j] = color
  return result[::-1]

def rescale(ns, lo, hi):
  return lo + (hi - lo) * ns

data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data')
proto_fn = os.path.join(data_dir, 'spaceshapes_prototypes.pkl')
if os.path.exists(proto_fn):
  with open(proto_fn, 'rb') as f:
    prototypes = pickle.load(f)
else:
  print('precaching prototype images (may take a sec)...')
  star_params = np.linspace(0, 0.2, 256)
  moon_params = np.linspace(0, 0.4, 256)
  ship_params = np.linspace(0.25, 0.67, 256)

  star_protos = [make_square(0,0,a) for a in star_params]
  moon_protos = [make_circle(0,0,a) for a in moon_params]

  ship_proto = shift(make_triangle(0,0,0,0,0), (25,0))
  ship_protos_1 = [shift(make_triangle(0,0,0,1,c), (25,0)) for c in ship_params]
  ship_protos_2 = [shift(make_triangle(0,0,0,2,c), (25,0)) for c in ship_params]

  prototypes = {
      'star_params': star_params,
      'moon_params': moon_params,
      'ship_params': ship_params,
      'star_protos': star_protos,
      'moon_protos': moon_protos,
      'ship_proto_none': ship_proto,
      'ship_protos_one': ship_protos_1,
      'ship_protos_two': ship_protos_2
  }

  with open(proto_fn, 'wb') as f:
    pickle.dump(prototypes, f)

def make_moon(x,y,phase,**kw):
  x = rescale(x, -64, 64)
  y = rescale(y, -64, 64)
  phase = rescale(1-phase, 0, 0.4)

  idx = np.argmin(np.abs(prototypes['moon_params'] - phase))
  proto = prototypes['moon_protos'][idx]
  
  shifted = shift(proto, (round(-1*y),round(x)))
  resized = imresize(shifted, (64,64), interp='bilinear')
  return resized

def make_star(x,y,light,max_l=0.165,**kw):
  x = rescale(x, -64, 64)
  y = rescale(y, -64, 64)
  light = rescale(1-light, 0, max_l)
  
  idx = np.argmin(np.abs(prototypes['star_params'] - light))
  proto = prototypes['star_protos'][idx]

  shifted = shift(proto, (round(-1*y),round(x)))
  resized = imresize(shifted, (64,64), interp='bilinear')
  return resized

def make_ship(x,y,rot,jettype=1,jetlen=0,**kw):
  x = rescale(x, -64, 64)
  y = rescale(y, -64, 64)
  rot = rescale(1-rot, -45, 45)
  jetlen = rescale(jetlen, 0.25, 0.67)

  idx = np.argmin(np.abs(prototypes['ship_params'] - jetlen))
  assert(jettype in [1,2])
  if jettype == 1:
    proto = prototypes['ship_proto_none']
  elif jettype == 2:
    proto = prototypes['ship_protos_two'][idx]

  rotated = rotate(proto, rot, reshape=False, order=0)
  shifted = shift(rotated, (round(-1*y),round(x)))
  resized = imresize(shifted, (64,64), interp=INTERP)
  return resized

def randu(n):
  return np.random.uniform(size=(n,))

def make_dataset():
  moon_xs = randu(20000)
  moon_ys = randu(20000)
  moon_phases = randu(20000)
  moon_params = zip(moon_xs, moon_ys, moon_phases)

  star_xs = randu(20000)
  star_ys = randu(20000)
  star_lights = randu(20000)
  star_params = zip(star_xs, star_ys, star_lights)

  ship_none_xs = randu(10000)
  ship_none_ys = randu(10000)
  ship_none_rots = randu(10000)
  ship_none_params = zip(
      ship_none_xs,
      ship_none_ys,
      ship_none_rots,
      np.full_like(ship_none_xs, 1),
      np.zeros_like(ship_none_ys))

  ship_one_xs = randu(10000)
  ship_one_ys = randu(10000)
  ship_one_rots = randu(10000)
  ship_one_jets = randu(10000)
  ship_one_params = zip(
      ship_one_xs,
      ship_one_ys,
      ship_one_rots,
      np.full_like(ship_one_jets, 2),
      ship_one_jets)

  ship_xs = np.hstack((ship_none_xs, ship_one_xs))
  ship_ys = np.hstack((ship_none_ys, ship_one_ys))
  ship_rots = np.hstack((ship_none_rots, ship_one_rots))
  ship_jets = np.hstack((np.zeros(10000), ship_one_jets))

  print('making ships')
  ships = np.array(
      [make_ship(*p) for p in ship_none_params] +
      [make_ship(*p) for p in ship_one_params])
  print('making moons')
  moons = np.array([make_moon(*p) for p in moon_params])
  print('making stars')
  stars = np.array([make_star(*p) for p in star_params])
  print('done')

  data = np.vstack((moons, stars, ships))

  cats = np.array([
    np.array([1]*20000 + [2]*20000 + [3]*20000),
    np.array([0]*40000 + [1]*10000 + [2]*10000)
  ]).T

  NOTHING = np.zeros(20000)

  latents = np.vstack((
    np.hstack((moon_xs, star_xs, ship_xs)),
    np.hstack((moon_ys, star_ys, ship_ys)),
    np.hstack((moon_phases, NOTHING,     NOTHING)),
    np.hstack((NOTHING,     star_lights, NOTHING)),
    np.hstack((NOTHING,     NOTHING,     ship_rots)),
    np.hstack((NOTHING,     NOTHING,     ship_jets)))).T

  idx = np.arange(60000)
  np.random.shuffle(idx)
  for i in idx[:100]:
    decoded = true_decoder(cats[i,0], cats[i,1], latents[i])
    np.testing.assert_allclose(data[i], decoded)

  np.save(os.path.join(data_dir, 'spaceshapes_X.npy'), data)
  np.save(os.path.join(data_dir, 'spaceshapes_Z.npy'), latents)
  np.save(os.path.join(data_dir, 'spaceshapes_A.npy'), cats)

def true_decoder(c1, c2, z):
  assert(c1 == 1 or c1 == 2 or c1 == 3)
  assert(c2 == 0 or c2 == 1 or c2 == 2)

  if c1 == 1:
    return make_moon(z[0], z[1], z[2])
  elif c1 == 2:
    return make_star(z[0], z[1], z[3])
  else:
    return make_ship(z[0], z[1], z[4], c2, z[5])

class Spaceshapes():
  @property
  def data(self):
    dirname = os.path.dirname(os.path.realpath(__file__))
    prefix = os.path.join(dirname, f"data/spaceshapes")
    if not os.path.exists(f"{prefix}_X.npy"):
      print("Creating dataset for the first time... (may take a while)")
      make_dataset()

    Dataset = namedtuple('Dataset', ['A', 'Z', 'X', 'AMZ'])

    A = np.load(f"{prefix}_A.npy")
    Z = np.load(f"{prefix}_Z.npy")
    X = np.load(f"{prefix}_X.npy")

    AMZ = np.hstack([A, Z])

    A_list = []
    for a in A:
        a1 = [0,0,0]
        a2 = [0,0]
        a1[a[0]-1] = 1
        if a[1] > 0: a2[a[1]-1] = 1
        A_list.append([a1, a2])
    A = np.array(A_list)

    return Dataset(X=X, A=A, Z=Z, AMZ=AMZ)
    
  @property
  def hierarchy(self):
    return [
        { "type": "continuous" },
        { "type": "continuous" },
        { "type": "categorical",
          "options": [
            [{ "type": "continuous" }],
            [{ "type": "continuous" }],
            [
              { "type": "continuous" },
              { "type": "categorical",
                "options": [
                    [],
                    [{ "type": "continuous" }]
                  ]
                }
              ]
            ]
          }]
