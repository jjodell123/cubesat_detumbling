import numpy as np
import matplotlib.pyplot as plt
import math
import colorsys
import plotly.graph_objs as go
from scipy.spatial.transform import Rotation as R
# import ppigrf
# from datetime import datetime

# Encompassing class for orbital objects
class Entity:
    def __init__(self, pos, vel, rot, rot_vel, forward=np.array([0, 0, 1]), mass=1):
        self.vel = vel
        self.mass = mass
        self.tail = pos
        self.tails = np.array([pos])
        self.forward_start = forward
        self.rot_vel = rot_vel
        self.rot = rot
        self.dirs = np.array([forward])
        self.heads = np.array([pos + forward])
        self.times = np.array([0])
        self.rot_vels = np.array([rot_vel])
        self.powers = np.array([np.sqrt(3 * (current_max * voltage)**2)])

    def get_state(self):
        return np.append(self.tail, (self.vel, self.rot, self.rot_vel))

    def get_pos(self):
        return self.tail

    def get_tails(self):
        return self.tails

    def set_pos(self, tail, add_to_array=False):
        self.tail = tail
        if add_to_array:
            self.tails = np.vstack((self.tails, tail))

    def get_vel(self):
        return self.vel

    def set_vel(self, vel):
        self.vel = vel

    def add_vel(self, vel):
        self.vel += vel

    def get_rot_vel(self):
        return self.rot_vel

    def get_rot_vels(self):
        return self.rot_vels

    def set_rot_vel(self, rot_vel, add_to_array=False):
        self.rot_vel = rot_vel
        if add_to_array:
            self.rot_vels = np.vstack((self.rot_vels, rot_vel))

    def add_rot_vel(self, rot_vel):
        self.rot_vel += rot_vel

    def get_mass(self):
        return self.mass

    def append_time(self, time):
        self.times = np.append(self.times, time)

    def get_times(self):
        return self.times

    def get_rot(self):
        return self.rot

    def get_forward(self):
        r = R.from_rotvec(self.rot)
        return r.apply(self.forward_start)

    def set_rot(self, rot, add_to_array=False):
        self.rot = rot
        if add_to_array:
            r = R.from_rotvec(self.rot)
            self.dirs = np.vstack((self.dirs, r.apply(self.forward_start)))
            self.heads = np.vstack((self.heads, self.tails[-1] + self.dirs[-1]))

    def get_pos_ray(self):
        return [self.heads[-1][0], self.heads[-1][1], self.heads[-1][2], 
                self.tails[-1][0], self.tails[-1][1], self.tails[-1][2]]

    def get_pos_rays(self):
        rays = []
        for i in range(len(self.tails)):
            rays.append([self.heads[i, 0], self.heads[i, 1], self.heads[i, 2], 
                self.tails[i, 0], self.tails[i, 1], self.tails[i, 2]])
        return rays

    def get_pos_rays_dir(self):
        rays = []
        for i in range(len(self.tails)):
            rays.append([self.tails[i, 0], self.tails[i, 1], self.tails[i, 2], 
                self.dirs[i, 0], self.dirs[i, 1], self.dirs[i, 2]])
        return rays

    def get_powers(self):
        return self.powers

    def append_powers(self, power):
        self.powers = np.append(self.powers, power)

    def print(self):
        print('Pos:', self.get_pos(), 'Vel:', self.vel, 'Rot:', self.rot, 'RVel:', self.rot_vel, 'Time:', self.times[-1])


def rk4(x, t, tau, derivsRK):
    """
    Runge-Kutta integrator (4th order)
    Input arguments -
        x = current value of dependent variable
        t = independent variable (usually time)
        tau = step size (usually timestep)
         derivsRK = right hand side of the ODE; derivsRK is the
        name of the function which returns dx/dt
    Calling format derivsRK(x,t).
    Output arguments -
        xout = new value of x after a step of size tau
    """
    half_tau = 0.5 * tau
    F1 = derivsRK(x, t)
    t_half = t + half_tau
    xtemp = x + half_tau * F1
    F2 = derivsRK(xtemp, t_half)
    xtemp = x + half_tau * F2
    F3 = derivsRK(xtemp, t_half)
    t_full = t + tau
    xtemp = x + tau * F3
    F4 = derivsRK(xtemp, t_full)
    xout = x + tau / 6. * (F1 + F4 + 2. * (F2 + F3))
    return xout


def make_entities_from_state(s):
    ents = []
    for i in range(len(s)):
        if i % 12 == 11:
            ents.append(Entity(s[i - 11: i - 8], s[i - 8: i - 5], s[i - 5: i -2], s[i - 2: i + 1]))

    return ents


# Convert rect coords to lat, long, alt
def rect_to_lla(r):
    x = r[0] if not np.isnan(r[0]) else 0
    y = r[1] if not np.isnan(r[1]) else 0
    z = r[2] if not np.isnan(r[2]) else 0

    # radius
    r = np.sqrt(x**2 + y**2 + z**2)

    long = np.arctan2(y, x)
    if y < 0:
        long += 2 * np.pi

    lat = np.pi - np.arccos(z / r) - np.pi / 2

    # lat is between -90 and 90, long is between 0  and 360, and radius is altitude above sea level
    return lat * 180 / np.pi, long * 180 / np.pi, r - rad_e

# Convert magnetic field in spherical coords to rectangular
def Bsphere_to_Brect(Bsphere, r):
    """
    Bsphere is in [phi, theta, r0]
    r is [r, theta, phi]
    """
    phi = r[2] * np.pi / 180
    theta = r[1] * np.pi / 180
    Bphi = Bsphere[0]  if not np.isnan(Bsphere[0]) else 0
    Btheta = Bsphere[1] if not np.isnan(Bsphere[1]) else 0
    Br = Bsphere[2] if not np.isnan(Bsphere[2]) else 0

    Bx = Br * np.cos(theta) * np.cos(phi) - Btheta * np.sin(theta) * np.cos(phi) - Bphi * np.sin(phi)
    By = Br * np.cos(theta) * np.sin(phi) - Btheta * np.sin(theta) * np.sin(phi) + Bphi * np.cos(phi)
    Bz = Br * np.sin(theta) + Btheta * np.cos(theta)
    return np.array([Bx, By, Bz])


# Bang bang control method
def bangbangcontrol(Bdot):
    # B-dot Bang Bang Control method: u_i = -umax*sgn(dB/dt _i)
    if Bdot[0] > 0:
        ux = -umax
    else:
        ux = umax
    if Bdot[1] > 0:
        uy = -umax
    else:
        uy = umax
    if Bdot[2] > 0:
        uz = -umax
    else:
        uz = umax
    return np.array([-ux, -uy, -uz])


# Handles proportional control method
def proportionalcontrol(Bdot):
    return k_gain * Bdot


# Updates state
def update_entities(s, t):
    """
    Returns right-hand side of Kepler ODE; used by Runge-Kutta routines
    Inputs
        s      State vector [r(1) r(2) r(3) v(1) v(2) v(3)]
        t      Time (not used)
    Output
        deriv  Derivatives [dr(1)/dt dr(2)/dt dr(3)/dt dv(1)/dt dv(2)/dt dv(3)/dt]
    """
    objs_new = []

    input_objs = make_entities_from_state(s)
    
    # Sets up the s_new so that the accelecation begins at 0 and so we save the velocity
    for i in range(len(objs)):
        # objs_new is the derivative, so vel derivatives start at 0 unless code later defines an accelaration
        # Set up as (vel, accel, ang_vel, ang_accel, forward_direction, mass)
        objs_new.append(Entity(input_objs[i].get_vel(), np.zeros(3), input_objs[i].get_rot_vel(), 
                               np.zeros(3), np.zeros(3), objs[i].get_mass()))

    # Goes through each combination of objects and calculates their accelaration based on the other.
    # Adds up all the accelarations for the final result
    for i in range(len(objs) - 1):
        for j in range(i + 1, len(objs)):
            r1 = np.copy(input_objs[i].get_pos())
            r2 = np.copy(input_objs[j].get_pos())

            # Calculates displacement and accelaration
            r1to2 = r2 - r1
            accel2 = -G * objs[i].get_mass() * r1to2 / np.linalg.norm(r1to2) ** 3
            
            # Basic Dipole Earth version
            runit = r2 / np.linalg.norm(r2)
            B = u0 / (4 * np.pi * np.linalg.norm(r2)**3) * (3 * np.dot(uearth, runit) * runit - uearth)
            
            # Uses igrf library for magnetic field. Note: this takes super long so I decided against it
            # Convert rectangular coordinates to lat, long, alt coords
            # lat, long, alt = rect_to_lla(r2)
            # B_lla_temp = ppigrf.igrf(long, lat, alt - rad_e, rundate) # returns as [[B_east]], [[B_north]], [[B_up]]
            # B_lla = np.array([B_lla_temp[0][0][0], B_lla_temp[1][0][0], B_lla_temp[2][0][0]])
            # B = Bsphere_to_Brect(B_lla, np.array([alt + rad_e, lat, long]))

            omega = np.copy(input_objs[j].get_rot_vel())
            Bdot = np.cross(omega, B)
            
            # Since the Earth doesn't move in its local frame, we have to make sure we don't give it acceleration
            if i != 0:
                objs_new[j].add_vel(accel2)
            else:
                objs_new[j].add_vel(accel2)

                # Note, assuming that dB/dt (body) is negligible, so dB/dt (intertial) = dB/dt (body) + omega x B = omega x B
                u = control_method(Bdot)

                torque = np.cross(u, B)

                # torque = I * alpha + omega x (I*omega) = I * alpha. (I * omega is parallel to omega so omega x (parallel vector) = 0)
                OmegaDot = 1 / I[0,0] * torque # (since I is digonal and every value is the same, just divide each component)
                objs_new[j].set_rot_vel(OmegaDot)

            # if r2[0] < 0:
            #     objs_new[j].set_forward(np.array([0, 0, -1]))

    return_states = np.array([])
    for obj in objs_new:
        # for elem in obj.get_state():
        return_states = np.append(return_states, np.copy(obj.get_state()))

    return return_states


# Get Bdot control method
control_num = int(input('Select your control method  1 - proportional control, 2 - bang bang control : '))
if control_num == 1:
    control_method = proportionalcontrol
else:
    control_method = bangbangcontrol

# Get inclination
inc = float(input('Enter a number for the orbital inclination (between 1 and 90 degrees): '))
inc = min(inc, 90)
inc = max(1, inc)

# Get altitude
alt0 = float(input('Enter the altitude (km) above sea level (I recommend inbetween 200 and 1000): ')) * 1000

# Get angular velocity
ang_vel_str = input('Enter the rotational velocity (deg/s) vector with commas between numbers (ex: 0,5,0): ')
ang_vel_str = ''.join(filter(lambda ch : ch != ' ', ang_vel_str))
ang_vel_list = ang_vel_str.split(',')
ang_vel = np.array([float(ang_vel_list[0]), float(ang_vel_list[1]), float(ang_vel_list[2])])

# Get time step
tau = float(input("Enter time step (s): "))

G = 6.67408e-11
umax = 0.03
# rundate = datetime(2022, 4, 28)
num_coils = 84
enclosed_area = 0.02
voltage = 5
current_max = umax / num_coils / enclosed_area

# dipole field constants
uearth = np.array([0, 0, 6.48e22])
u0 = 1.256e-6


r_e0 = 0
rad_e = 6.3781e6
v_e0 = 0
M = 5.97219e24
earth = Entity(
    np.array([0, 0, 0]), 
    np.array([0, 0, 0]), 
    np.array([0, 1, 0]), 
    np.array([0, 0, 1]),
    np.array([0, 0, 1]),
    M
)

r_s = alt0 + rad_e
m_s = 2
side_s = .1
sat_speed = math.sqrt(G * M / r_s)
inclination = inc / 180 * np.pi
sat_rot = np.array([0, 0, 0])
sat_rot_vel = ang_vel / 180 * np.pi
satellite = Entity(
    np.array([r_s, 0, 0]), 
    np.array([0, sat_speed * np.cos(inclination), sat_speed * np.sin(inclination)]), 
    sat_rot,
    sat_rot_vel, 
    np.array([0, 0, 1]),
    m_s
)

I = m_s / 6 * np.matrix([[side_s**2, 0, 0], 
                        [0, side_s**2, 0],
                        [0, 0, side_s**2]])

r0unit = satellite.get_pos() / np.linalg.norm(satellite.get_pos())
B0 = u0 / (4 * np.pi * np.linalg.norm(satellite.get_pos())**3) * (3 * np.dot(uearth, r0unit) * r0unit - uearth)
omega0 = np.copy(satellite.get_rot_vel())
Bdot0 = np.cross(omega0, B0)
k_gain = umax / np.linalg.norm(Bdot0)

objs = [earth, satellite]

# Set physical parameters (mass, G*M)
adaptErr = 1.e-4  # Error parameter used by adaptive Runge-Kutta
time = 0.0

maxsteps = 30000
prevTheta = 0
a = 0
b = float('inf')
count = 0

tplot = np.array([])
step = 1
vrot_mag = np.linalg.norm(objs[1].get_rot_vel() * 180 / np.pi)
freq_inverse = 10 # frequency for recording would be 1 / X, freq inverse is X

print()
while(600 > vrot_mag > .1):

    tplot = np.append(tplot, time)

    state = np.array([])
    for obj in objs:
        state = np.append(state, np.copy(obj.get_state()))
    prev_y = state[13]
    # Calculate new position and velocity using rka.
    state = rk4(state, time, tau, update_entities)
    # [state, time, tau] = rka(state, time, tau, adaptErr, update_entities)
    new_y = state[13]
    if prev_y < 0 and new_y >= 0:
        break

    outofbounds = False

    record = False
    if step % freq_inverse == 0:
        record = True

    for i in range(len(objs)):

        # Update state values
        objs[i].set_pos(state[i * 12: i * 12 + 3], record)
        objs[i].set_vel(state[i * 12 + 3: i * 12 + 6])
        objs[i].set_rot(state[i * 12 + 6: i * 12 + 9], record)
        objs[i].set_rot_vel(state[i * 12 + 9: i * 12 + 12], record)

        # Record time
        if record:
            objs[i].append_time(time + tau)

        # Record power measurement for satellite
        if i != 0 and record:
            runit = objs[i].get_pos() / np.linalg.norm(objs[i].get_pos())
            B = u0 / (4 * np.pi * np.linalg.norm(objs[i].get_pos())**3) * (3 * np.dot(uearth, runit) * runit - uearth)
            omega = np.copy(objs[i].get_rot_vel())
            Bdot = np.cross(omega, B)
            u = control_method(Bdot)
            current = u / num_coils / enclosed_area
            objs[i].append_powers(np.linalg.norm(current * voltage))

        # Checks if the satellite is flying out of orbit
        if abs(np.linalg.norm(objs[i].get_pos())) > r_s * 1.05:
            outofbounds = True
        
    vrot_mag = np.linalg.norm(objs[1].get_rot_vel() * 180 / np.pi)
    time = time + tau

    # Whether or not to print out it's iteration
    if step % 100 == 0:
        print('Time:', time, ', Rotational velocity magnitude:', vrot_mag)

    step += 1
    if outofbounds:
        print('You found some inputs that make this go crazy. Well done. It is broken now.')
        break
if vrot_mag < 0.1:
    print('Satellite detumbled successfully')
else:
    print('Satellite could not detumble, likely the initial rotational velocity was too high')

# Create rotation arrows
rays = objs[1].get_pos_rays_dir()
X, Y, Z, U, V, W = zip(*rays)
point_times = objs[1].get_times()
C = np.array(colorsys.hsv_to_rgb(0, 1.0, 1.0))
for t in point_times[1:]:
    rgb = colorsys.hsv_to_rgb(t / time, 1.0, 1.0)
    C = np.vstack((C, rgb))

# Calculate total energy consumed
powers = objs[1].get_powers()
energy = 0

for i in range(1, len(powers)):
    prev_p = powers[i - 1]
    curr_p = powers[i]
    time_step = point_times[i] - point_times[i - 1]

    tri_area = abs(curr_p - prev_p) / 2 * time_step

    min_p = min(prev_p, curr_p)
    square_area = min_p * time_step
    energy += square_area + tri_area

print('The total energy is:', energy)


rot_vels = objs[1].get_rot_vels()
rot_mags = np.array([])
for i in rot_vels:
    rot_mags = np.append(rot_mags, np.linalg.norm(i))

fig1 = plt.figure(1)
fig1.clf()
plt.scatter(point_times, rot_mags * 180 / np.pi)
plt.xlabel('Time (s)')
plt.ylabel('Magnitude of rotational velocity')
plt.title('Rotational Velocity vs Time')

fig2 = plt.figure(2)
fig2.clf()
plt.scatter(point_times, objs[1].get_powers())
plt.xlabel('Time (s)')
plt.ylabel('Power required')
plt.title('Required Power vs Time')

# Create earth and add it to the figure
u, v = np.mgrid[0:2 * np.pi:30j, 0:np.pi:20j]
x = np.cos(u) * np.sin(v) * rad_e
y = np.sin(u) * np.sin(v) * rad_e
z = np.cos(v) * rad_e

fig = plt.figure(3)
fig.clf()
ax=plt.axes(projection='3d')
# Draw the earth
ax.plot_surface(x, y, z, rstride=1, cstride=1)
# Draw the rotation arrows at each position
ax.quiver(X, Y, Z, U, V, W, colors=C, length=1500000, normalize=True)
ax.scatter(X[0], Y[0], Z[0], color=(0, 0, 0)) # Notes start point
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.title('Satellite path around the earth with its direction')
plt.show()
