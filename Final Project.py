#%%
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy as sp
from mpl_toolkits.mplot3d import Axes3D

rng = np.random.default_rng()

#%%
'''
Setup: There is a solid sphere with radius 5m made of carbon sitting 5m above the ground.
At the ground level, there is a square detector 20m per side. 5m above the sphere, there is
another detector of the same size such that both detectors are aligned and the sphere is
along the central axis. 
'''
#%% Constants
sphere_radius = 5 #m
sphere_loc = 0 #m
top_detector_loc = 10 #m
bottom_detector_loc = -10 #m
detector_dist = top_detector_loc - bottom_detector_loc #m
detector_side_length = 10 #m
detection_time_scale = 10 #s
detector_x_coords = np.linspace(-(detector_side_length/2), detector_side_length/2, 1000)
detector_y_coords = np.linspace(-(detector_side_length/2), detector_side_length/2, 1000)
detector_resolution = 0.05 #m, bin width

Z_nucl = 6
Z_mu = 1
A = 12 # g/mol
A_r = 12.011 # amu
rho = 2 # g/cm^3
N_A = 6.022 * 10**23
r_0 = 2.818 * 10**-13 # cm
m_e = 0.511 * 10**6 # eV
c = 2.998 * 10**8 # m/s
K = 4 * np.pi * N_A * r_0**2 * m_e # cm^2 eV
m_mu = 105.66 * 10**6 # eV
I = 10 * Z_nucl #eV
E_p = np.sqrt(rho * Z_nucl / A) * 28.816 # eV, plasma energy
alpha = 1 / 137
L_rad = np.log(184.15 * Z_nucl**(-1/3))
L_rad_prime = np.log(1194 * Z_nucl**(-2/3))
a = alpha * Z_nucl
f_Z = a**2 * ((1 + a**2)**(-1) + 0.20206 - (0.0369 * a**2) + (0.0083 * a**4) - (0.002 * a**6))
inverse_rad_length = 4 * alpha * r_0**2 * (N_A / A) * (Z_nucl**2 * (L_rad - f_Z) + Z_nucl * L_rad_prime) #cm^2/g
rad_length = 1 / inverse_rad_length #g/cm^2

#%% Muon Flux and Distribution Functions
total_muon_intensity = 1 * 100**2 / 60 #number per cm^2 per min -> number per m^2 per s
muon_flux = round(total_muon_intensity * detector_side_length**2 * detection_time_scale)

'''
The intensity is distributed along the zenith angle by (total_muon_intensity/3)*cos(theta)^2 function.
To make this a probability distribution, I will integrate from 0 to pi/2 and set equal to 1 to find a
constant normalization factor.
'''

intensity_normalization_factor = 12 / (np.pi * total_muon_intensity)

def zenith_pdf(theta):
    return intensity_normalization_factor * (total_muon_intensity / 3) * np.cos(theta)**2

mean_muon_energy = 4e9 #eV
power_law_exponent = 2.7
minimum_energy = (mean_muon_energy * (power_law_exponent - 2)) / (power_law_exponent - 1)

def energy_pdf_power_law(energy):
    return ((power_law_exponent - 1) / (minimum_energy)) * (energy / minimum_energy)**(-power_law_exponent)

def energy_pdf_exponential(energy):
    return (1 / mean_muon_energy) * np.exp(-energy / mean_muon_energy)

#%% Calculator Functions

def beta_from_vel(vel):
    return vel / c

def gamma_from_vel(vel):
    return 1 / np.sqrt(1-beta_from_vel(vel)**2)

def p_from_vel(vel):
    beta = beta_from_vel(vel)
    gamma = gamma_from_vel(vel)
    return beta * gamma * m_mu

def vel_from_energy(energy):
    return c * np.sqrt(energy * (energy + (2 * m_mu))) / (energy + m_mu)

def T_max(vel):
    beta = beta_from_vel(vel)
    gamma = gamma_from_vel(vel)
    
    return (2 * m_e * beta**2 * gamma**2) / (1 + (2 * gamma * m_e / m_mu) + (m_e / m_mu)**2) #eV

def delta(vel):
    beta = beta_from_vel(vel)
    gamma = gamma_from_vel(vel)
    
    return 2 * (np.log(E_p / I) + np.log(beta * gamma) - 0.5)

def bethe_bloch(energy):
    vel = vel_from_energy(energy)
    
    beta = beta_from_vel(vel)
    gamma = gamma_from_vel(vel)
    
    dE_dx = rho * (Z_nucl / A_r) * (0.307) * (Z_mu**2 / beta**2) * (0.5 * np.log(2 * m_e * beta**2 * gamma**2 * T_max(vel) / I**2) - beta**2 - (delta(vel) / 2)) # MeV/cm
    
    dE_dx = dE_dx * 10**6 * 100 # MeV/cm * 10^6 ev/MeV * 100 cm/m = eV/m
    
    return dE_dx

def scattering_angle_std(energy, path_length):
    x = rho * path_length * 100 #g/cm^3 * m * cm/m = g/cm^2
    vel = vel_from_energy(energy)
    beta = beta_from_vel(vel)
    p = p_from_vel(vel)
    
    return (13.6e6 / (beta * p)) * Z_mu * np.sqrt(x / rad_length) * (1 + 0.038 * np.log(x / rad_length))
    

#%% Muography Functions

def find_sphere_contact_points(xi, yi, xf, yf):
    zi = top_detector_loc
    zf = bottom_detector_loc
    
    xc = 0
    yc = 0
    zc = sphere_loc
    rad = sphere_radius
    
    #Coefficients of parameterized quadratic At^2 + Bt + C = 0
    quad_C = (xi-xc)**2 + (yi-yc)**2 + (zi-zc)**2 - rad**2
    quad_A = (xi-xf)**2 + (yi-yf)**2 + (zi-zf)**2
    quad_B = (xf-xc)**2 + (yf-yc)**2 + (zf-zc)**2 - quad_A - quad_C - rad**2
    
    t_roots = np.roots([quad_A, quad_B, quad_C])
    
    if len(t_roots) > 1 and np.all(np.isreal(t_roots)):
        x_coords = (xi * (1 - t_roots)) + (t_roots * xf)
        y_coords = (yi * (1 - t_roots)) + (t_roots * yf)
        z_coords = (zi * (1 - t_roots)) + (t_roots * zf)
    else:
        x_coords = [0,0]
        y_coords = [0,0]
        z_coords = [0,0] #This is impossible, used to signify muon did not pass through sphere
    
    return x_coords, y_coords, z_coords

def sphere_path_length_alt(x_coords, y_coords, z_coords):
    return np.sqrt((x_coords[0]-x_coords[1])**2+(y_coords[0]-y_coords[1])**2+(z_coords[0]-z_coords[1])**2)

def stopping_range_integrand(energy):
    return 1 / bethe_bloch(energy)

def stopping_range(init_energy):
    return sp.integrate.quad(stopping_range_integrand, 0, init_energy)[0]

def draw_scattering_angle(energy, path_length):
    sigma = scattering_angle_std(energy, path_length)
    
    return rng.normal(loc=0, scale=sigma)

class muon:
    def __init__(self):
        self.init_x = 0 #m
        self.init_y = 0 #m
        self.init_E = 0 #eV
        self.init_zen = 0 #rad
        self.init_az = 0 #rad
        self.fin_x = 0 #m
        self.fin_y = 0 #m
        self.fin_zen = 0 #rad
        self.fin_az = 0 #rad
        self.reaches_second_detector = True
        self.path_length = 0 #m
        self.sphere_intersections = 0 #coords in m
        
    def __str__(self):
        return f"Muon:\n\tInitial position (x,y): ({self.init_x}, {self.init_y})\n\tInitial energy: {self.init_E}"

def generate_incoming_muons():
    print("Generating incoming muons")
    incoming_muons = []
    
    possible_energies = np.linspace(minimum_energy, 1e12, num=10000)
    possible_zeniths = np.linspace(0, np.pi/2, num=1000)
    possible_azimuths = np.linspace(0, 2*np.pi, num=1000)

    energy_probs = energy_pdf_power_law(possible_energies) * np.abs(np.mean(np.diff(possible_energies)))
    zenith_probs = zenith_pdf(possible_zeniths) * np.abs(np.mean(np.diff(possible_zeniths)))

    energy_probs /= np.sum(energy_probs)
    zenith_probs /= np.sum(zenith_probs)
    
    for m in range(0, muon_flux):
        if m % round(muon_flux / 10) == 0: print(f"{m/muon_flux*100}% of incoming muons generated")
        new_muon = muon()
        new_muon.fin_x = rng.choice(detector_x_coords)
        new_muon.fin_y = rng.choice(detector_y_coords)
        new_muon.init_E = rng.choice(possible_energies, p=energy_probs)
        new_muon.init_zen = rng.choice(possible_zeniths, p=zenith_probs)
        new_muon.init_az = rng.choice(possible_azimuths)

        delta_z = top_detector_loc - bottom_detector_loc
        r_coord = delta_z / np.cos(new_muon.init_zen)
        delta_x = r_coord * np.sin(new_muon.init_zen) * np.cos(np.pi - new_muon.init_az)
        delta_y = r_coord * np.sin(new_muon.init_zen) * np.sin(np.pi - new_muon.init_az)

        new_muon.init_x = new_muon.fin_x - delta_x
        new_muon.init_y = new_muon.fin_y - delta_y
        
        incoming_muons.append(new_muon)
        
    return incoming_muons

def cut_decayed_muons(muons):
    print("Cutting muons that decay traversing sphere")
    surviving_muons = []
    for muon in range(0, len(muons)):
        if muon % round(muon_flux / 10) == 0: print(f"{muon/muon_flux*100}% of muons passed through sphere")
        m = muons[muon]
        x_coords, y_coords, z_coords = find_sphere_contact_points(m.init_x, m.init_y, m.fin_x, m.fin_y)
        m.sphere_intersections = np.array([x_coords, y_coords, z_coords])
        muon_sphere_path_length = sphere_path_length_alt(x_coords, y_coords, z_coords)
        muon_stopping_range = stopping_range(m.init_E)
        m.path_length = muon_sphere_path_length
        if muon_stopping_range > muon_sphere_path_length:
            surviving_muons.append(m)
        else:
            m.reaches_second_detector = False
            
    return surviving_muons
    
def determine_final_muon_locations(muons):
    print("Determining muons' final locations")
    m: muon
    for m in muons:
        if m.path_length != 0:
            angle_x = draw_scattering_angle(m.init_E, m.path_length)
            angle_y = draw_scattering_angle(m.init_E, m.path_length)
        
            delta_zen = rng.choice([-1,1]) * np.sqrt(angle_x**2 + angle_y**2)
            delta_az = np.arctan(angle_y / angle_x)
            
            if m.init_zen + delta_zen >= 0:
                m.fin_zen = m.init_zen + delta_zen
                m.fin_az = m.init_az + delta_az
            else:
                m.fin_zen = np.abs(m.init_zen + delta_zen)
                m.fin_az = m.init_az + delta_az + np.pi
            
            if m.sphere_intersections[2,0] > m.sphere_intersections[2,1]:
                exit_index = 1
            else:
                exit_index = 0
                
            delta_z = np.abs(bottom_detector_loc - m.sphere_intersections[2,exit_index])
            rad = delta_z / np.cos(m.fin_zen)
            delta_x = rad * np.sin(m.fin_zen) * np.cos(m.fin_az)
            delta_y = rad * np.sin(m.fin_zen) * np.sin(m.fin_az)
            
            m.fin_x = m.sphere_intersections[0, exit_index] + delta_x
            m.fin_y = m.sphere_intersections[1, exit_index] + delta_y
            
            if np.abs(m.fin_x) > 5 or np.abs(m.fin_y) > 5:
                m.reaches_second_detector = False
        
def cut_scattered_muons(muons):
    print("Cutting muons scattered at too large an angle")
    surviving_muons = []
    
    for m in muons:
        if m.reaches_second_detector:
            surviving_muons.append(m)
            
    return surviving_muons        

def plot_detected_muon_flux(muons):
    muon_x_locs = []
    muon_y_locs = []
    
    for m in muons:
        muon_x_locs.append(m.fin_x)
        muon_y_locs.append(m.fin_y)
    
    muon_x_locs = np.array(muon_x_locs)
    muon_y_locs = np.array(muon_y_locs)
    
    num_detector_bins = int(detector_side_length / detector_resolution)
    
    total_detected_flux = len(muons)
    possible_indices = range(0, len(muons))
    flux_scenarios = [0.01, 0.1, 1]
    
    for flux in flux_scenarios:
        considered_muons = rng.choice(possible_indices, round(flux*total_detected_flux), replace=False)
        
        hist, xedges, yedges = np.histogram2d(muon_x_locs[considered_muons], muon_y_locs[considered_muons], bins=[num_detector_bins, num_detector_bins], range=[[detector_x_coords[0], detector_x_coords[-1]], [detector_y_coords[0], detector_y_coords[-1]]])
        
        num_pixels = num_detector_bins**2
        expected_flux_per_pixel = round(flux*(muon_flux / num_pixels))
        
        hist -= expected_flux_per_pixel
        
        im = plt.imshow(hist, extent=[-detector_side_length/2, detector_side_length/2, -detector_side_length/2, detector_side_length/2], origin='lower')
        plt.colorbar(im, label="Number of Incident Muons above Expected")
        # sphere = mpl.patches.Circle((0,0), 5, edgecolor='black', facecolor='none', linewidth=1)
        # plt.gca().add_patch(sphere)
        plt.xlabel("$x$ Position [m]")
        plt.ylabel("$y$ Position [m]")
        plt.title(f"Muon Flux Excess at Detector after $t=${round(flux*detection_time_scale)} s")
        plt.show()
    
def plot_scattering_locations(muons):
    doubly_detected_muons = []
    
    m: muon
    for m in muons:
        if np.abs(m.init_x) < detector_side_length/2 and np.abs(m.init_y) < detector_side_length/2:
            doubly_detected_muons.append(m)
    
    print(f"Number of doubly detected muons: {len(doubly_detected_muons)}")
    muons_subset = rng.choice(doubly_detected_muons, round(0.3*len(doubly_detected_muons)), replace=False)
    print(f"Number of used muons for scattering plots: {len(muons_subset)}")
    
    muon_scatter_x_locs = []
    muon_scatter_y_locs = []
    muon_scatter_z_locs = []
    
    m: muon
    for m in muons_subset:
        if m.sphere_intersections[2,0] > m.sphere_intersections[2,1]:
                exit_index = 1
        else:
                exit_index = 0
                
        muon_scatter_x_locs.append(m.sphere_intersections[0,exit_index])
        muon_scatter_y_locs.append(m.sphere_intersections[1,exit_index])
        muon_scatter_z_locs.append(m.sphere_intersections[2,exit_index])
        
    muon_scatter_x_locs = np.array(muon_scatter_x_locs)
    muon_scatter_y_locs = np.array(muon_scatter_y_locs)
    muon_scatter_z_locs = np.array(muon_scatter_z_locs)
    
    z_bounds = np.linspace(0, -5, num=100)
    for z in range(0, len(z_bounds)-1):
        indices = np.intersect1d(np.where(muon_scatter_z_locs < z_bounds[z]), np.where(muon_scatter_z_locs > z_bounds[z+1]))
        if z % 20 == 0:
            plt.scatter(muon_scatter_x_locs[indices], muon_scatter_y_locs[indices], s=2, c='black', alpha=1-z/100, label=f"z={round(z_bounds[z])}m")
        else:
            plt.scatter(muon_scatter_x_locs[indices], muon_scatter_y_locs[indices], s=2, c='black', alpha=1-z/100)
    plt.xlim(-5.5, 5.5)
    plt.ylim(-5.5, 5.5)
    plt.gca().set_aspect('equal')
    plt.xlabel("$x$ Position")
    plt.ylabel("$y$ Position")
    plt.title("Scattering Locations Looking from the Top Down")
    plt.legend(loc='lower right')
    plt.show()
    
    for z in range(0, len(z_bounds)-1):
        indices = np.intersect1d(np.where(muon_scatter_z_locs < z_bounds[z]), np.where(muon_scatter_z_locs > z_bounds[z+1]))
        if z % 20 == 0:
            plt.scatter(muon_scatter_x_locs[indices], muon_scatter_z_locs[indices], s=2, c='black', alpha=1-z/100, label=f"z={round(z_bounds[z])}m")
        else:
            plt.scatter(muon_scatter_x_locs[indices], muon_scatter_z_locs[indices], s=2, c='black', alpha=1-z/100)
    plt.xlim(-5.5, 5.5)
    plt.ylim(-5.5, 0.5)
    plt.gca().set_aspect('equal')
    plt.xlabel("$x$ Position")
    plt.ylabel("$z$ Position")
    plt.title("Scattering Locations Looking from the $y$-axis")
    plt.legend(loc='lower right')
    plt.show()
    
    fig = plt.figure(figsize=(7, 7), dpi=200)
    ax = fig.add_subplot(projection='3d')
    scatter = ax.scatter(muon_scatter_x_locs, muon_scatter_y_locs, muon_scatter_z_locs, c=muon_scatter_z_locs)
    
    ax.set_xlabel("$x$ Position [m]")
    ax.set_ylabel("$y$ Position [m]")
    ax.set_zlabel("$z$ Position [m]")
    ax.set_title("Scattering Locations", y=0.98)
    ax.set_box_aspect(aspect=None, zoom=0.9)
    plt.show()

def plot_stopping_range():
    energies = np.logspace(8, 12, 100)
    stopping_ranges = []
    for e in energies:
        stopping_ranges.append(stopping_range(e))
    
    fig, ax1 = plt.subplots()
    ax1.plot(energies, stopping_ranges, label="Stopping Range")
    ax1.fill_between(energies, 0, 10, color='red', alpha=0.25, label='Possible Sphere Path Lengths')
    ax1.vlines(1.647e9, 0, 1e3, linestyle='--', label="Minimum Simulated Muon Energy", color='green')
    ax1.vlines(4e9, 0, 1e3, linestyle='--', label="Mean Muon Energy", color='purple')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('Energy [eV]')
    ax1.set_ylabel('Stopping Range [m]')
    
    ax2 = ax1.twinx()
    energies2 = np.linspace(1.647e9, 1e12, 100)
    ax2.plot(energies2, energy_pdf_power_law(energies2), linestyle=':', color='black', label='Muon Energy PDF')
    ax2.set_yscale('log')
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("Probability of Muon Energy")
    
    ax1.set_title('Stopping Range as Function of Incident Energy')
    ax1.legend()
    ax2.legend()
    plt.show()
    
def plot_scattering_angle_stds():
    energies = np.geomspace(1e9, 1e12, 100)
    path_lengths = [0.1, 1, 5, 10]
    all_stds = []
    for path in path_lengths:
        stds = scattering_angle_std(energies, path)
        all_stds.append(stds)
        
    plt.plot(energies, all_stds[0], label='0.1 m Path Length')
    plt.plot(energies, all_stds[1], label='1 m Path Length')
    plt.plot(energies, all_stds[2], label='5 m Path Length')
    plt.plot(energies, all_stds[3], label='10 m Path Length')
    plt.vlines(1.647e9, 0, 0.1, linestyle='--', label="Minimum Simulated Muon Energy", color='pink')
    plt.vlines(4e9, 0, 0.1, linestyle='--', label="Mean Muon Energy", color='purple')
    plt.xscale('log')
    plt.xlabel('Energy [eV]')
    plt.ylabel(r'Scattering Angle St. Dev. $\sigma_{\theta}$ [rad]')
    plt.title('Scattering Angle Standard Deviations for Varying Path Lengths')
    plt.legend()
    plt.show()
    

#%%

muon_array = generate_incoming_muons()
cut_muon_array = cut_decayed_muons(muon_array)
determine_final_muon_locations(cut_muon_array)
cut_muon_array = cut_scattered_muons(cut_muon_array)
plot_detected_muon_flux(cut_muon_array)

# %%
