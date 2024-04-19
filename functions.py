import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import h5py
from astropy import units as u
from scipy.fftpack import fft2
from functions import *

def get_data(file_path):
    
    with h5py.File(file_path, "r") as f:
        dataset = f["FuzzyDM/DENSITY"]
        fdm_density = np.array(dataset)
        #print(np.shape(fdm_density))
        
    return fdm_density

def find_radius(fdm_3d_density):
    
    h = 0.7
    pix_size = 1.16 * u.kpc / h # Pixel size

    L = np.shape(fdm_3d_density)[0]
    R_200 = L/4 * pix_size
    
    return R_200

def find_mass(fdm_3d_density):
    h = 0.7
    pix_size = 1.16 * u.kpc / h  # Pixel size

    rho = 10 **10 * h **2 * u.solMass / (u.kpc)**3

    L = fdm_3d_density.shape[0]
    R_200 = L / 4
    ctr = L / 2

    i, j, k = np.indices(fdm_3d_density.shape)
    dist_to_ctr = np.sqrt((i - ctr)**2 + (j - ctr)**2 + (k - ctr)**2)

    mask = dist_to_ctr < R_200
    mass = np.sum(fdm_3d_density[mask])

    M_200 = mass * rho * (pix_size)**3

    return M_200

# def find_mass(fdm_3d_density):
    
#     h = 0.7
#     pix_size = 1.16 * u.kpc / h # Pixel size

#     # Units of fuzzy dark matter density
#     rho = 10**10 * h**2 * u.solMass/(u.kpc)**3
    
#     L = np.shape(fdm_3d_density)[0]
#     R_200 = L/4 
#     ctr = L/2
    
#     mass = 0
    
#     # Now, we want to compute the mass within this radius.
#     for i in range(L):
#         for j in range(L):
#             for k in range(L):
                                            
#                 dist_to_ctr = np.sqrt((i-ctr)**2 + (j-ctr)**2 + (k-ctr)**2)
                
#                 # If distance to center is less than R_200, we add up the density in this pixel.
                
#                 if dist_to_ctr < R_200:
                    
#                     mass += fdm_3d_density[i][j][k] 
    
#     # Fixing the mass units
#     M_200 = mass * rho * (pix_size)**3
    
#     return M_200

def get_mass_rad(catalog_path, halo_ID):
    mass_rad_catalog = np.loadtxt(catalog_path)
    halo_IDs = []
    halo_masses = []
    halo_radii = []
    for line in mass_rad_catalog:
        halo_IDs.append(int(line[0]))
        halo_masses.append(line[1])
        halo_radii.append(line[2])
    halo_indx = halo_IDs.index(halo_ID)
    halo_mass = halo_masses[halo_indx]
    halo_radius = halo_radii[halo_indx]
    return halo_mass, halo_radius

def get_projections(fdm_density):

    # Create simple 2D projections along x, y, and z axes
    x_proj = np.sum(fdm_density, axis=0)
    y_proj = np.sum(fdm_density, axis=1)
    z_proj = np.sum(fdm_density, axis=2)
        
    return x_proj, y_proj, z_proj

def plot_2d_projections(x_proj, y_proj, z_proj, halo_ID):

    f, ax = plt.subplots(1, 3, figsize=(30, 9))
    ax[0].imshow(x_proj, cmap='viridis', origin='lower', norm=LogNorm())
    ax[1].imshow(y_proj, cmap='viridis', origin='lower', norm=LogNorm())
    ax[2].imshow(z_proj, cmap='viridis', origin='lower', norm=LogNorm())

    # Add colorbars and labels
    cbar = f.colorbar(ax[0].imshow(x_proj, cmap='viridis', origin='lower', norm=LogNorm()), ax=ax[0])
    cbar.set_label('Density')
    ax[0].set_title('X-axis Projection')
    ax[0].set_xlabel('Y-axis')
    ax[0].set_ylabel('Z-axis')

    cbar = f.colorbar(ax[1].imshow(y_proj, cmap='viridis', origin='lower', norm=LogNorm()), ax=ax[1])
    cbar.set_label('Density')
    ax[1].set_title('Y-axis Projection')
    ax[1].set_xlabel('X-axis')
    ax[1].set_ylabel('Z-axis')

    cbar = f.colorbar(ax[2].imshow(z_proj, cmap='viridis', origin='lower', norm=LogNorm()), ax=ax[2])
    cbar.set_label('Density')
    ax[2].set_title('Z-axis Projection')
    ax[2].set_xlabel('Y-axis')
    ax[2].set_ylabel('X-axis')

    plt.suptitle('2D Projections of Density (Log Scale)')
    f.savefig("visualize_halo_"+halo_ID+".png")
    plt.show()

def visualize_projs(density_fields, halo_ID):
    N = len(density_fields)
    ncols = int(np.ceil(np.sqrt(N)))
    nrows = int(np.ceil(N / ncols))
    
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(9 * ncols, 9 * nrows))
    axes = axes.flatten()
    
    for i, proj_density in enumerate(density_fields):
        ax = axes[i]
        im = ax.imshow(proj_density, cmap='viridis', origin='lower', norm=LogNorm())
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Density')
        ax.set_title(f'Projected Density {i+1}')
        ax.set_xlabel('Y-axis')
        ax.set_ylabel('X-axis')
    
    for ax in axes[len(density_fields):]:
        ax.axis('off')
    
    plt.tight_layout()
    #fig.savefig("all_projections_"+halo_ID+".png")
    plt.show()
    
def visualize_substructure(density_fields, halo_ID):
    N = len(density_fields)
    ncols = int(np.ceil(np.sqrt(N)))
    nrows = int(np.ceil(N / ncols))
    
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(9 * ncols, 9 * nrows))
    axes = axes.flatten()
    
    for i, proj_density in enumerate(density_fields):
        ax = axes[i]
        im = ax.imshow(proj_density, cmap='viridis', origin='lower', norm=LogNorm())
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Density')
        ax.set_title(f'Projected Density {i+1}')
        ax.set_xlabel('Y-axis')
        ax.set_ylabel('X-axis')
    
    for ax in axes[len(density_fields):]:
        ax.axis('off')
    
    plt.tight_layout()
    #fig.savefig("all_substructure_"+halo_ID+".png")
    plt.show()
    
def visualize_host(proj_density, halo_ID):
    fig, ax = plt.subplots(figsize=(9, 9))
    im = ax.imshow(proj_density, cmap='viridis', origin='lower', norm=LogNorm())
    cbar = fig.colorbar(im)
    cbar.set_label('Density')
    ax.set_title('Projected Density')
    ax.set_xlabel('Y-axis')
    ax.set_ylabel('X-axis')
    fig.savefig("host_"+halo_ID+".png")
    plt.show()

    
def visualize_3d(fdm_3d_density, halo_ID):
    x_proj, y_proj, z_proj = get_projections(fdm_3d_density)
    plot_2d_projections(x_proj, y_proj, z_proj, halo_ID)    

# Create a spherical mask around the halo so we can rotate and average density field
def make_mask(fdm_3d_density, radius):
    
    L = np.shape(fdm_3d_density)[0]
    
    for i in range(L):
        for j in range(L):
            for k in range(L):
                
                ctr = L/2
                dist_from_center = np.sqrt((i - ctr)**2 + (j - ctr)**2 + (k - ctr)**2)
                
                if dist_from_center > radius:
                    
                    # Make this Boolean instead
                    
                    fdm_3d_density[i][j][k] = 0.0
                    
    return fdm_3d_density


def make_mask_bool(L, mask_radius):
    # Create a 3D grid of coordinates
    x, y, z = np.ogrid[:L, :L, :L]
    ctr = L / 2
    
    # Calculate distances from the center
    dist_from_center = np.sqrt((x - ctr) ** 2 + (y - ctr) ** 2 + (z - ctr) ** 2)
    
    # Create the mask based on the distance condition
    mask = dist_from_center < mask_radius
    
    return mask

def rotation(nx,ny,nz,theta):
    R = np.array([np.array([np.cos(theta) + (nx**2)*(1-np.cos(theta)) , nx*ny*(1-np.cos(theta)) - nz*np.sin(theta) , nx*nz*(1-np.cos(theta)) + ny*np.sin(theta)]),
                  np.array([nx*ny*(1-np.cos(theta)) + nz*np.sin(theta) , np.cos(theta) + (ny**2)*(1-np.cos(theta)) , ny*nz*(1-np.cos(theta)) - nx*np.sin(theta)]),
                  np.array([nz*nx*(1-np.cos(theta)) - ny*np.sin(theta) , nz*ny*(1-np.cos(theta)) + nx*np.sin(theta) , np.cos(theta) + (nz**2)*(1-np.cos(theta))])])
    return R

# Generate projection 
def transform_3d_density(R, fdm_3d_density, mask_radius):
    
    L = np.shape(fdm_3d_density)[0]
    ctr = L/2

    transformed_density = np.zeros((L, L, L))
    
    for i in range(L):
        for j in range(L):
            for k in range(L):
                # If distance from halo is within 100 pixels, we will rotate and transform, rest is left untouched
                pos_vector = np.array([i-ctr, j-ctr, k-ctr])
                
                dist_from_center = np.sqrt(pos_vector[0]**2 + pos_vector[1]**2 + pos_vector[2]**2)
                
                if dist_from_center < mask_radius:
                    trf_pos_vector = np.dot(R, pos_vector)
                    i_tr, j_tr, k_tr = int(trf_pos_vector[0]+ctr), int(trf_pos_vector[1]+ctr), int(trf_pos_vector[2]+ctr)
                    
                    transformed_density[i][j][k] = fdm_3d_density[i_tr][j_tr][k_tr]

    return transformed_density


def transform_3d_density_efficient(R, fdm_3d_density, mask_radius):
    L = fdm_3d_density.shape[0]
    ctr = L // 2

    x, y, z = np.meshgrid(np.arange(L) - ctr, np.arange(L) - ctr, np.arange(L) - ctr, indexing='ij')
    pos_vectors = np.stack([x, y, z], axis=-1)
    dist_from_center = np.sqrt(np.sum(pos_vectors ** 2, axis=-1))
    mask = dist_from_center < mask_radius

    masked_density = fdm_3d_density[mask]

    trf_pos_vectors = np.dot(pos_vectors[mask], R.T)
    trf_indices = np.round(trf_pos_vectors + ctr).astype(int)

    transformed_density = np.zeros((L, L, L))
    transformed_density[trf_indices[:, 0], trf_indices[:, 1], trf_indices[:, 2]] = masked_density

    return transformed_density


# Make N_proj random projections of the density field
def make_random_projections(N_proj, fdm_3d_density, mask_radius, halo_num):

    all_random_projs = []
    
    for i in range(N_proj):
        # Generate random vector components
        nnx = np.random.uniform(0,10)
        nny = np.random.uniform(0,10)
        nnz = np.random.uniform(0,10)
        theta = np.random.uniform(0,2*np.pi)

        # Normalize vector components to get a unit normal vector to our plane of projection
        nx = 1/np.sqrt(nnx**2 + nny**2 + nnz**2) * nnx
        ny = 1/np.sqrt(nnx**2 + nny**2 + nnz**2) * nny
        nz = 1/np.sqrt(nnx**2 + nny**2 + nnz**2) * nnz

        R = rotation(nx,ny,nz,theta)

        transformed_density = transform_3d_density_efficient(R, fdm_3d_density, mask_radius)
        #print("Projection #:", i)
        x_proj, y_proj, z_proj = get_projections(transformed_density)
        #plot_2d_projections(x_proj, y_proj, z_proj)    
        #np.save("Proj_"+str(i)+"_x.npy", x_proj)
        #np.save("Proj_"+str(i)+"_y.npy", y_proj)
        #np.save("Proj_"+str(i)+"_z.npy", z_proj)

        all_random_projs.append(x_proj)
        all_random_projs.append(y_proj)
        all_random_projs.append(z_proj)
        
    #np.save("all_projections_"+halo_num+".npy", all_random_projs)
    
    return all_random_projs

# Find the smooth density profile of host by averaging all the random realizations of projected density field

def make_host(all_random_projs, halo_num):

    allprojs = all_random_projs

#     for i in range(N_proj):
#         x_proj = np.load("all_projs/Proj_"+str(i)+"_x.npy")
#         allprojs.append(x_proj)
#         y_proj = np.load("all_projs/Proj_"+str(i)+"_y.npy")
#         allprojs.append(y_proj)
#         z_proj = np.load("all_projs/Proj_"+str(i)+"_z.npy")
#         allprojs.append(z_proj)
#    print(np.shape(allprojs))

    host = np.mean(allprojs,axis=0)
    #plot_projection(host)
    #np.save("host_k0_"+halo_num+".npy", host)
    
    return host

# Subtract host halo from all the projections

def find_substructure(host, allprojs, halo_num):

    all_substructure = []

    for i in range(len(allprojs)):
        proj = allprojs[i]

        substrcuture = proj - host
        all_substructure.append(substrcuture)
        #np.save("all_substructure_"+halo_num+".npy", all_substructure)
        
    return all_substructure


# COMPUTING POWER SPECTRUM

def compute_power_spectrum(density_field):
    """
    Compute the power spectrum of the density field.
    """
    # Compute the Fourier transform of the projected density field
    density_fft = fft2(density_field)
    
    # Compute the power spectrum
    power_spectrum = np.abs(density_fft)**2
    
    return power_spectrum

def radial_average(power_spectrum):
    """
    Compute the radial average of the power spectrum.
    """
    pix_size = 1.16/0.7
    
    # Calculate the magnitude of wave vectors
    nx, ny = power_spectrum.shape
    kx = np.fft.fftfreq(nx) * 2 * np.pi
    ky = np.fft.fftfreq(ny) * 2 * np.pi
    k = np.sqrt(kx[:, np.newaxis]**2 + ky**2)
    
    # Compute the radial average
    k_bins = np.linspace(0, np.max(k), min(nx, ny) // 2)
    #print(np.max(k))
    k_centers = 0.5 * (k_bins[1:] + k_bins[:-1])
    radial_profile, _ = np.histogram(k, bins=k_bins, weights=power_spectrum)
    counts, _ = np.histogram(k, bins=k_bins)
    radial_average = radial_profile / counts
    
    k_centers /= pix_size
    radial_average /= pix_size
    
    return k_centers, radial_average

def plot_power_spectrum(k, radial_average, label=None, linewidth=1, color = None, alpha = 0.5):
    plt.plot(k, radial_average, label=label, linewidth=linewidth, color = color, alpha = alpha)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('k [$kpc^{-1}$]', fontsize=16)
    plt.ylabel('$P_{sub}(k)$', fontsize=16)
    

def plot_all_power_spectra(all_sub, host, halo_num, R_200, M_200):

    # Initialize arrays to store power spectra
    all_power_spectra = []
    all_k = []
    pix_size = 1.16/0.7
    
    fig, ax = plt.subplots(figsize=(10, 6))
    # Compute and plot power spectra for each substructure density field
    for i, substructure in enumerate(all_sub):
        # Compute the power spectrum
        
        power_spectrum = compute_power_spectrum(substructure)

        # Compute the radial average
        k, radial_avg = (radial_average(power_spectrum))

        # Plot the power spectrum
        plot_power_spectrum(k, radial_avg, linewidth=0.5, alpha = 0.7)

        # Store power spectrum and k values
        all_power_spectra.append(radial_avg)
        all_k.append(k)

    # Average Substructure Power Spectrum
    avg_power_spectrum = np.mean(all_power_spectra, axis=0)
    avg_k = np.mean(all_k, axis=0)
    plot_power_spectrum(avg_k, avg_power_spectrum, label='Substructure', linewidth=2, color = 'darkmagenta', alpha = 1.0)
    sub_power_spectrum = np.array([avg_k, avg_power_spectrum])
    np.save("sub_avg_power_spectrum_"+halo_num+".npy", sub_power_spectrum)

    # Host halo power spectrum
    power_spectrum = compute_power_spectrum(host)
    k, radial_avg = (radial_average(power_spectrum))
    plot_power_spectrum(k, radial_avg, linewidth= 2.0, label = "Host", color = "darkgreen", alpha = 1.0)
    host_power_spectrum = np.array([k, radial_avg])
    np.save("host_power_spectrum_"+halo_num+".npy", host_power_spectrum)

    plt.legend(fontsize=12)
    plt.subplots_adjust(top=0.9)
    #plt.rcParams["figure.figsize"] = (10, 6)  
    plt.tick_params(axis='both', which='major', labelsize=12)  
    plt.tick_params(axis='both', which='minor', labelsize=10) 
    plt.tight_layout()  
    plt.title(('Power Spectrum of Halo '+halo_num+", "+f" M_200 = {M_200:.2e} $\mathrm{{M}}_\odot$, R_200 = "+str(int(R_200))+" kpc"), fontsize=16)
    plt.xlim(0.005, 3)
    plt.ylim(0.00001, 100)
    plt.savefig("Power_spectrum_halo"+str(halo_num)+".png", bbox_inches='tight')
    plt.show()