# Definitely faster than V1 but I can still see jagged edges on the smooth one and also the jump from 0 to 1 is too significant
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from numba import njit, prange

# --- Data-Oriented Scene Setup ---
# Flattening objects into parallel arrays for memory efficiency and C-level compilation

# Spheres: Red, Green, Blue
sphere_centers = np.array([
    [0.0, 1.0, -5.0], 
    [-1.5, 0.5, -4.0], 
    [1.5, 0.5, -4.0]
], dtype=np.float64)
sphere_radii = np.array([1.0, 0.5, 0.5], dtype=np.float64)
sphere_colors = np.array([
    [0.9, 0.1, 0.1], 
    [0.1, 0.9, 0.1], 
    [0.1, 0.1, 0.9]
], dtype=np.float64)
sphere_reflectiveness = np.array([0.2, 0.1, 0.1], dtype=np.float64)
sphere_shininess = np.array([50.0, 50.0, 50.0], dtype=np.float64)

# Planes: Floor
plane_points = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
plane_normals = np.array([[0.0, 1.0, 0.0]], dtype=np.float64)
plane_colors = np.array([[0.6, 0.6, 0.6]], dtype=np.float64)
plane_reflectiveness = np.array([0.3], dtype=np.float64)
plane_shininess = np.array([10.0], dtype=np.float64)

# Light
light_pos = np.array([5.0, 5.0, 0.0], dtype=np.float64)
light_intensity = 1.0
ambient_light = 0.15

# Camera & Screen (Increased resolution because it's fast now)
width, height = 320, 240
aspect_ratio = width / height
fov = np.pi / 3.0
camera_origin = np.array([0.0, 1.0, 1.0], dtype=np.float64)

# --- High-Speed Compiled Math ---
@njit(fastmath=True)
def normalize(v):
    norm = np.sqrt(v[0]**2 + v[1]**2 + v[2]**2)
    if norm == 0: return v
    return v / norm

@njit(fastmath=True)
def intersect_sphere(origin, direction, center, radius):
    oc = origin - center
    b = 2.0 * (oc[0]*direction[0] + oc[1]*direction[1] + oc[2]*direction[2])
    c = (oc[0]**2 + oc[1]**2 + oc[2]**2) - radius * radius
    discriminant = b * b - 4 * c
    
    if discriminant > 0:
        sqrt_disc = np.sqrt(discriminant)
        t1 = (-b - sqrt_disc) / 2.0
        if t1 > 1e-4: return t1
        t2 = (-b + sqrt_disc) / 2.0
        if t2 > 1e-4: return t2
    return np.inf

@njit(fastmath=True)
def intersect_plane(origin, direction, point, normal):
    denom = direction[0]*normal[0] + direction[1]*normal[1] + direction[2]*normal[2]
    if abs(denom) > 1e-6:
        v = point - origin
        t = (v[0]*normal[0] + v[1]*normal[1] + v[2]*normal[2]) / denom
        if t > 1e-4: return t
    return np.inf

# --- Core Rendering Logic ---
@njit(fastmath=True)
def trace_ray(origin, direction, depth, num_bands):
    if depth > 1: # Recursion limit
        return np.array([0.0, 0.0, 0.0], dtype=np.float64)

    closest_t = np.inf
    obj_type = -1 # 0 for sphere, 1 for plane
    obj_idx = -1

    # Check spheres
    for i in range(len(sphere_centers)):
        t = intersect_sphere(origin, direction, sphere_centers[i], sphere_radii[i])
        if t < closest_t:
            closest_t = t
            obj_type = 0
            obj_idx = i

    # Check planes
    for i in range(len(plane_points)):
        t = intersect_plane(origin, direction, plane_points[i], plane_normals[i])
        if t < closest_t:
            closest_t = t
            obj_type = 1
            obj_idx = i

    if obj_idx == -1:
        return np.array([0.05, 0.05, 0.05], dtype=np.float64) # Background

    # Get object properties
    hit_point = origin + closest_t * direction
    if obj_type == 0:
        normal = normalize(hit_point - sphere_centers[obj_idx])
        color = sphere_colors[obj_idx]
        shininess = sphere_shininess[obj_idx]
        reflectiveness = sphere_reflectiveness[obj_idx]
        is_sphere = True
    else:
        normal = plane_normals[obj_idx]
        color = plane_colors[obj_idx]
        shininess = plane_shininess[obj_idx]
        reflectiveness = plane_reflectiveness[obj_idx]
        is_sphere = False

    view_dir = normalize(-direction)

    # TRUE CEL-SHADING: Edge Detection / Outlines
    if num_bands > 0 and is_sphere:
        edge_dot = view_dir[0]*normal[0] + view_dir[1]*normal[1] + view_dir[2]*normal[2]
        if edge_dot < 0.25: 
            return np.array([0.0, 0.0, 0.0], dtype=np.float64)

    # Shadows
    light_dir = normalize(light_pos - hit_point)
    shadow_origin = hit_point + normal * 1e-4
    in_shadow = False
    
    for i in range(len(sphere_centers)):
        if intersect_sphere(shadow_origin, light_dir, sphere_centers[i], sphere_radii[i]) < np.inf:
            in_shadow = True; break
    if not in_shadow:
        for i in range(len(plane_points)):
            if intersect_plane(shadow_origin, light_dir, plane_points[i], plane_normals[i]) < np.inf:
                in_shadow = True; break

    # Phong Shading
    diffuse_dot = normal[0]*light_dir[0] + normal[1]*light_dir[1] + normal[2]*light_dir[2]
    diffuse_intensity = max(0.0, diffuse_dot)
    
    half_vector = normalize(light_dir + view_dir)
    specular_dot = normal[0]*half_vector[0] + normal[1]*half_vector[1] + normal[2]*half_vector[2]
    specular_intensity = max(0.0, specular_dot) ** shininess
    
    if in_shadow:
        diffuse_intensity *= 0.1
        specular_intensity = 0.0

    # CEL-SHADING: Quantization
    if num_bands > 0:
        step_size = 1.0 / num_bands
        diffuse_intensity = np.ceil(diffuse_intensity / step_size) * step_size
        specular_intensity = 1.0 if specular_intensity > 0.5 else 0.0

    final_color = color * ambient_light
    final_color += color * diffuse_intensity * light_intensity
    final_color += np.array([1.0, 1.0, 1.0]) * specular_intensity * light_intensity

    # Reflections (Recursion)
    if reflectiveness > 0:
        dot_nd = direction[0]*normal[0] + direction[1]*normal[1] + direction[2]*normal[2]
        reflect_dir = normalize(direction - 2.0 * dot_nd * normal)
        reflect_origin = hit_point + normal * 1e-4
        reflect_color = trace_ray(reflect_origin, reflect_dir, depth + 1, num_bands)
        final_color = final_color * (1.0 - reflectiveness) + reflect_color * reflectiveness

    # Clip to 0-1 range manually
    final_color[0] = min(max(final_color[0], 0.0), 1.0)
    final_color[1] = min(max(final_color[1], 0.0), 1.0)
    final_color[2] = min(max(final_color[2], 0.0), 1.0)
    return final_color

# --- Parallel Processing Frame Generation ---
@njit(parallel=True, fastmath=True)
def render_numba(num_bands):
    image = np.zeros((height, width, 3), dtype=np.float64)
    
    screen_x = np.linspace(-1, 1, width) * np.tan(fov / 2.0) * aspect_ratio
    screen_y = np.linspace(1, -1, height) * np.tan(fov / 2.0)

    # prange allows Numba to automatically parallelize this outer loop across all CPU cores
    for i in prange(height):
        for j in range(width):
            pixel_coord = np.array([screen_x[j], screen_y[i], -1.0], dtype=np.float64)
            ray_direction = normalize(pixel_coord - camera_origin)
            image[i, j] = trace_ray(camera_origin, ray_direction, 0, num_bands)
            
    return image

# --- Matplotlib UI Setup ---
fig, ax = plt.subplots(figsize=(8, 6))
plt.subplots_adjust(bottom=0.25)

print("Compiling engine... (This takes a few seconds on the first run)")
initial_bands = 0
current_image = render_numba(initial_bands) # First call triggers the compiler
print("Compilation complete! Engine running.")

img_plot = ax.imshow(current_image)
ax.axis('off')
ax.set_title("Hardware-Accelerated JIT Ray Tracer")

ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
band_slider = Slider(ax_slider, 'Stylization Bands\n(0 = Smooth)', 0, 8, valinit=initial_bands, valstep=1)

def update(val):
    bands = int(band_slider.val)
    new_image = render_numba(bands)
    img_plot.set_data(new_image)
    fig.canvas.draw_idle()

band_slider.on_changed(update)
plt.show()