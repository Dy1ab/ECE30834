import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from numba import njit, prange

# --- Data-Oriented Scene Setup ---
sphere_centers = np.array([
    [0.0, 0.0, -1.9],  # Center red
    [-1.3, -0.55, -1.4], # Left green
    [1.7, -0.55, -1.4]   # Right blue
], dtype=np.float64)
sphere_radii = np.array([1.0, 0.45, 0.45], dtype=np.float64)

# Pure Colors for Comic Book Vibrancy
sphere_colors = np.array([
    [1.0, 0.0, 0.0], # Pure Red
    [0.0, 1.0, 0.0], # Pure Green
    [0.0, 0.0, 1.0]  # Pure Blue
], dtype=np.float64)

# Zero reflections to prevent the background from washing out the pure colors
sphere_reflectiveness = np.array([0.0, 0.0, 0.0], dtype=np.float64)
sphere_shininess = np.array([50.0, 50.0, 50.0], dtype=np.float64)

# Planes: Pure White Floor
plane_points = np.array([[0.0, -1.0, 0.0]], dtype=np.float64) 
plane_normals = np.array([[0.0, 1.0, 0.0]], dtype=np.float64)
plane_colors = np.array([[1.0, 1.0, 1.0]], dtype=np.float64) 
plane_reflectiveness = np.array([0.0], dtype=np.float64)
plane_shininess = np.array([10.0], dtype=np.float64)

# Overdriven Lighting for high contrast
light_pos = np.array([5.0, 5.0, 2.0], dtype=np.float64) 
light_intensity = 1.2  
ambient_light = 0.05   

# --- HIGH RESOLUTION SETTINGS ---
width, height = 640, 480 # Doubled the resolution for sharper edges
aspect_ratio = width / height
fov = np.pi / 3.0
camera_origin = np.array([0.0, 0.0, 2.0], dtype=np.float64)

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
def trace_ray(origin, direction, depth, style_mix):
    if depth > 1: 
        return np.array([0.0, 0.0, 0.0], dtype=np.float64)

    closest_t = np.inf
    obj_type = -1 
    obj_idx = -1

    for i in range(len(sphere_centers)):
        t = intersect_sphere(origin, direction, sphere_centers[i], sphere_radii[i])
        if t < closest_t:
            closest_t = t
            obj_type = 0
            obj_idx = i

    for i in range(len(plane_points)):
        t = intersect_plane(origin, direction, plane_points[i], plane_normals[i])
        if t < closest_t:
            closest_t = t
            obj_type = 1
            obj_idx = i

    if obj_idx == -1:
        # Pure white sky background for maximum contrast
        return np.array([1.0, 1.0, 1.0], dtype=np.float64) 

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

    # Outline Detection
    outline_factor = 1.0
    if is_sphere:
        edge_dot = view_dir[0]*normal[0] + view_dir[1]*normal[1] + view_dir[2]*normal[2]
        if edge_dot < 0.25: 
            outline_factor = 1.0 - style_mix 

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

    # Base Smooth Phong Lighting
    base_diffuse = max(0.0, normal[0]*light_dir[0] + normal[1]*light_dir[1] + normal[2]*light_dir[2])
    half_vector = normalize(light_dir + view_dir)
    base_specular = max(0.0, normal[0]*half_vector[0] + normal[1]*half_vector[1] + normal[2]*half_vector[2]) ** shininess
    
    if in_shadow:
        base_diffuse *= 0.1
        base_specular = 0.0

    # Base Cel Lighting 
    BANDS = 3.0 
    cel_diffuse = np.ceil(base_diffuse * BANDS) / BANDS
    cel_specular = 1.0 if base_specular > 0.5 else 0.0

    # Blend the smooth and comic math based on slider
    diffuse_intensity = base_diffuse * (1.0 - style_mix) + cel_diffuse * style_mix
    specular_intensity = base_specular * (1.0 - style_mix) + cel_specular * style_mix

    final_color = color * ambient_light
    final_color += color * diffuse_intensity * light_intensity
    final_color += np.array([1.0, 1.0, 1.0]) * specular_intensity * light_intensity

    # --- APPLY OUTLINE AND GAMMA CORRECTION ---
    final_color *= outline_factor
    final_color[0] = min(max(final_color[0], 0.0), 1.0) ** (1.0 / 2.2)
    final_color[1] = min(max(final_color[1], 0.0), 1.0) ** (1.0 / 2.2)
    final_color[2] = min(max(final_color[2], 0.0), 1.0) ** (1.0 / 2.2)
    return final_color

# --- Parallel Processing Frame Generation ---
@njit(parallel=True, fastmath=True)
def render_numba(style_mix):
    image = np.zeros((height, width, 3), dtype=np.float64)
    
    # MSAA: 4 Sub-pixel samples per actual pixel
    offsets = np.array([-0.25, 0.25], dtype=np.float64)
    
    for i in prange(height):
        for j in range(width):
            pixel_color = np.array([0.0, 0.0, 0.0], dtype=np.float64)
            
            # Shoot 4 rays slightly offset from each other
            for ox in offsets:
                for oy in offsets:
                    sx = (2.0 * (j + 0.5 + ox) / width - 1.0) * np.tan(fov / 2.0) * aspect_ratio
                    sy = (1.0 - 2.0 * (i + 0.5 + oy) / height) * np.tan(fov / 2.0)
                    
                    pixel_coord = np.array([camera_origin[0] + sx, camera_origin[1] + sy, camera_origin[2] - 1.0], dtype=np.float64)
                    ray_direction = normalize(pixel_coord - camera_origin)
                    pixel_color += trace_ray(camera_origin, ray_direction, 0, style_mix)
            
            image[i, j] = pixel_color / 4.0
            
    return image

# --- Matplotlib UI Setup ---
# Bumped up figsize and DPI for a much crisper viewing window
fig, ax = plt.subplots(figsize=(10, 8), dpi=100)

# whole window background
fig.patch.set_facecolor("#A8CFFF")   # window background color
#fig.patch.set_edgecolor("#2a2a2a")   # border color around whole figure
fig.patch.set_linewidth(2)           # border thickness

plt.subplots_adjust(bottom=0.25)

print(f"Compiling High-Res MSAA engine ({width}x{height})... (Takes a few seconds)")
initial_mix = 0.0
current_image = render_numba(initial_mix) 
print("Compilation complete! Engine running.")

# Using interpolation='bilinear' helps smooth out the display inside the matplotlib window
img_plot = ax.imshow(current_image, interpolation='bilinear')
ax.axis('off')
# ax.set_frame_on(True)
ax.set_title("Vibrant Cel-Shaded Ray Tracer (High-Res)", fontsize=20)

ax_slider = plt.axes([0.25, 0.1, 0.6, 0.06])
style_slider = Slider(
    ax_slider,
    'Stylization\n(0=Real, 1=Comic)',
    0.0,
    1.0,
    valinit=initial_mix,
    valstep=0.05,
    handle_style={'size': 20}
)

def update(val):
    mix = style_slider.val
    new_image = render_numba(mix)
    img_plot.set_data(new_image)
    fig.canvas.draw_idle()

style_slider.on_changed(update)
plt.show()