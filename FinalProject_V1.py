# Kinda works. Very slow at rendering and the smooth end still has jagged edges
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import concurrent.futures

# --- Highly Optimized Math Helpers ---
def normalize(v):
    norm = np.sqrt(np.dot(v, v))
    return v / norm if norm > 0 else v

# --- Scene Objects ---
class Sphere:
    def __init__(self, center, radius, color, specular, shininess, reflectiveness):
        self.center = np.array(center, dtype=np.float64)
        self.radius = radius
        self.color = np.array(color, dtype=np.float64)
        self.specular = np.array(specular, dtype=np.float64)
        self.shininess = shininess
        self.reflectiveness = reflectiveness
        self.is_sphere = True # Used for outline logic

    def intersect(self, ray_origin, ray_direction):
        oc = ray_origin - self.center
        b = 2.0 * np.dot(oc, ray_direction)
        c = np.dot(oc, oc) - self.radius * self.radius
        discriminant = b * b - 4 * c # 'a' is 1 since direction is normalized

        if discriminant > 0:
            sqrt_disc = np.sqrt(discriminant)
            t1 = (-b - sqrt_disc) / 2.0
            if t1 > 1e-4: return t1
            t2 = (-b + sqrt_disc) / 2.0
            if t2 > 1e-4: return t2
        return np.inf

    def get_normal(self, point):
        return normalize(point - self.center)

class Plane:
    def __init__(self, point, normal, color, specular, shininess, reflectiveness):
        self.point = np.array(point, dtype=np.float64)
        self.normal = normalize(np.array(normal, dtype=np.float64))
        self.color = np.array(color, dtype=np.float64)
        self.specular = np.array(specular, dtype=np.float64)
        self.shininess = shininess
        self.reflectiveness = reflectiveness
        self.is_sphere = False

    def intersect(self, ray_origin, ray_direction):
        denom = np.dot(ray_direction, self.normal)
        if abs(denom) > 1e-6:
            t = np.dot(self.point - ray_origin, self.normal) / denom
            if t > 1e-4: return t
        return np.inf

    def get_normal(self, point):
        return self.normal

# --- Light Setup ---
class Light:
    def __init__(self, position, intensity):
        self.position = np.array(position, dtype=np.float64)
        self.intensity = intensity

# --- Engine Configuration ---
spheres = [
    Sphere([0, 1, -5], 1.0, [0.9, 0.1, 0.1], [1, 1, 1], 50, 0.2),     
    Sphere([-1.5, 0.5, -4], 0.5, [0.1, 0.9, 0.1], [1, 1, 1], 50, 0.1), 
    Sphere([1.5, 0.5, -4], 0.5, [0.1, 0.1, 0.9], [1, 1, 1], 50, 0.1)   
]
planes = [
    Plane([0, 0, 0], [0, 1, 0], [0.6, 0.6, 0.6], [1, 1, 1], 10, 0.3)   
]
objects = spheres + planes
light = Light([5, 5, 0], 1.0)
ambient_light = 0.15

width, height = 200, 150 # Increased resolution now that it's faster
aspect_ratio = width / height
fov = np.pi / 3
max_depth = 1 # Keep at 1 or 2 for speedy interactive previews

def trace_ray(origin, direction, depth, num_bands):
    if depth > max_depth:
        return np.array([0.0, 0.0, 0.0])

    closest_t = np.inf
    closest_obj = None

    for obj in objects:
        t = obj.intersect(origin, direction)
        if t < closest_t:
            closest_t = t
            closest_obj = obj

    if closest_obj is None:
        return np.array([0.05, 0.05, 0.05]) # Background

    hit_point = origin + closest_t * direction
    normal = closest_obj.get_normal(hit_point)
    view_dir = normalize(-direction)

    # --- TRUE CEL-SHADING: Edge Detection / Outlines ---
    if num_bands > 0 and closest_obj.is_sphere:
        edge_dot = np.dot(view_dir, normal)
        # If the viewing angle is near perpendicular to the normal, draw a black line
        if edge_dot < 0.25: 
            return np.array([0.0, 0.0, 0.0])

    # Shadow calculation
    light_dir = normalize(light.position - hit_point)
    shadow_origin = hit_point + normal * 1e-4
    in_shadow = False
    
    for obj in objects:
        if obj.intersect(shadow_origin, light_dir) < np.inf:
            in_shadow = True
            break

    # Phong shading logic
    diffuse_intensity = max(0, np.dot(normal, light_dir))
    half_vector = normalize(light_dir + view_dir)
    specular_intensity = max(0, np.dot(normal, half_vector)) ** closest_obj.shininess
    
    if in_shadow:
        diffuse_intensity *= 0.1
        specular_intensity = 0.0

    # --- CEL-SHADING: Clean Light Quantization ---
    if num_bands > 0:
        # Snap light to clean, distinct bands
        step_size = 1.0 / num_bands
        diffuse_intensity = np.ceil(diffuse_intensity / step_size) * step_size
        
        # Hard cutoff for specular highlights (comic book style gloss)
        specular_intensity = 1.0 if specular_intensity > 0.5 else 0.0

    color = closest_obj.color * ambient_light
    color += closest_obj.color * diffuse_intensity * light.intensity
    color += closest_obj.specular * specular_intensity * light.intensity

    # Reflections
    if closest_obj.reflectiveness > 0:
        reflect_dir = direction - 2 * np.dot(direction, normal) * normal
        reflect_origin = hit_point + normal * 1e-4
        reflect_color = trace_ray(reflect_origin, reflect_dir, depth + 1, num_bands)
        color = color * (1 - closest_obj.reflectiveness) + reflect_color * closest_obj.reflectiveness

    return np.clip(color, 0, 1)

def render_row(y_idx, screen_x, screen_y, camera_origin, num_bands):
    """Worker function to render a single row of pixels."""
    row_colors = np.zeros((width, 3))
    for j in range(width):
        pixel_coord = np.array([screen_x[j], screen_y[y_idx], -1.0])
        ray_direction = normalize(pixel_coord - camera_origin)
        row_colors[j] = trace_ray(camera_origin, ray_direction, 0, num_bands)
    return y_idx, row_colors

def render(num_bands):
    image = np.zeros((height, width, 3))
    
    screen_x = np.linspace(-1, 1, width) * np.tan(fov / 2) * aspect_ratio
    screen_y = np.linspace(1, -1, height) * np.tan(fov / 2)
    camera_origin = np.array([0.0, 1.0, 1.0])

    # --- MULTITHREADING: Render rows in parallel for massive speedup ---
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(render_row, i, screen_x, screen_y, camera_origin, num_bands) 
            for i in range(height)
        ]
        for future in concurrent.futures.as_completed(futures):
            y_idx, row_colors = future.result()
            image[y_idx] = row_colors
            
    return image

# --- Matplotlib UI Setup ---
fig, ax = plt.subplots(figsize=(8, 6))
plt.subplots_adjust(bottom=0.25)

initial_bands = 0
current_image = render(initial_bands)
img_plot = ax.imshow(current_image)
ax.axis('off')
ax.set_title("Optimized Ray Tracer with Outline Cel-Shader")

ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
# Values 2-4 give the best comic book look
band_slider = Slider(ax_slider, 'Stylization Bands\n(0 = Smooth)', 0, 8, valinit=initial_bands, valstep=1)

def update(val):
    bands = int(band_slider.val)
    print(f"Rendering frame with bands = {bands}...")
    new_image = render(bands)
    img_plot.set_data(new_image)
    fig.canvas.draw_idle()

band_slider.on_changed(update)
plt.show()