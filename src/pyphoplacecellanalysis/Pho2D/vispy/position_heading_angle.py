"""
VisPy Line Shader with Heading-to-Hue Mapping (North = Red)

Heading angle mapping:
- 0° (North/Up) = Red
- 90° (East/Right) = Cyan
- 180° (South/Down) = Green  
- 270° (West/Left) = Magenta
- 360° wraps back to Red
"""

import numpy as np
from vispy import app, gloo
from vispy.util.transforms import ortho

# Vertex shader
VERT_SHADER = """
#version 120

uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_projection;

attribute vec2 a_position;
attribute vec2 a_tangent;  // Direction vector of the line at this vertex

varying float v_heading;

const float PI = 3.14159265359;

void main() {
    gl_Position = u_projection * u_view * u_model * vec4(a_position, 0.0, 1.0);
    
    // Calculate heading angle from tangent vector
    // atan returns angle in radians from -PI to PI
    float angle = atan(a_tangent.y, a_tangent.x);
    
    // Convert to compass heading where 0° is North (up)
    // atan2 returns 0 for East (+x), PI/2 for North (+y)
    // We want 0 for North, so rotate by -90° (subtract PI/2)
    v_heading = angle - PI / 2.0;
}
"""

# Fragment shader
FRAG_SHADER = """
#version 120

varying float v_heading;

const float PI = 3.14159265359;

// HSV to RGB conversion
vec3 hsv2rgb(vec3 c) {
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

void main() {
    // Normalize heading from [-PI, PI] to [0, 1] for hue
    // Add PI to shift range, then divide by 2*PI
    float hue = (v_heading + PI) / (2.0 * PI);
    
    // Ensure hue wraps correctly [0, 1]
    hue = fract(hue);
    
    // Convert HSV to RGB (saturation=1.0, value=1.0 for vivid colors)
    vec3 color = hsv2rgb(vec3(hue, 1.0, 1.0));
    
    gl_FragColor = vec4(color, 1.0);
}
"""


class HeadingColoredLine(app.Canvas):
    def __init__(self, **kwargs):
        app.Canvas.__init__(self, keys='interactive', size=(800, 600), **kwargs)
        
        # Create a sample path (spiral)
        t = np.linspace(0, 4 * np.pi, 1000)
        x = t * np.cos(t) * 0.1
        y = t * np.sin(t) * 0.1
        positions = np.c_[x, y].astype(np.float32)
        
        # Calculate tangent vectors (heading direction) at each vertex
        tangents = np.zeros_like(positions)
        
        # Forward difference for tangents
        tangents[:-1] = positions[1:] - positions[:-1]
        tangents[-1] = tangents[-2]  # Repeat last tangent
        
        # Normalize tangent vectors
        tangent_lengths = np.sqrt((tangents ** 2).sum(axis=1))
        tangent_lengths[tangent_lengths == 0] = 1  # Avoid division by zero
        tangents /= tangent_lengths[:, np.newaxis]
        
        # Create the program
        self.program = gloo.Program(VERT_SHADER, FRAG_SHADER)
        self.program['a_position'] = positions
        self.program['a_tangent'] = tangents.astype(np.float32)
        
        # Set up transformations
        self.program['u_model'] = np.eye(4, dtype=np.float32)
        self.program['u_view'] = np.eye(4, dtype=np.float32)
        self.update_projection()
        
        # OpenGL settings
        gloo.set_state(clear_color='black', blend=True,
                      blend_func=('src_alpha', 'one_minus_src_alpha'),
                      line_width=2.0)
        
        self.show()
    
    def update_projection(self):
        w, h = self.physical_size
        self.program['u_projection'] = ortho(-1, 1, -1, 1, -1, 1)
    
    def on_draw(self, event):
        gloo.clear()
        self.program.draw('line_strip')
    
    def on_resize(self, event):
        gloo.set_viewport(0, 0, *event.physical_size)
        self.update_projection()


class CompassDemo(app.Canvas):
    """
    Demonstrates the heading-to-color mapping with lines pointing in cardinal directions
    """
    def __init__(self, **kwargs):
        app.Canvas.__init__(self, keys='interactive', size=(800, 800), **kwargs)
        
        # Create lines pointing in different directions from center
        center = np.array([0.0, 0.0])
        length = 0.6
        
        # Create 8 cardinal/intercardinal directions
        angles = np.linspace(0, 2 * np.pi, 9)[:-1]  # 0, 45, 90, ..., 315 degrees
        
        all_positions = []
        all_tangents = []
        
        for angle in angles:
            # Each line goes from center outward
            end = center + length * np.array([np.cos(angle), np.sin(angle)])
            
            # Create line with multiple points for smooth rendering
            line_points = 20
            t = np.linspace(0, 1, line_points)
            x = center[0] + t * (end[0] - center[0])
            y = center[1] + t * (end[1] - center[1])
            positions = np.c_[x, y]
            
            # Tangent is constant for straight line
            tangent = (end - center) / np.linalg.norm(end - center)
            tangents = np.tile(tangent, (line_points, 1))
            
            all_positions.append(positions)
            all_tangents.append(tangents)
            
            # Add a gap between lines (NaN values)
            all_positions.append(np.array([[np.nan, np.nan]]))
            all_tangents.append(np.array([[0.0, 0.0]]))
        
        positions = np.vstack(all_positions).astype(np.float32)
        tangents = np.vstack(all_tangents).astype(np.float32)
        
        # Create the program
        self.program = gloo.Program(VERT_SHADER, FRAG_SHADER)
        self.program['a_position'] = positions
        self.program['a_tangent'] = tangents
        
        # Set up transformations
        self.program['u_model'] = np.eye(4, dtype=np.float32)
        self.program['u_view'] = np.eye(4, dtype=np.float32)
        self.update_projection()
        
        # OpenGL settings
        gloo.set_state(clear_color='black', blend=True,
                      blend_func=('src_alpha', 'one_minus_src_alpha'),
                      line_width=5.0)
        
        self.show()
        print("\nCompass Rose Color Mapping:")
        print("North (↑, 0°): Red")
        print("Northeast (↗, 45°): Orange/Yellow")
        print("East (→, 90°): Cyan")
        print("Southeast (↘, 135°): Blue")
        print("South (↓, 180°): Green")
        print("Southwest (↙, 225°): Yellow-Green")
        print("West (←, 270°): Magenta")
        print("Northwest (↖, 315°): Red-Magenta")
    
    def update_projection(self):
        w, h = self.physical_size
        self.program['u_projection'] = ortho(-1, 1, -1, 1, -1, 1)
    
    def on_draw(self, event):
        gloo.clear()
        self.program.draw('line_strip')
    
    def on_resize(self, event):
        gloo.set_viewport(0, 0, *event.physical_size)
        self.update_projection()


class InteractiveHeadingLine(app.Canvas):
    """
    Interactive version where you can draw lines with the mouse
    """
    def __init__(self, **kwargs):
        app.Canvas.__init__(self, keys='interactive', size=(800, 600), **kwargs)
        
        self.positions = []
        self.drawing = False
        
        # Create initial empty buffers
        self.program = gloo.Program(VERT_SHADER, FRAG_SHADER)
        self.program['u_model'] = np.eye(4, dtype=np.float32)
        self.program['u_view'] = np.eye(4, dtype=np.float32)
        self.update_projection()
        
        gloo.set_state(clear_color='black', blend=True,
                      blend_func=('src_alpha', 'one_minus_src_alpha'),
                      line_width=3.0)
        
        self.show()
        print("Click and drag to draw lines. Press 'C' to clear.")
        print("Draw upward for red, rightward for cyan, downward for green, leftward for magenta")
    
    def update_projection(self):
        w, h = self.physical_size
        self.program['u_projection'] = ortho(-1, 1, -1, 1, -1, 1)
    
    def on_mouse_press(self, event):
        self.drawing = True
        # Convert to normalized coordinates
        x = 2 * event.pos[0] / self.size[0] - 1
        y = 1 - 2 * event.pos[1] / self.size[1]
        self.positions = [[x, y]]
    
    def on_mouse_move(self, event):
        if self.drawing:
            x = 2 * event.pos[0] / self.size[0] - 1
            y = 1 - 2 * event.pos[1] / self.size[1]
            self.positions.append([x, y])
            self.update_line()
            self.update()
    
    def on_mouse_release(self, event):
        self.drawing = False
    
    def on_key_press(self, event):
        if event.key == 'C':
            self.positions = []
            self.update()
    
    def update_line(self):
        if len(self.positions) < 2:
            return
        
        positions = np.array(self.positions, dtype=np.float32)
        
        # Calculate tangents
        tangents = np.zeros_like(positions)
        tangents[:-1] = positions[1:] - positions[:-1]
        tangents[-1] = tangents[-2]
        
        # Normalize
        tangent_lengths = np.sqrt((tangents ** 2).sum(axis=1))
        tangent_lengths[tangent_lengths == 0] = 1
        tangents /= tangent_lengths[:, np.newaxis]
        
        self.program['a_position'] = positions
        self.program['a_tangent'] = tangents
    
    def on_draw(self, event):
        gloo.clear()
        if len(self.positions) >= 2:
            self.program.draw('line_strip')
    
    def on_resize(self, event):
        gloo.set_viewport(0, 0, *event.physical_size)
        self.update_projection()


if __name__ == '__main__':
    # Choose which example to run
    print("1. Spiral example")
    print("2. Compass rose (shows color mapping)")
    print("3. Interactive drawing")
    choice = input("Choose (1, 2, or 3): ").strip()
    
    if choice == '2':
        canvas = CompassDemo()
    elif choice == '3':
        canvas = InteractiveHeadingLine()
    else:
        canvas = HeadingColoredLine()
    
    app.run()

