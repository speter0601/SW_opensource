import numpy as np

MATERIALS = ['wall', 'metal', 'empty_space']

class WiFiSimulator:
    def __init__(self, space_size=(10, 10), num_bins=50):
        self.space_size = space_size
        self.num_bins = num_bins
        
    def generate_scatter_points(self, material, center, size):
        points = []
        num_points = 20
        if material == 'wall':
            # Straight line for wall
            for x in np.linspace(-size*2, size*2, num_points):
                points.append((center[0] + x, center[1]))
        elif material == 'metal':
            # Dense object with high reflection
            angles = np.linspace(0, 2*np.pi, num_points, endpoint=False)
            for a in angles:
                points.append((center[0] + size * np.cos(a), center[1] + size * np.sin(a)))
        elif material == 'empty_space':
            # No points
            pass
            
        return np.array(points) if points else np.empty((0, 2))

    def simulate(self, material=None):
        if material is None:
            material = np.random.choice(MATERIALS)
        
        # Place transmitters at random boundary locations or corners
        tx1 = np.random.uniform(0, self.space_size[0]/4, 2)
        tx2 = np.random.uniform(self.space_size[0]*3/4, self.space_size[0], 2)
        
        # Place object in the middle area to ensure visibility
        obj_center = np.random.uniform(3, self.space_size[0]-3, 2)
        obj_size = np.random.uniform(0.5, 1.5)
        
        scatter_points = self.generate_scatter_points(material, obj_center, obj_size)
        
        signal1 = np.zeros(self.num_bins)
        signal2 = np.zeros(self.num_bins)
        
        max_dist = np.sqrt(self.space_size[0]**2 + self.space_size[1]**2) * 2
        
        if len(scatter_points) > 0:
            for pt in scatter_points:
                dist1 = 2 * np.linalg.norm(pt - tx1)
                dist2 = 2 * np.linalg.norm(pt - tx2)
                
                att1 = 1.0 / (dist1**2 + 1e-2)
                att2 = 1.0 / (dist2**2 + 1e-2)
                
                if material == 'metal':
                    att1 *= 3.0
                    att2 *= 3.0
                
                bin1 = int((dist1 / max_dist) * self.num_bins)
                bin2 = int((dist2 / max_dist) * self.num_bins)
                
                if 0 <= bin1 < self.num_bins: signal1[bin1] += att1
                if 0 <= bin2 < self.num_bins: signal2[bin2] += att2
            
        # Environmental noise
        signal1 += np.random.normal(0, 0.05, self.num_bins)
        signal2 += np.random.normal(0, 0.05, self.num_bins)
        
        features = np.concatenate([signal1, signal2])
        return {
            'features': features.tolist(),
            'label_name': material,
            'label_idx': MATERIALS.index(material),
            'tx1': tx1.tolist(),
            'tx2': tx2.tolist(),
            'obj_center': obj_center.tolist()
        }
