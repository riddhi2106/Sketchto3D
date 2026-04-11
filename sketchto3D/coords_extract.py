import numpy as np

def extract_coordinates(edges):
    coords = np.column_stack(np.where(edges > 0))
    
    # Remove duplicate points (safety)
    coords = np.unique(coords, axis=0)
    
    # Reduce density (cleaner + faster)
    coords = coords[::2]
    
    return coords