import numpy as np

class Grid:
    def __init__(self, 
                 grange=200, 
                 gstep=1):
        self.grange = grange
        self.gstep = gstep
        self.grid = self.create_grid()
        
    def create_grid(self):
        size = int(self.grange/self.gstep)
        return [[0] * size for _ in range(size)]

def bresenham_line(x0, y0, x1, y1, 
                   grid,
                   grange,
                   gstep,
                   weigth=1):
    grid_size = len(grid)
    
    x0 = int(np.rint(x0 * 1/gstep))
    y0 = int(np.rint(y0 * 1/gstep))
    x1 = int(np.rint(x1 * 1/gstep))
    y1 = int(np.rint(y1 * 1/gstep))
    
    # Ensure the endpoints are within the grid bounds
    x0, y0 = max(0, min(grid_size - 1, x0)), max(0, min(grid_size - 1, y0))
    x1, y1 = max(0, min(grid_size - 1, x1)), max(0, min(grid_size - 1, y1))
    
    # Bresenham's line algorithm
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    
    while True:
        # Mark the current grid cell as filled
        grid[grid_size - 1 - y0][x0] += weigth 
                      
        if x0 == x1 and y0 == y1:
            break
        
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy
    
    return grid

def add_to_grid(grid,
                grange,
                gstep,
                positions,
                parents,
                xaxis=0, 
                yaxis=1, 
                weigth=1):
    for i in range(2,len(parents)):
        grid = bresenham_line(
                                positions[parents[i]][xaxis] + 100,
                                positions[parents[i]][yaxis],
                                positions[i][xaxis] + 100,
                                positions[i][yaxis],
                                grid=grid,
                                grange=grange,
                                gstep=gstep,
                                weigth=weigth,
        )
    return grid

def rasterize(positions,
              parents, 
              xaxis=0, 
              yaxis=1, 
              weigth=0.1, 
              frame_skip=5,
              grid_range=200,
              grid_step=0.5):
    grid = Grid(grange=grid_range, gstep=grid_step)
    for i in range(0, positions.shape[0], frame_skip):
        grid.grid = add_to_grid(grid.grid,
                                grid.grange,
                                grid.gstep,
                                positions[i],
                                parents,
                                xaxis=xaxis, 
                                yaxis=yaxis, 
                                weigth=weigth)
    return grid