

import imageio
import numpy
from scipy.ndimage import uniform_filter

def main_processor():
    path_src = "browser/ctscan/orange_615.gif"
    
    original = imageio.v3.imread(path_src)
    print(original.shape) # (48, 615, 615, 3)

    loop_frame_count = original.shape[0]
    original = original[0:loop_frame_count//2, :, :, :]

    square_size = 512
    crop = center_crop(original, square_size)

    if True:
        crop = flood_fill_out(crop)

    frame_count = crop.shape[0]
    frame_paths = []
    for frame_index in range(frame_count):
        slice = crop[frame_index]
        path_dst = f"browser/ctscan/orange/slice_{frame_index}.png"
        frame_paths.append(path_dst)
        print(f"Saving to {path_dst}")
        imageio.v3.imwrite(path_dst, slice)

def flood_fill_out(volume):
    
    # extract opacity channel and prepare fills array
    opacities = volume[:,:,:,1]

    unit_opacity = opacities.astype(float) / 255.0
    for layer_index in range(unit_opacity.shape[0]):
        unit_opacity[layer_index] = uniform_filter(unit_opacity[layer_index], size=7)
    normals = numpy.gradient(unit_opacity)
    normals = [numpy.expand_dims(n,axis=-1) for n in normals]
    normals[0] = normals[0] / (float(volume.shape[1]) / float(volume.shape[0]))
    normal_dirs = numpy.concat( normals, axis=-1 )
    normal_len = numpy.linalg.norm(normal_dirs, axis=-1, keepdims=True)
    normal_dirs = normal_dirs / normal_len
    normal_dirs = ( ( normal_dirs * 0.5) + 0.5 )
    normal_dirs = normal_dirs * 255.0

    alphas = numpy.expand_dims(opacities, axis=-1)
    volume = numpy.repeat( alphas, axis=-1, repeats=4)
    volume[:,:,:,0] = normal_dirs[:,:,:,0]
    volume[:,:,:,1] = normal_dirs[:,:,:,1]
    volume[:,:,:,2] = normal_dirs[:,:,:,2]
    volume[:,:,:,3] = alphas.squeeze()
    return volume

    # we'll accumulate a running total in fills; start with zeros
    fills = numpy.zeros_like(opacities, dtype=numpy.int32)
    # compute the starting (center) voxel
    center_pos = [fills.shape[0]//4, fills.shape[1]//4, fills.shape[2]//4]

    # perform a breadth‑first flood fill from the center
    from collections import deque
    queue = deque()
    cx, cy, cz = center_pos
    # fills[cx, cy, cz] = int(opacities[cx, cy, cz])
    # visited = numpy.zeros_like(fills)
    queue.append((cx, cy, cz, 1))

    # 6‑connected neighborhood (axis‑aligned)
    neighbors = ((1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1))
    
    class NeighborInfo:
        def __init__(self, index, offset):
            self.index = index
            self.offset = offset
            self.opacity = 0
            self.valid = True
            pass
    fill_shape = fills.shape
    def is_valid_xyz(x,y,z):
        if ((x >= 0) and (x < fill_shape[0]) and
            (y >= 0) and (y < fill_shape[1])
            and (z >= 0) and (z < fill_shape[2])):
            return True
        return False

    neighbor_opacities = [NeighborInfo(i,neighbors[i]) for i in range(len(neighbors))]
    print("Starting flood fill...")
    step_max = fills.size
    step_cur = 0
    while queue:

        # main loop:
        x, y, z, prev_sum = queue.popleft()
        current_sum = fills[x, y, z].item()
        if current_sum != 0:
            continue # already set
        current_sum = prev_sum + int(opacities[x, y, z])
        fills[x,y,z] = current_sum

        if (step_cur % 1000) == 0:
            print(f"Step {step_cur} of {step_max} = {float(step_cur)/float(step_max)}")
        step_cur += 1

        # check all neighbor opacities:
        for n in neighbor_opacities:
            dx, dy, dz = n.offset
            nx, ny, nz = x + dx, y + dy, z + dz
            n.valid = is_valid_xyz(nx, ny, nz)
            if n.valid:
                n.opacity = int(opacities[nx, ny, nz].item())
        # sort them by highest opacity
        neighbor_opacities.sort(key=lambda ni: - ni.opacity)
        # push them onto stack in order
        for n in neighbor_opacities:
            if not n.valid:
                continue
            dx, dy, dz = n.offset
            nx, ny, nz = x + dx, y + dy, z + dz
            n_sum = fills[nx, ny, nz].item()
            if n_sum == 0:
                queue.append((nx, ny, nz, current_sum))

    print("Done flood fill...")
    fills = ( fills / ( 128 ) ) % 256

    # append an extra channel for convenience (not strictly needed)
    extra_opacity = numpy.expand_dims(opacities, axis=-1)
    volume = numpy.concatenate((volume, extra_opacity), axis=3)
    volume[:,:,:,0] = fills
    volume[:,:,:,1] = opacities
    volume[:,:,:,2] = opacities
    volume[:,:,:,3] = opacities
    return volume


def center_crop(frame, target_square):
    height = frame.shape[1]
    width = frame.shape[2]
    yoffset = (height - target_square) // 2
    xoffset = (width - target_square) // 2
    ans = frame[:,yoffset:yoffset+target_square, xoffset:xoffset+target_square, :]
    return ans

main_processor()