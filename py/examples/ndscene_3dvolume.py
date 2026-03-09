

import imageio
import numpy

def main_processor():
    path_src = "browser/ctscan/orange_615.gif"
    
    original = imageio.v3.imread(path_src)
    print(original.shape) # (48, 615, 615, 3)

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
    # we'll accumulate a running total in fills; start with zeros
    fills = numpy.zeros_like(opacities, dtype=numpy.int32)
    # compute the starting (center) voxel
    center_pos = [fills.shape[0]//2, fills.shape[1]//2, fills.shape[2]//2]

    # perform a breadth‑first flood fill from the center
    from collections import deque
    queue = deque()
    cx, cy, cz = center_pos
    fills[cx, cy, cz] = int(opacities[cx, cy, cz])
    queue.append((cx, cy, cz))

    # 6‑connected neighborhood (axis‑aligned)
    neighbors = ((1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1))
    print("Starting flood fill...")
    while queue:
        x, y, z = queue.popleft()
        current_sum = fills[x, y, z]
        for dx, dy, dz in neighbors:
            nx, ny, nz = x + dx, y + dy, z + dz
            if 0 <= nx < fills.shape[0] and 0 <= ny < fills.shape[1] and 0 <= nz < fills.shape[2]:
                if fills[nx, ny, nz] == 0:
                    # add neighbor's opacity to running total and mark it
                    new_sum = current_sum + int(opacities[nx, ny, nz])
                    fills[nx, ny, nz] = new_sum
                    queue.append((nx, ny, nz))

    print("Done flood fill...")
    fills = ( fills / ( 256 * 1 ) ) % 256

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