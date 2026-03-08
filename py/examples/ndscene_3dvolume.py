

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
    
    opacities = volume[:,:,:,1]
    fills = volume[:,:,:,0]
    fills[:,:,:] = numpy.zeros_like(fills)
    fills = fills.astype(int)
    center_pos = fills.shape
    center_pos = [center_pos[0]//2, center_pos[1]//2, center_pos[2]//2]


    extra_opacity = numpy.expand_dims( opacities, axis=-1)
    volume = numpy.concat((volume,extra_opacity), axis=3)
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