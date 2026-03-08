

import imageio


def main_processor():
    path_src = "browser/ctscan/orange_615.gif"
    
    original = imageio.v3.imread(path_src)
    print(original.shape) # (48, 615, 615, 3)

    square_size = 512
    crop = center_crop(original, square_size)

    frame_count = crop.shape[0]
    frame_paths = []
    for frame_index in range(frame_count):
        slice = crop[frame_index]
        path_dst = f"browser/ctscan/orange/slice_{frame_index}.png"
        frame_paths.append(path_dst)
        print(f"Saving to {path_dst}")
        imageio.v3.imwrite(path_dst, slice)

    


def center_crop(frame, target_square):
    height = frame.shape[1]
    width = frame.shape[2]
    yoffset = (height - target_square) // 2
    xoffset = (width - target_square) // 2
    ans = frame[:,yoffset:yoffset+target_square, xoffset:xoffset+target_square, :]
    return ans

main_processor()