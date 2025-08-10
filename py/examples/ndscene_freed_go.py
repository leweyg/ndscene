
# without install:
import os
print("pwd=", os.getcwd())
import sys
sys.path.append(os.getcwd())

import ndscenepy.ndscene as ndscene

def read_file_by_path(path):
    with open(path,"r") as file:
        return file.read()

def bounds_2n_from_mn(vals):
    import torch
    low = vals.min(dim=0).values.unsqueeze(0)
    high = vals.max(dim=0).values.unsqueeze(0)
    bounds = torch.concat([low,high],dim=0)
    return bounds

def calc_board_size_2n_from_mn(scene):
    import torch
    summary = scene.tensors['projected_points']
    summary = summary[:,0,:]
    summary = bounds_2n_from_mn(summary)
    pass


def main_freed_go_test():
    print("Main Freed Go test...")
    scene = ndscene.NDJson.scene_from_path("../json/freed_go/view_3_scene.json")
    ndscene.NDMethod.setup_standard_methods(scene)
    board_size = calc_board_size_2n_from_mn(scene)
    voxel_scene = ndscene.NDJson.scene_from_path("../json/freed_go/voxels.json")

    world = scene.root.child_find("world")
    voxels = voxel_scene.root.child_find("voxels")
    # FIX THIS: world.child_add(voxels)

    test_camera = scene.root.child_find('camera', recursive=True)
    image_path = test_camera.child_find('image', recursive=True)
    image_tensor = image_path.content.ensure_tensor(scene)
    #print("test_camera:", test_camera)
    # rendering tests:
    renderer = ndscene.NDRender()
    renderer.update_object_from_world(image_path, scene)
    #print("Scene=", scene)
    image_path.content.data.save_to_path("test_output.png")
    pass;


if __name__ == "__main__":
    main_freed_go_test()

