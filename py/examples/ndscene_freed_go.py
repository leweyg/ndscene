
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
    #print(bounds)
    return bounds

def unit_to_bounds_from_bounds_2n(bounds):
    import torch
    low = bounds[0]
    high = bounds[1]
    n = bounds.shape[1]
    ans = torch.zeros( [n+1,n] )
    for di in range(n):
        ans[di,di] = high[di] - low[di]
        ans[n,di] = low[di]
    #print(ans)
    #print("BoundsPose=", ans.shape, "\n = ", ans)
    return ans

def calc_board_size_2n_from_mn(scene):
    import torch
    summary = scene.tensors['projected_points']
    summary = summary[:,0,:]
    summary = bounds_2n_from_mn(summary)
    summary = unit_to_bounds_from_bounds_2n(summary)
    pass


def main_freed_go_test():
    print("Main Freed Go test...")
    voxel_scene = ndscene.NDJson.scene_from_path("../json/freed_go/voxels.json")
    voxels = voxel_scene.root.child_find('voxels')
    voxel_data = voxels.child_find('voxel_data')
    assert(voxels)

    input_views = [
        "../json/freed_go/view_1_scene.json",
        "../json/freed_go/view_2_scene.json",
        "../json/freed_go/view_3_scene.json",
    ]
    for view_path in input_views:
        scene = ndscene.NDJson.scene_from_path(view_path)
        ndscene.NDMethod.setup_standard_methods(scene)
        #board_size = calc_board_size_2n_from_mn(scene)

        world = scene.root.child_find("world")
        
        world.child_add(voxels) # adds the voxels to the main scene

        test_camera = scene.root.child_find('camera', recursive=True)
        image_path = test_camera.child_find('image', recursive=True)
        image_tensor = image_path.content.native_tensor(scene)
        #print("image_tensor.shape=", image_tensor.shape)
        # reset all (except alpha) to zero/black:
        
        #print("test_camera:", test_camera)
        # rendering tests:
        renderer = ndscene.NDRender()

        # 1. first sample the image into the voxels:
        # renderer.update_object_from_world(voxel_data, scene)

        # 2. draw the voxels to the world:
        #image_tensor[:,:,0:3].fill_(0)
        renderer.update_object_from_world(image_path, scene)
        
        #print("Scene=", scene)
        image_path.content.data.path += ".out.png"
        image_path.content.data.save_to_path()
        pass;


if __name__ == "__main__":
    main_freed_go_test()

