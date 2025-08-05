
# without install:
import os
print("pwd=", os.getcwd())
import sys
sys.path.append(os.getcwd())

import ndscenepy.ndscene as ndscene

def read_file_by_path(path):
    with open(path,"r") as file:
        return file.read()

def main_freed_go_test():
    print("Main Freed Go test...")
    scene = ndscene.NDJson.scene_from_path("../json/freed_go/view_2_scene.json")
    test_points = scene.tensors['projected_points']
    test_camera = scene.root.child_find('camera', recursive=True)
    image_path = test_camera.child_find('image', recursive=True)
    image_tensor = image_path.content.ensure_tensor(scene)
    print("test_camera:", test_camera)
    test_pos_world = test_points[:,0,:]
    test_pos_lens = test_points[:,1,:]
    # rendering tests:
    renderer = ndscene.NDRender()
    renderer.update_object_from_world(image_path, scene)
    #print("Scene=", scene)
    pass;


if __name__ == "__main__":
    main_freed_go_test()

