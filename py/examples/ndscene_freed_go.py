
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
    test_points = ndscene.NDTensor.ensure_is_tensor( scene['tensors']['projected_points'] )
    #print("Scene=", scene)
    pass;


if __name__ == "__main__":
    main_freed_go_test()

