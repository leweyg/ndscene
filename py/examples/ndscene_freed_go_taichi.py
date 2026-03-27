
# without install:
import os
print("pwd=", os.getcwd())
import sys
sys.path.append(os.getcwd())

print("Importing ndscene...")
import ndscenepy.ndscene as ndscene
print("Importing taichi...")
import taichi 
print("Starting...")

_TAICHI_INITIALIZED = False


def ensure_taichi_initialized():
    global _TAICHI_INITIALIZED
    if _TAICHI_INITIALIZED:
        return

    init_errors = []
    for arch in (taichi.vulkan, taichi.metal):
        try:
            print(f"Initializing Taichi with arch={arch}...")
            taichi.init(arch=arch)
            _TAICHI_INITIALIZED = True
            return
        except Exception as exc:
            init_errors.append(f"{arch}: {exc}")

    raise RuntimeError("Unable to initialize Taichi. Tried: " + " | ".join(init_errors))

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


class GoGameAppState:
    def __init__(self, scene:ndscene.NDScene, img_path:ndscene.NDObject):
        self.scene = scene
        self.img_path = img_path
        self.img_tensor = img_path.content.native_tensor(scene)
        self.camera = scene.root.child_find('camera', recursive=True)
        self.camera_initial = self.camera.pose.clone()
        self.renderer = ndscene.NDRender()
        self.is_running = True
        self.is_paused = False
        self.step_frames = 0
        self.frame_dt = 1.0 / 60.0
        self.accum_time = 0.0
        self.window = None
        self.canvas = None
        self.window_gui = None
        pass
    def main_redraw(self):
        self.img_tensor.fill_(0)
        self.renderer.update_object_from_world(self.img_path, self.scene)
        pass
    def run_main_init(self):
        print("GoGameAppState: run_main_init")
        ensure_taichi_initialized()
        sz = self.img_tensor.shape
        self.window = taichi.ui.Window("Freed Go", res=(sz[0], sz[1]))
        self.canvas = self.window.get_canvas()
        self.window_gui = self.window.get_gui()

        self.main_loop()
        return
    
        # main loop:
    def main_loop(self):
        frameIndex = 0

        def handle_input():
            if self.window.get_event(taichi.ui.PRESS) and self.window.event.key == taichi.ui.ESCAPE:
                self.window.running = False
            cursor_unit = self.window.get_cursor_pos()
            cursor_unit = taichi.math.vec2(cursor_unit[0], cursor_unit[1])
            cursor_sunit = (cursor_unit * 2.0) - 1.0

            self.camera.pose.copy_( self.camera_initial )
            motion_scale = 50
            self.camera.pose[3,1] -= cursor_sunit[0] * motion_scale
            self.camera.pose[3,0] -= cursor_sunit[1] * motion_scale

            #cursor_now[0] = cursor_state[0] * gui.img.shape[0]
            #cursor_now[1] = cursor_state[1] * gui.img.shape[1]
            #cursor_now.from_numpy(numpy.array(cursor_state, dtype=numpy.float32))
            # print(cursor_now)

        while self.window.running:
            handle_input()
            self.main_redraw();
            #paint(i * 0.03 ); #, cursor_now)
            self.canvas.set_image(self.img_tensor.numpy())
            with self.window_gui.sub_window("Info", 0.02, 0.02, 0.2, 0.12) as gui:
                gui.text("Some text", color=(1.0, 1.0, 1.0))
            self.window.show()
            frameIndex += 1
    
        pass

def main_freed_go_test():
    print("Main Freed Go test...")
    voxel_scene = ndscene.NDJson.scene_from_path("../json/freed_go/voxels.json")
    voxels = voxel_scene.root.child_find('voxels')
    voxel_data = voxels.child_find('voxel_data')
    assert(voxels)

    input_views = [
        "../json/freed_go/view_1_scene.json",
        #"../json/freed_go/view_2_scene.json",
        #"../json/freed_go/view_3_scene.json",
    ]
    app_state = None
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
        #renderer.update_object_from_world(voxel_data, scene)

        # 2. draw the voxels to the world:
        #image_tensor[:,:,0:3].fill_(0)
        renderer.update_object_from_world(image_path, scene)

        app_state = GoGameAppState(scene, image_path)
        app_state.run_main_init()
        
        #print("Scene=", scene)
        #image_path.content.data.path += ".out.png"
        #image_path.content.data.save_to_path()
        pass;


if __name__ == "__main__":
    main_freed_go_test()
