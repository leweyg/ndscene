
# ndScene Summary
# NDScene { NDObject { NDTensor { NDData { tensor/compression/path } } } } 

def NDTODO(desc=""):
    raise Exception("NDTODO:" + desc)

"""Data is used to load and store tensors, converting between remote 'path',
to local 'buffer' and via 'format' (dtype or MIME) to 'tensor' on demand."""
class NDData():
    tensor = None
    """Native tensor-representation, consider 'ensure_tensor()' to access """
    format = None # MIME type or dtype
    """dtype or MIME type"""
    buffer = None
    """Loaded buffer representation"""
    path : str = None
    """File or URL path to the buffer of type format to load into the tensor"""

    def __init__(self, buffer=None, format=None, path=None, tensor=None):
        if (buffer):
            self.buffer = buffer
        if (format):
            self.format = format
        if (path):
            self.path = path
        if (tensor):
            self.tensor = tensor
        pass

    def ensure_tensor(self, scene:"NDScene"):
        if (self.tensor):
            return self.tensor
        if (self.buffer):
            if (self.format):
                return self.tensor_formatted()
            NDTODO()
        if (self.path):
            format = self.format if self.format else NDData.geuss_format_from_path(self.path)
            if (format.startswith("image/")):
                import imageio
                self.tensor = imageio.v3.imread(scene.path_root + "/" + self.path)
                return self.tensor
            NDTODO()
        NDTODO()


    @staticmethod
    def geuss_format_from_path(path:str):
        path = path.lower()
        if (path.endswith(".png")):
            return "image/png"
        if (path.endswith(".jpg") or path.endswith(".jpeg")):
            return "image/jpeg"
        if (path.endswith(".png")):
            return "image/png"
        NDTODO("geuss_format_from_path for '" + path + "'")
        return None

    def ensure_buffer(self):
        if (self.buffer):
            return self.buffer
        NDTODO()
        
    def tensor_formatted(self):
        if (self.format == "text/plain"):
            data = self.ensure_buffer()
            import numpy
            ans = numpy.array( list(data) ) #, dtype='U1' )
            self.tensor = ans
            return ans
        NDTODO()

    def __str__(self):
        ans = "{";
        if (self.tensor is not None):
            ans += f"\"tensor\":{self.tensor}"
        else:
            if (self.format):
                ans += f"\"format\":\"{self.format}\","
            if (self.buffer is not None):
                ans += f"\"buffer\":\"{self.buffer}\","
            if (self.path):
                ans += f"\"path\":\"{self.path}\","
        ans += "}"
        return ans;

    @staticmethod
    def from_text(text:str):
        ans = NDData()
        ans.buffer = [ord(t) for t in text]
        ans.format = "text/plain"
        return ans
    @staticmethod
    def from_tensor(tensor):
        ans = NDData()
        ans.tensor = tensor
        return ans


"""nestable/recursivly-defined tensor
"""
class NDTensor():
    key :str = None
    size :int = None
    shape :list["NDTensor"] = None
    dtype :str = None
    data :NDData = None

    def __init__(self, size:int=None, key:str=None, data:NDData=None):
        if (size):
            self.size = size
        if (key):
            self.key = key
        if (data):
            self.data = data
        pass


    @staticmethod
    def native_tensor_from_dict(obj, scene):
        import torch
        torchDType = torch.float
        shape = []
        data = None
        if ("dtype" in obj):
            pass # more of these
        if ("shape" in obj):
            for sv in obj["shape"]:
                if (isinstance(sv,int)):
                    shape.append(sv)
                else:
                    NDTODO()
        if ('data' in obj):
            dv = obj['data']
            if (isinstance(dv,dict)):
                if ("0" in dv or 0 in dv):
                    dv = list(dv.values())
            if (isinstance(dv,list)):
                data = dv
        ans = torch.tensor(data, dtype=torchDType).reshape(shape)
        return ans
    
    def ensure_tensor(self, scene:"NDScene"):
        if (self.data and self.data.tensor):
            return self.data.tensor
        if (self.data):
            tensor = self.data.ensure_tensor(scene)
            if (not self.shape):
                self.shape = [NDTensor(size=si) for si in tensor.shape]
            return tensor
        NDTODO()
    
    @staticmethod
    def ensure_is_tensor(obj, scene):
        if (isinstance(obj,NDTensor)):
            return obj
        return NDTensor.from_object(obj, scene)
    
    @staticmethod
    def from_method_object(obj, scene:"NDScene"):
        if (isinstance(obj,str)):
            if (obj not in scene.tensors):
                method = scene.methods.get(obj)
                if (not method):
                    method = NDMethod(obj)
                    scene.methods[obj] = method
                return method
        return NDTensor.from_object(obj, scene);

    @staticmethod
    def from_object(obj, scene):
        if (isinstance(obj,str)):
            if (scene and obj in scene.tensors):
                raw_state = scene.tensors[obj]
                ans = NDTensor.ensure_is_tensor(raw_state)
                if (raw_state is not ans):
                    # modify the scene to reference the allocated tensor
                    scene.tensors[obj] = ans
                return ans
            assert(not isinstance(obj,str))
        if (isinstance(obj,NDTensor)):
            return obj
        if (isinstance(obj,NDData)):
            return NDTensor.from_data(obj)
        if (NDTorch.is_tensor(obj)):
            ans = NDTensor()
            ans.data = NDData.from_tensor(obj)
            return ans
        if (isinstance(obj,dict)):
            # generic python dictionary, let's check it out:
            if ("shape" in obj and "data" in obj):
                return NDTensor.native_tensor_from_dict(obj, scene)
            if ("shape" in obj):
                NDTODO()
            if ("data" in obj):
                NDTODO()
            if ("path" in obj):
                path = obj['path']
                if (path in scene.tensors):
                    return scene.tensors[path]
                loader = NDTensor.from_data(NDData(path=path))
                scene.tensors[path] = loader
                return loader
            dans = {}
            for k,v in obj.items():
                dans[k]  = NDTensor.from_object(v, scene)
            return dans
        NDTODO()

    def shape_ensure(self):
        if (self.shape is None):
            self.shape = []
        return self.shape

    def shape_find(self, key:str):
        if (self.shape is None):
            return None
        for sh in self.shape:
            if (sh.key == key):
                return sh
        return None

    def shape_append(self, tensor:"NDTensor"):
        assert(isinstance(tensor,NDTensor))
        if (tensor.key):
            assert(self.shape_find(tensor.key) is None)
        self.shape_ensure().append(tensor)

    @staticmethod
    def from_data(data:NDData):
        ans = NDTensor()
        ans.data = data
        return ans

    @staticmethod
    def from_tensor(tensor):
        if (isinstance(tensor, NDTensor)):
            return tensor
        return NDTensor.from_data(NDData.from_tensor(tensor))

    def __str__(self):
        ans = "{"
        if (self.key):
            ans += f"\"{self.key}\":"
        if (self.size):
            ans += f"x{self.size}"
        if (self.shape and len(self.shape) > 0):
            ans += "["
            for s in self.shape:
                ans += str(s) + ","
            ans += "]"
        if (self.dtype):
            ans += f"<{self.dtype}>"
        if (self.data):
            ans += f"={self.data}"
        ans += "}"
        return ans
    
    def __repr__(self):
        return str(self)
    
    @staticmethod
    def shape_of(input_list : list)->list[int]:
        shape = [ len(input_list) ]
        if (shape[0] > 0):
            first = input_list[0]
            while (isinstance(first, list) and len(first)>0):
                shape.append(len(first))
                first = first[0]
        return shape
    @staticmethod
    def from_arrays(input_list):
        shape = NDTensor.shape_of(input_list)
        ans = NDTensor()
        ans.shape = [NDTensor(s) for s in shape]
        ans.data = NDData()
        ans.data.buffer = input_list
        return ans



"""n-dimensional scene graph object/element
"""
class NDObject():
    name :str = None
    """Name/id of this NDObject"""

    parents :list["NDObject"] = None
    """Parent/output objects which this concatenates into"""

    children :list["NDObject"] = None
    """Child/input objects which concatenate into this one"""

    content :NDTensor = None
    """Content to be multipled by pose or encoded via unpose"""

    pose :NDTensor = None
    """Transform from local/child space to parent space"""

    unpose :NDTensor = None
    """Transform to data/child space from local space"""

    def __init__(self, key:str=None, pose:NDTensor=None, content:NDTensor=None, scene:"NDScene"=None):
        if (key):
            self.name = key
        if (content is not None):
            content = NDTensor.ensure_is_tensor(content, scene)
            self.content = content
        if (pose is not None):
            self.pose = pose
        pass

    @staticmethod
    def from_dict(obj, scene):
        ans = NDObject()
        ans.__dict__.update(obj)
        if ans.children:
            nc = []
            for child in ans.children:
                if (isinstance(child,str)):
                    nchild = scene.objects[child]
                else:
                    assert(isinstance(child,dict))
                    nchild = NDObject.from_dict(child, scene)
                nc.append(nchild)
            ans.children = nc
        if ans.pose:
            ans.pose = NDTensor.from_method_object(ans.pose, scene)
        if ans.unpose:
            ans.unpose = NDTensor.from_method_object(ans.unpose, scene)
        if ans.content:
            ans.content = NDTensor.from_object(ans.content, scene)
        return ans
    
    def parent_any(self):
        if (self.parents and len(self.parents)):
            return self.parents[0]
        return None
    
    def parent_any_to_world(self):
        # todo: improve this
        return self.parent_any()
    
    def child_find_parent(self, target, recursive=True):
        if (self.children is None):
            return None
        for child in self.children:
            if child == target:
                return self
            if recursive:
                ans = child.child_find_parent(target, recursive)
                if (ans): return ans
        return None

    def child_find(self, name, recursive=False):
        if (self.children is None):
            return None
        for k in self.children:
            if k.name == name:
                return k
        if recursive:
            for k in self.children:
                ans = k.child_find(name, recursive=recursive)
                if ans: return ans
        return None

    def child_add(self, child : "NDObject"):
        assert(not child.name or not self.child_find(child.name))
        assert(child is not self)
        if (self.children is None):
            self.children = []
        self.children.append(child)
        return self

    def __str__(self):
        ans = "{"; #\"ndobject\":true,"
        if (self.name):
            ans += f"\"name\"={self.name},"
        if (self.pose is not None):
            ans += f"\"pose\"={self.pose},"
        if (self.unpose is not None):
            ans += f"\"unpose\"={self.unpose},"
        if (self.content is not None):
            ans += f"\"data\"={self.content},"
        if (self.children is not None and len(self.children) > 0):
            ans += "\"children\":["
            for child in self.children:
                # TODO(leweyg): check recursion here
                ans += f"{child},"
            ans += "]"
        ans += "}"
        return ans
    
class NDMethod():
    name :str = None
    
    def __init__(self, name:str):
        self.name = name
        pass

class NDScene():
    root :NDObject = None
    """Root query in the scene (not always world space)"""
    objects :dict[str,NDObject] = {}
    """Named NDObject's in the scene by name"""
    tensors :dict[str,NDTensor] = {}
    """Tensor of tensors in the scene"""
    methods :dict[str,NDMethod] = {}
    """Table of methods/functions used by the scene"""
    path_root :str = None
    """Path root used for file loading"""

    @staticmethod
    def from_object(obj:dict, path_dir:str=""):
        ans = NDScene()
        ans.__dict__.update(obj)
        ans.path_root = path_dir
        nv = {}
        for k,v in ans.tensors.items():
            ans.tensors[k] = NDTensor.from_object(v, ans)
        for k,v in ans.objects.items():
            ans.objects[k] = NDObject.from_dict(v, ans)
        ans.root = NDObject.from_dict(ans.root, ans)
        return ans
    
    @staticmethod
    def from_file_path(self, path:str):
        return NDJson.scene_from_path(path)
    
    def ensure_parents(self, obj:NDObject):
        if (obj.parent_any()):
            return
        par_top = self.root.child_find_parent(obj)
        par = par_top
        if (par):
            obj.parents = [par]
            while (par and not par.parent_any()):
                next_par = self.root.child_find_parent(par)
                if (next_par):
                    par.parents = [next_par]
                par = next_par
        return par_top


    def add_tensor(self, path:str, tensor:NDTensor):
        if (not tensor.key):
            tensor.key = path
        assert(not path in self.tensors)
        self.tensors[path] = tensor

    def add_data(self, path:str, data:NDData):
        tensor = NDTensor.from_data(data)
        tensor.key = path
        self.add_tensor(path, tensor)

    def _inner_objects_str_(self):
        ans = ""
        for k,v in self.objects.items():
            if (ans != ""):
                ans += ","
            ans += f"\"{k}\":{v}"
        return ans

    def __str__(self):
        ans = "{"
        if (self.root):
            ans += f"\"root\"={self.root},"
        ans += f"\"tensors\"={self.tensors},"
        ans += "\"objects\"={" + self._inner_objects_str_() + "}"
        ans += "}"
        return ans

class NDJson:
    # JSON read/write (static methods):
    @staticmethod
    def ensure_tensor(obj)->NDTensor:
        if (isinstance(obj, NDTensor)):
            return obj
        if (isinstance(obj, str)):
            ans = NDTensor()
            ans.data = NDJson.ensure_data(obj)
            ans.size = len(obj)
            ans.dtype = "char"
            ans.shape = [NDTensor(ans.size, "letter")]
            return ans
        NDTODO()
        return None
    @staticmethod
    def ensure_data(obj)->NDData:
        if (isinstance(obj, NDData)):
            return obj
        if (isinstance(obj, str)):
            ans = NDData()
            ans.text = str
            return ans
        NDTODO(f"ensure_data:{obj}")
        return None
    @staticmethod
    def json_object(data:NDObject)->dict:
        NDTODO()
        return None
    @staticmethod
    def scene_from_path(path:str):
        with open(path,"r") as file:
            json_text = file.read()
        import json
        json_obj = json.loads(json_text)
        path_root = ""
        if ("/" in path):
            path_root = path[:path.rfind("/")]
        return NDScene.from_object(json_obj, path_root)
    @staticmethod
    def find_child_by_name(obj, name:str):
        if ("name" in obj):
            if (obj["name"] == name):
                return obj
        if ("children" in obj):
            for child in obj["children"]:
                ans = NDJson.find_child_by_name(child, name)
                if (ans): return ans
        return None


class NDMath:
    @staticmethod
    def inverse_pose(pose:NDTensor):
        import torch
        pose = NDTorch.ensure_tensor(pose)
        pose = torch.linalg.inv(pose)
        return pose;
    @staticmethod
    def apply_pose_to_data(pose:NDTensor, data:NDTensor):
        if (pose is None):
            return data
        NDTODO() # batch-matrix-multiply by default
        return pose * data

class NDTorch:
    @staticmethod
    def torch():
        import torch
        return torch
    @staticmethod
    def size(NDTensor : NDTensor):
        import torch
        shape = torch.Size( [i.size for i in NDTensor.shape] )
        return shape
    @staticmethod
    def is_tensor(obj):
        import torch
        return torch.is_tensor(obj)
    @staticmethod
    def ensure_tensor(obj):
        import numpy
        import torch
        if (isinstance(obj, NDTensor)):
            obj = obj.ensure_tensor()
        if (torch.is_tensor(obj)):
            return obj
        if (isinstance(obj,numpy.ndarray)):
            return torch.from_numpy(obj)
        NDTODO()
        
    @staticmethod
    def tensor(NDTensor : NDTensor):
        if (NDTensor.data.tensor):
            return NDTensor.data.tensor
        import torch
        shape = torch.Size( [i.size for i in NDTensor.shape] )
        ans = torch.tensor( NDTensor.data.buffer )
        print("ans.shape=", ans.shape)
        print("shape", shape)
        assert( ans.shape == shape )
        NDTensor.data.tensor = ans
        return ans

class NDRender:
    state_result :NDTensor = None
    stack_pose   :list[NDTensor] = None
    scene :NDScene = None

    def __init__(self):
        self.stack_pose = []

    # Scene API (NDScene level):
    def update_object_from_world(self, target:NDObject, scene:NDScene):
        assert(self.state_result is None)
        assert(self.scene is None)
        assert(len(self.stack_pose) == 0)
        assert(scene)
        assert(target.content is not None)
        self.scene = scene

        self.scene.ensure_parents(target)

        excluding = {}
        excluding[target] = True
        self.set_result(target.content)
        world = self.push_scene_back_to_world(target)
        world_depth = len(self.stack_pose)
        
        self.push_children_data(world, excluding)
        done_depth = len(self.stack_pose)
        assert(done_depth == world_depth)
        self.stack_pose.clear()
        self.state_result = None
        self.scene = None

    def push_scene_back_to_world(self, target:NDObject):
        cursor = target
        latest = cursor
        while (cursor):
            # first push inverses back to world:
            latest = cursor
            self.push_pose(cursor.unpose, cursor.pose)
            cursor = cursor.parent_any_to_world()
        return latest
    
    def push_children_data(self, target:NDObject, excluding:dict[NDObject,bool]):
        if (target in excluding):
            return;
        self.push_pose(target.pose, target.unpose)
        if (target.content is not None):
            self.apply_data(target.content)
        if (target.children):
            for child in target.children:
                self.push_children_data(child, excluding)
        self.pop_pose()

    # Rendering API (NDTensor only):
    def set_result(self, res :NDTensor):
        self.state_result = NDJson.ensure_tensor(res)
        
    def push_pose(self, pose: NDTensor, unpose :NDTensor):
        """pushes a transform onto the stack (on the right), if pose is not provided, and unpose is provided, then the inverse of unpose will be pushed, otherwise it will be ignored."""
        if (pose is not None):
            self.stack_pose.append(pose)
            return
        if (unpose is not None):
            self.stack_pose.append(NDMath.inverse_pose(unpose))
            return;
        self.stack_pose.append(None)

    def apply_data(self, data :NDTensor):
        """concatenates the data to existing input data given the current transform stack."""
        ans = data
        for p in reversed(self.stack_pose):
            ans = NDMath.apply_pose_to_data(p, ans)
        if (self.state_result):
            self.state_result.copy(ans)
            return self.state_result
        return ans

    def pop_pose(self):
        n = len(self.stack_pose)
        if (n <= 0):
            raise "Can't call 'pop_pose' on an empty pose stack."
        self.stack_pose.pop()

    def get_result(self) -> NDTensor:
        """returns the data transformed by the poses"""
        return self.state_result
