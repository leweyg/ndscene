
# ndScene Summary
# NDScene { NDObject { NDTensor { NDData { tensor/compression/path } } } } 

def NDTODO(desc=""):
    print("NDTODO:", desc)
    raise Exception("NDTODO:" + desc)

"""Data is used to load and store tensors, converting between remote 'path',
to local 'buffer' and via 'format' (dtype or MIME) to 'tensor' on demand."""
class NDData():
    tensor = None
    """Native tensor-representation, consider 'native_tensor()' to access """
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

    def native_tensor(self, scene:"NDScene"):
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
                import torch
                self.tensor = torch.tensor(self.tensor)
                return self.tensor
            NDTODO()
        NDTODO()

    def save_to_path(self, new_path=None):
        if (new_path):
            self.path = new_path
        print(f"Saving to file '{self.path}'...")
        self.write_tensor_to_path(self.tensor.numpy(), self.path)

    @staticmethod
    def write_tensor_to_path(val, path):
        import imageio
        imageio.v3.imwrite(path, val)

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
        shape_product = 1
        if ("dtype" in obj):
            pass # more of these
        if ("shape" in obj):
            for sv in obj["shape"]:
                if (isinstance(sv,int)):
                    shape.append(sv)
                    shape_product *= sv
                else:
                    NDTODO()
        if ('data' in obj):
            dv = obj['data']
            if (isinstance(dv,dict)):
                if ("0" in dv or 0 in dv):
                    dv = list(dv.values())
            if (isinstance(dv,list)):
                data = dv
        ans = torch.tensor(data, dtype=torchDType);
        ans_el = ans.numel()
        if (ans_el == shape_product):
            ans = ans.reshape(shape)
        elif (ans_el == 1):
            ans = ans.repeat(shape_product).reshape(shape)
        else:
            NDTODO()
        return ans
    
    def native_tensor(self, scene:"NDScene"):
        if (self.data and self.data.tensor is not None):
            return self.data.tensor
        if (self.data):
            tensor = self.data.native_tensor(scene)
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
                    method = scene.ensure_method(obj)
                return method
        if (isinstance(obj,list)):
            return [NDTensor.from_method_object(step, scene) for step in obj]
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
        if (isinstance(obj,list)):
            if (len(obj)==0):
                NDTODO() # ?
            first = obj[0]
            if (isinstance(obj,float)):
                import torch
                return torch.tensor(obj)
            if (isinstance(obj,str)):
                NDTODO()
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

    def child_find(self, name, recursive=False, depth=0):
        if (self.name == name):
            return self
        if self.children and (recursive or depth==0):
            for k in self.children:
                ans = k.child_find(name, recursive=recursive, depth=depth+1)
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
    callback = None
    
    def __init__(self, name:str):
        self.name = name
        pass

    def apply_method_to_data(self, data:NDData, target:NDData):
        if (not self.callback):
            msg = "Need to associate a callback for '" + self.name + "'."
            print(msg)
            NDTODO(msg)
        return self.callback(data, target)
    

    @staticmethod
    def setup_standard_methods(scene:"NDScene"):
        scene.ensure_method('pixel_index_from_unit_viewport', NDMethod.std_pixel_index_from_unit_viewport)
        scene.ensure_method('unit_index_from_data', NDMethod.unit_index_from_data)
        scene.ensure_method('append_ones', NDMethod.append_ones)
        scene.ensure_method('as_vertices', NDMethod.as_vertices)

    @staticmethod
    def as_vertices(data, target=None):
        return {'vertices':data}

    @staticmethod
    def append_ones(data, target=None, dim=-1):
        import torch
        onesShape = list(data.shape);
        onesShape[dim] = 1;
        ones = torch.ones(onesShape);
        ans = torch.concat([ data, ones ], dim );
        return ans;

    @staticmethod
    def index_nd_from_data(data):
        import torch
        n = data.numel()
        d = len(data.shape)
        ids = torch.arange(n*d)
        ids = ids.reshape([n,d])
        stride = d
        for di in reversed(range(d)):
            dsize = data.shape[di]
            ids[:,di] = ((ids[:,di] / stride) % dsize)
            stride *= dsize
        #print(ids)
        return ids
    
    @staticmethod
    def unit_index_from_data(data, target):
        index_nd = NDMethod.index_nd_from_data(data)
        n = index_nd.shape[0]
        d = index_nd.shape[1]
        import torch
        scaler = [1.0/(s-1) for s in data.shape]
        scaler = torch.tensor(scaler)
        ans = scaler * index_nd
        #print(ans)
        return ans

    @staticmethod
    def std_batch_index1_from_indexN(target, src_index):
        import torch
        indexN = src_index.shape[-1]
        bounds = target.shape
        pos = src_index.int()
        pos_min = torch.zeros( [indexN], dtype=torch.int )
        pos_max = torch.tensor( bounds[0:indexN], dtype=torch.int ) - 1
        pos = torch.clamp(pos, min=pos_min, max=pos_max)
        pos = pos.int()
        stride = 1
        for di in reversed(range(indexN)):
            pos[:,di] = pos[:,di] * stride
            stride = stride * bounds[di]
        pos = torch.sum(pos, dim=-1)
        return pos

    @staticmethod
    def std_pixel_index_from_unit_viewport(data:NDData, target:NDData):
        import torch
        ans = data.copy()
        verts = ans['vertices']
        # XYZW to XY (normalize by W, drop Z)
        ws = verts[:,-1].unsqueeze(-1)
        xy = verts[:,0:2]
        xy = xy / ws
        # XY to XY1, scale to integer shape space
        xy1 = torch.concat([xy,torch.ones_like(ws)], -1)
        sx = target.shape[0].size
        sy = target.shape[1].size
        scl = [ 0.0, 0.5 * sx,
                0.5 * sy, 0.0,
                0.5 * sy, 0.5 * sx ]
        scl = torch.tensor(scl).reshape(3,2)
        xy = torch.mm(xy1, scl)
        ans['vertices'] = xy
        return ans

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
        for k,v in ans.tensors.items():
            ans.tensors[k] = NDTensor.from_object(v, ans)
        for k,v in ans.objects.items():
            ans.objects[k] = NDObject.from_dict(v, ans)
        ans.root = NDObject.from_dict(ans.root, ans)
        return ans
    
    @staticmethod
    def from_file_path(self, path:str):
        return NDJson.scene_from_path(path)
    
    def native_tensor(self, obj):
        if (isinstance(obj,NDTensor)):
            return obj.native_tensor(self)
        scene = self
        return NDTensor.ensure_is_tensor(obj,scene).native_tensor(scene)
    
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
    
    def ensure_method(self, path:str, callback=None) -> NDMethod:
        if (path in self.methods):
            ans = self.methods[path]
        else:
            ans = NDMethod(path)
            self.methods[path] = ans
        if (callback):
            ans.callback = callback
        return ans
    
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
    def native_tensor(obj)->NDTensor:
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
        pose = NDTorch.native_tensor(pose)
        pose = torch.linalg.inv(pose)
        return pose;
    @staticmethod
    def apply_pose_to_data(pose:NDTensor, data:NDTensor, target:NDTensor):
        if (pose is None):
            return data
        if (isinstance(pose,NDMethod)):
            return pose.apply_method_to_data(data,target)
        if (isinstance(pose,list)):
            for step in pose:
                data = NDMath.apply_pose_to_data(step, data, target)
            return data
        pose_is_dict = isinstance(pose,dict)
        data_is_dict = isinstance(data,dict)
        if (data_is_dict and not pose_is_dict):
            ans = data.copy()
            assert('vertices' in ans)
            verts = ans['vertices']
            import torch
            if (verts.shape[-1] == 3):
                ones = torch.ones_like(verts[:,0]).unsqueeze(-1)
                verts = torch.concat([verts,ones],-1)
            
            verts = torch.mm(verts, pose)
            ans['vertices'] = verts
            return ans
        pose_is_tensor = NDTorch.is_tensor(pose)
        data_is_tensor = NDTorch.is_tensor(data)
        if (pose_is_tensor and data_is_tensor):
            import torch
            ans = torch.mm( data, pose )
            return ans
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
    def native_tensor(obj):
        import numpy
        import torch
        if (isinstance(obj, NDTensor)):
            obj = obj.native_tensor()
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
        #print("ans.shape=", ans.shape)
        #print("shape", shape)
        #assert( ans.shape == shape )
        NDTensor.data.tensor = ans
        return ans

class NDRender:
    state_result :NDTensor = None
    state_result_tensor = None
    stack_pose   :list[NDTensor] = None
    scene :NDScene = None

    # Rendering API (NDTensor only):
    def ndBegin(self, res :NDTensor, res_tensor):
        self.state_result = NDJson.native_tensor(res)
        self.state_result_tensor = res_tensor
        
    def ndPush(self, pose: NDTensor, unpose :NDTensor):
        """pushes a transform onto the stack (on the right), if pose is not provided, and unpose is provided, then the inverse of unpose will be pushed, otherwise it will be ignored."""
        if (pose is not None):
            self.stack_pose.append(pose)
            return
        if (unpose is not None):
            self.stack_pose.append(NDMath.inverse_pose(unpose))
            return;
        self.stack_pose.append(None)

    def ndConcat(self, data :NDTensor):
        """concatenates the data to existing input data given the current transform stack."""
        ans = data
        for p in reversed(self.stack_pose):
            ans = NDMath.apply_pose_to_data(p, ans, self.state_result)
        if (self.state_result):
            dst = self.scene.native_tensor( self.state_result_tensor )
            dst = NDRender.scatterND(dst, ans['vertices'], ans.get('color'))
            self.state_result_tensor = dst
            self.state_result.data.tensor = dst
            return self.state_result
        return ans

    def ndPop(self):
        n = len(self.stack_pose)
        if (n <= 0):
            raise "Can't call 'ndPop' on an empty pose stack."
        self.stack_pose.pop()

    def ndEnd(self) -> NDTensor:
        """returns the data transformed by the poses"""
        return self.state_result
    

    def __init__(self):
        self.stack_pose = []

    # Scene API (NDScene level):
    def update_object_from_world(self, target:NDObject, scene:NDScene):
        if(target.content is None):
            print("Target must be a content node:", target.content)
            assert(False)
        assert(self.state_result is None)
        assert(self.scene is None)
        assert(len(self.stack_pose) == 0)
        assert(scene)
        
        self.scene = scene

        self.scene.ensure_parents(target)

        excluding = {}
        excluding[target] = True
        target_tensor = NDTensor.native_tensor(target.content, scene)
        self.ndBegin(target.content, target_tensor)
        world = self.push_scene_back_to_world(target)
        world_depth = len(self.stack_pose)
        res = self.state_result_tensor
        
        # main loop
        self.concat_children_recursive(world, excluding)

        done_depth = len(self.stack_pose)
        assert(done_depth == world_depth)

        res = self.ndEnd()

        self.stack_pose.clear()
        self.state_result = None
        self.state_result_tensor = None
        self.scene = None
        return res

    def push_scene_back_to_world(self, target:NDObject):
        cursor = target
        latest = cursor
        while (cursor):
            # first push inverses back to world:
            latest = cursor
            self.ndPush(cursor.unpose, cursor.pose)
            cursor = cursor.parent_any_to_world()
        return latest
    
    def concat_children_recursive(self, cursor:NDObject, excluding:dict[NDObject,bool]):
        ans = None
        stop_at_first = False
        if (cursor in excluding):
            return ans
        if (cursor.name == "voxels"):
            print("Found it...");
        self.ndPush(cursor.pose, cursor.unpose)
        if (cursor.content is not None):
            ans = self.ndConcat(cursor.content)
        if (cursor.children and (not stop_at_first or not ans)):
            for child in cursor.children:
                ans = self.concat_children_recursive(child, excluding)
                if (ans and stop_at_first): break
        self.ndPop()
        return ans

    @staticmethod
    def scatterND(target, src_index, src_vals):
        import torch
        # target.shape = [N0,N1,...,C] | N is dimension count, C is channels
        # src_index.shape = [M,N], M is vertex count, N is dimension count
        # src_vals.shape = [M,C] M instances of C channels

        # ensure the values are the right shape [M,channels]
        channels = target.shape[-1]
        vals_shape = list(src_index.shape)
        vals_shape[-1] = channels
        flat_size = 1
        for i in range(len(target.shape)-1):
            flat_size *= target.shape[i]
        if (src_vals is None):
            src_vals = torch.ones( vals_shape )
            if (target.dtype == torch.uint8):
                src_vals = src_vals * 255
                src_vals = src_vals.byte()
        # convert indexN to index1:
        pos = NDMethod.std_batch_index1_from_indexN(target, src_index)
        pos = pos.unsqueeze(-1).repeat([1,2]) # repeat indices for now...
        linear_target_shape = [ flat_size, channels ]
        flat_target = target.reshape( linear_target_shape )
        # the actual in-place '1D' scatter:
        flat_target.scatter_( 0, pos, src_vals )
        # bring back to N dimensions:
        target = flat_target.reshape( target.shape )
        return target;
