
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

    def ensure_tensor(self):
        if (self.tensor):
            return self.tensor
        if (self.format):
            return self.tensor_formatted()
        NDTODO()

    def ensure_buffer(self):
        if (self.buffer):
            return self.buffer
        NDTODO()
        
    def tensor_formatted(self):
        if (self.format == "text/plain"):
            data = self.ensure_buffer()
            import numpy
            ans = numpy.array( list(data), dtype='U1' )
            self.tensor = ans
            return ans
        NDTODO()

    @staticmethod
    def from_text(text:str):
        ans = NDData()
        ans.buffer = text
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

    def __init__(self, initSize:int=None, initKey:str=None, initData:NDData=None):
        if (initSize):
            self.size = initSize
        if (initKey):
            self.key = initKey
        if (initData):
            self.data = initData
        pass

    def __str__(self):
        ans = "{"
        if (self.key):
            ans += f"\"{self.key}\":"
        if (self.size):
            ans += "x" + self.size
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



"""n-dimensional scene graph element
"""
class NDObject():
    key :str = None
    """Name/id of this NDObject"""

    parents :list["NDObject"] = []
    """Parent objects which this concatenates into"""

    children :list["NDObject"] = []
    """Child object which concatenate into this one"""

    data :NDTensor = None
    """Content to be multipled by pose or encoded via unpose"""

    pose :NDTensor = None
    """Transform from local/child space to parent space"""

    unpose :NDTensor = None
    """Transform to data/child space from local space"""

    def __str__(self):
        ans = "{"; #\"ndobject\":true,"
        if (self.key):
            ans += f"\"key\"={self.key},"
        if (self.pose != 0):
            ans += f"\"pose\"={self.pose},"
        ans += "}"
        return ans

class NDScene():
    root :NDObject = None
    """Root query in the scene (not always world space)"""
    objects :dict[str,NDObject] = {}
    """Named NDObject's in the scene by name"""
    tensors :dict[str,NDTensor] = {}
    """Named tensors in the scene by name"""

    def add_tensor(self, path:str, tensor:NDTensor):
        self.tensors[path] = tensor
    def add_data(self, path:str, data:NDData):
        self.tensors[path] = NDTensor.from_data(data)

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


class NDMath:
    @staticmethod
    def inverse_pose(pose:NDTensor):
        NDTODO()
        return pose;
    @staticmethod
    def apply_pose_to_data(pose:NDTensor, data:NDTensor):
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

    # Core API (NDTensor only):
    def set_result(self, res :NDTensor):
        self.state_result = NDJson.ensure_tensor(res)
        
    def push_pose(self, pose: NDTensor, unpose :NDTensor):
        """pushes a transform onto the stack (on the right), if pose is not provided, and unpose is provided, then the inverse of unpose will be pushed, otherwise it will be ignored."""
        if (pose):
            self.stack_pose.append(pose)
            return
        if (unpose):
            self.stack_pose.append(NDMath.inverse_pose(unpose))
            return;
        raise "Either pose or unpose must be defined for 'push_pose'."

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
