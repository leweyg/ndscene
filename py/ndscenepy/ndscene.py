
# ndScene Summary
# sceneND { objectND { tensorND { dataND { tensor/compression/path } } } } 

def TodoND(desc=""):
    raise Exception("TODO_ND:" + desc)
def NDTODO(desc=""):
    TodoND(desc)

"""Data is used to load and store tensors, converting between remote 'path',
to local 'buffer' and via 'format' (dtype or MIME) to 'tensor' on demand."""
class DataND():
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
        ans = DataND()
        ans.buffer = text
        ans.format = "text/plain"
        return ans
    @staticmethod
    def from_tensor(tensor):
        ans = DataND()
        ans.tensor = tensor
        return ans


"""nestable/recursivly-defined tensor
"""
class TensorND():
    key :str = None
    size :int = None
    shape :list["TensorND"] = None
    dtype :str = None
    data :DataND = None

    @staticmethod
    def from_data(data:DataND):
        ans = TensorND()
        ans.data = data
        return ans

    @staticmethod
    def from_tensor(tensor):
        if (isinstance(tensor, TensorND)):
            return tensor
        return TensorND.from_data(DataND.from_tensor(tensor))

    def __init__(self, initSize:int=None, initKey:str=None, initData:DataND=None):
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
        shape = TensorND.shape_of(input_list)
        ans = TensorND()
        ans.shape = [TensorND(s) for s in shape]
        ans.data = DataND()
        ans.data.buffer = input_list
        return ans



"""n-dimensional scene graph element
"""
class ObjectND():
    key :str = None
    """Name/id of this ObjectND"""

    parents :list["ObjectND"] = []
    """Parent objects which this concatenates into"""

    children :list["ObjectND"] = []
    """Child object which concatenate into this one"""

    data :TensorND = None
    """Content to be multipled by pose or encoded via unpose"""

    pose :TensorND = None
    """Transform from local/child space to parent space"""

    unpose :TensorND = None
    """Transform to data/child space from local space"""

    def __str__(self):
        ans = "{"; #\"ndobject\":true,"
        if (self.key):
            ans += f"\"key\"={self.key},"
        if (self.pose != 0):
            ans += f"\"pose\"={self.pose},"
        ans += "}"
        return ans

class SceneND():
    root :ObjectND = None
    """Root query in the scene (not always world space)"""
    objects :dict[str,ObjectND] = {}
    """Named ObjectND's in the scene by name"""
    tensors :dict[str,TensorND] = {}
    """Named tensors in the scene by name"""

    def add_tensor(self, path:str, tensor:TensorND):
        self.tensors[path] = tensor
    def add_data(self, path:str, data:DataND):
        self.tensors[path] = TensorND.from_data(data)

class JsonND:
    # JSON read/write (static methods):
    @staticmethod
    def ensure_tensor(obj)->TensorND:
        if (isinstance(obj, TensorND)):
            return obj
        if (isinstance(obj, str)):
            ans = TensorND()
            ans.data = JsonND.ensure_data(obj)
            ans.size = len(obj)
            ans.dtype = "char"
            ans.shape = [TensorND(ans.size, "letter")]
            return ans
        TodoND()
        return None
    @staticmethod
    def ensure_data(obj)->DataND:
        if (isinstance(obj, DataND)):
            return obj
        if (isinstance(obj, str)):
            ans = DataND()
            ans.text = str
            return ans
        TodoND(f"ensure_data:{obj}")
        return None
    @staticmethod
    def json_object(data:ObjectND)->dict:
        TodoND()
        return None


class MathND:
    @staticmethod
    def inverse_pose(pose:TensorND):
        TodoND()
        return pose;
    @staticmethod
    def apply_pose_to_data(pose:TensorND, data:TensorND):
        TodoND() # batch-matrix-multiply by default
        return pose * data

class TorchND:
    @staticmethod
    def torch():
        import torch
        return torch
    @staticmethod
    def size(tensornd : TensorND):
        import torch
        shape = torch.Size( [i.size for i in tensornd.shape] )
        return shape
    @staticmethod
    def tensor(tensornd : TensorND):
        if (tensornd.data.tensor):
            return tensornd.data.tensor
        import torch
        shape = torch.Size( [i.size for i in tensornd.shape] )
        ans = torch.tensor( tensornd.data.buffer )
        print("ans.shape=", ans.shape)
        print("shape", shape)
        assert( ans.shape == shape )
        tensornd.data.tensor = ans
        return ans

class RenderND:
    state_result :TensorND = None
    stack_pose   :list[TensorND] = None

    # Core API (TensorND only):
    def set_result(self, res :TensorND):
        self.state_result = JsonND.ensure_tensor(res)
        
    def push_pose(self, pose: TensorND, unpose :TensorND):
        """pushes a transform onto the stack (on the right), if pose is not provided, and unpose is provided, then the inverse of unpose will be pushed, otherwise it will be ignored."""
        if (pose):
            self.stack_pose.append(pose)
            return
        if (unpose):
            self.stack_pose.append(MathND.inverse_pose(unpose))
            return;
        raise "Either pose or unpose must be defined for 'push_pose'."

    def apply_data(self, data :TensorND):
        """concatenates the data to existing input data given the current transform stack."""
        ans = data
        for p in reversed(self.stack_pose):
            ans = MathND.apply_pose_to_data(p, ans)
        if (self.state_result):
            self.state_result.copy(ans)
            return self.state_result
        return ans

    def pop_pose(self):
        n = len(self.stack_pose)
        if (n <= 0):
            raise "Can't call 'pop_pose' on an empty pose stack."
        self.stack_pose.pop()

    def get_result(self) -> TensorND:
        """returns the data transformed by the poses"""
        return self.state_result
