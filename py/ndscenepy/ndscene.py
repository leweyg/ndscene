
# ndScene Summary
# sceneND { objectND { tensorND { dataND { tensor/compression/path } } } } 

def TodoND(desc=""):
    raise Exception("TODO_ND:" + desc)

"""Data used to back tensors, this data can be a readily availabe native 'tensor',
an uncompressed buffer, compressed MIME data, or an external path."""
class DataND():
    tensor = None
    buffer = None
    buffer_dtype : str = None
    buffer_size = None
    compressed = None
    compressed_type : str = None
    compressed_size = None
    path : str = None
    path_size : int = None


"""nestable/recursivly-defined tensor
"""
class TensorND():
    key :str = None
    size :int = None
    shape :list["TensorND"] = None
    dtype :str = None
    data :DataND = None

    def __init__(self, _size:int=None, _key:str=None):
        if (_size):
            self.size = _size
        if (_key):
            self.key = _key
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


"""n-dimensional scene graph element
"""
class ObjectND():
    key :str = None
    """Name/id of this ObjectND"""

    parents :list["ObjectND"] = []
    """Parent objects which this concatenates into"""

    children :list["ObjectND"] = []
    """Child object which concatenate into this one"""

    components :dict = {}
    """Generic components by key for extensbility"""

    pose :TensorND = None
    """Transform from local/child space to parent space"""

    unpose :TensorND = None
    """Transform to data/child space from local space"""

    data :TensorND = None
    """Content to be multipled by pose or encoded via unpose"""

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
    objects :list[ObjectND] = []
    paths :dict[str,ObjectND|TensorND] = {}

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
    def json_object(data:TensorND|DataND)->dict:
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
